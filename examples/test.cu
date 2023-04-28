#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;

__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  double beta = temp_sum;
  double temp;

  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp = tmp[cta.thread_rank() + i];
      beta += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
    beta = 0.0;
    for (int i = 0; i < cta.size(); i += tile32.size()) {
      beta += tmp[i];
    }
    outputVec[blockIdx.x] = beta;
  }
}

__global__ void reduceFinal(double *inputVec, double *result,
                            size_t inputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  // do reduction in shared mem
  if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
  }

  cg::sync(cta);

  if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
  }

  cg::sync(cta);

  if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 64];
  }

  cg::sync(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockDim.x >= 64) temp_sum += tmp[cta.thread_rank() + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      temp_sum += tile32.shfl_down(temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0) result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

double asyncReduce(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
  double result_h = 0.0;
  checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h,sizeof(float) * inputSize, cudaMemcpyDefault));
  checkCudaErrors(cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks));
  checkCudaErrors(cudaMemsetAsync(result_d, 0, sizeof(double)));

  reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);
  reduceFinal<<<1, THREADS_PER_BLOCK, 0>>>(outputVec_d, result_d, numOfBlocks);
  checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(double),cudaMemcpyDefault));
  //cudaDeviceSynchronize();
  //printf("async reduced sum = %lf\n", result_h);
  return result_h;
}

double syncReduce(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
  double result_h = 0.0;

  checkCudaErrors(cudaMemcpy(inputVec_d, inputVec_h,sizeof(float) * inputSize, cudaMemcpyDefault));
  checkCudaErrors(cudaMemset(outputVec_d, 0, sizeof(double) * numOfBlocks));
  checkCudaErrors(cudaMemset(result_d, 0, sizeof(double)));

  reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);
  reduceFinal<<<1, THREADS_PER_BLOCK, 0>>>(outputVec_d, result_d,numOfBlocks);
  checkCudaErrors(cudaMemcpy(&result_h, result_d, sizeof(double),cudaMemcpyDefault));

  //printf("sync reduced sum = %lf\n", result_h);
  return result_h;
}

void graphReduce(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
  cudaStream_t stream1;
  double result_h = 0.0;
  cudaGraph_t graph;
  checkCudaErrors(cudaStreamCreate(&stream1));
  checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
  checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h,
                                  sizeof(float) * inputSize, cudaMemcpyDefault,
                                  stream1));

  checkCudaErrors(cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream1));
  checkCudaErrors(cudaMemsetAsync(result_d, 0, sizeof(double), stream1));

  reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);
  reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d,numOfBlocks);
  checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(double),cudaMemcpyDefault, stream1));

  checkCudaErrors(cudaStreamEndCapture(stream1, &graph));
  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphLaunch(graphExec, stream1));

  printf("graph reduced sum = %lf\n", result_h);

  checkCudaErrors(cudaStreamDestroy(stream1));
}

int main(int argc, char **argv) {
  size_t size = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  // This will pick the best possible CUDA capable device
  int devID = findCudaDevice(argc, (const char **)argv);

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;

  checkCudaErrors(cudaMallocHost(&inputVec_h, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&inputVec_d, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&outputVec_d, sizeof(double) * maxBlocks));
  checkCudaErrors(cudaMalloc(&result_d, sizeof(double)));

  init_input(inputVec_h, size);

  // cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
  //                  maxBlocks);
  // double ans = asyncReduce(inputVec_h, inputVec_d, outputVec_d, result_d,
  //                              size, maxBlocks);
  double ans = syncReduce(inputVec_h, inputVec_d, outputVec_d, result_d,
                            size, maxBlocks);
  // graphReduce(inputVec_h, inputVec_d, outputVec_d, result_d,
  //                           size, maxBlocks);
  printf("reduced sum = %lf\n", ans);
  checkCudaErrors(cudaFree(inputVec_d));
  checkCudaErrors(cudaFree(outputVec_d));
  checkCudaErrors(cudaFree(result_d));
  checkCudaErrors(cudaFreeHost(inputVec_h));
  return EXIT_SUCCESS;
}
