BIN_FILE=test
SRC_FILE=$(BIN_FILE).cu

main: $(BIN_FILE)
$(BIN_FILE)-host-x86_64-unknown-linux-gnu.bc: $(SRC_FILE)
	clang++ test.cu -Xclang -disable-O0-optnone -I/home/kangadi6/cuda-samples/Common \
	--cuda-gpu-arch=sm_60 -L/usr/local/cuda/lib64/ -lcudart_static -ldl -lrt -pthread -save-temps=obj

$(BIN_FILE).bc:$(BIN_FILE)-host-x86_64-unknown-linux-gnu.bc
	opt -load-pass-plugin=/fast_data/kangadi6/llvm-project/build/lib/insertCugraph.so \
	-passes="insertCugraph" $(BIN_FILE)-host-x86_64-unknown-linux-gnu.bc -o $(BIN_FILE).bc

$(BIN_FILE).s: $(BIN_FILE).bc
	clang -cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -S -save-temps=obj  \
   	-main-file-name $(SRC_FILE) -target-cpu x86-64 -disable-O0-optnone \
   	-o $(BIN_FILE).s -x ir $(BIN_FILE).bc

$(BIN_FILE).o: $(BIN_FILE).s
	clang -cc1as -triple x86_64-unknown-linux-gnu -filetype obj -main-file-name $(SRC_FILE) \
	-target-cpu x86-64 -I /home/kangadi6/cuda-samples/Common -dwarf-version=5 -mrelocation-model pic -mrelax-all \
  	-o $(BIN_FILE).o $(BIN_FILE).s

$(BIN_FILE): $(BIN_FILE).o
	g++ -pie -z relro -o $(BIN_FILE) $(BIN_FILE).o \
	-L/usr/local/cuda/lib64/ \
	-lcudart_static -ldl -lrt -lstdc++ -lm -lgcc_s -lgcc -lpthread -lc -lgcc_s -lgcc

clean:
	rm *.s *.fatbin *.cui *.bc *.o *.cubin $(BIN_FILE)