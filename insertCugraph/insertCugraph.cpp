#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

bool contains_kernel_call(Function *F) {
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        Function *callee = CI->getCalledFunction();
        if (callee && callee->getName().contains("cudaLaunchKernel")) {
          errs()<<"found kernel call contains_kernel_call\n";
          return true;
        } else if (callee && contains_kernel_call(callee)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool contains_sync(Function *F) {
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        Function *callee = CI->getCalledFunction();
        if (!callee->getName().contains("cudaLaunchKernel")) {
          if( callee && (!callee->getName().contains("cudaError")) && callee->getName().starts_with("cuda") &&
            !(callee->getName().contains("Async"))){
            errs()<<"found sync func ";
            CI->print(errs());
            errs()<<"\n";
            return true;
          } else if(callee && (!callee->getName().contains("cudaError")) && contains_sync(callee)){
              errs()<<"found sync func ";
              CI->print(errs());
              errs()<<"\n";
              return true;
          }
        }
      }
    }
  }
  return false;
}

bool contains_async(Function *F) {
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        Function *callee = CI->getCalledFunction();
        if( callee && callee->getName().contains("Async") && callee->getName().starts_with("cuda")){
          return true;
        } else if (callee && contains_async(callee)) {
          return true;
        }
      }
    }
  }
  return false;
}

void change_sync_to_async(Function &F){
  LLVMContext &ctx = F.getContext();
  Module *Mod = F.getParent();

  Type *Int32Ty = Type::getInt32Ty(ctx);
  Type *voidPtrTy = Type::getInt8PtrTy(ctx);
  Type *sizeTy = Type::getInt64Ty(ctx);
  Type *streamTy = Type::getInt8PtrTy(ctx);

  if(!Mod->getNamedGlobal("my_stream"))
  {
    errs()<<"declaring global stream variable\n";
    Mod->getOrInsertGlobal("my_stream", streamTy);
    auto g_var = Mod->getNamedGlobal("my_stream");
    g_var->setLinkage(GlobalValue::InternalLinkage);
    g_var->setInitializer(ConstantPointerNull::get(Type::getInt8PtrTy(ctx)));
    g_var->setAlignment(Align(8));
    g_var->setConstant(false);
  }

  if (F.getName() == "main")
  {
    BasicBlock &BB = F.getEntryBlock();
    IRBuilder<> builder(&BB, BB.begin());

    errs()<<"inserting cudaStreamCreate call\n";
    FunctionType *cudaStreamCreateType = FunctionType::get(Int32Ty, {streamTy}, false);
    auto cudaStreamCreate = Mod->getOrInsertFunction("cudaStreamCreate", cudaStreamCreateType);
    builder.CreateCall(cudaStreamCreate, {Mod->getNamedGlobal("my_stream")}, "cuStreamCreate");
  }

  SmallVector<CallInst *, 16> instToDelete;
  // Iterate through the instructions in the function and replace cudaMalloc calls
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (CallInst *callInst = dyn_cast<CallInst>(&I)) {
        Function *callee = callInst->getCalledFunction();
        if (callee && callee->getName() == "cudaMalloc") {
          FunctionType *mallocAsyncType = FunctionType::get(Int32Ty, {voidPtrTy, sizeTy, streamTy}, false);
          auto mallocAsync = Mod->getOrInsertFunction("cudaMallocAsync", mallocAsyncType);

          auto streamVal = new LoadInst(streamTy, Mod->getNamedGlobal("my_stream"), "", false, callInst);
          errs()<<"replacing cudaMalloc call\n";
          CallInst *asyncCall = CallInst::Create(mallocAsync,
                                {callInst->getArgOperand(0), callInst->getArgOperand(1), streamVal}, "", callInst);
          if (!callInst->use_empty())
            callInst->replaceAllUsesWith(asyncCall);
          instToDelete.push_back(callInst);
        }

        if (callee && callee->getName() == "cudaMemset"){
          errs()<<"updating memset calls\n";
          std::vector<Value *> Args;
          std::vector<Type *> ArgType;

          for (unsigned int i = 0; i < callInst->arg_size(); i++) {
            Args.push_back(callInst->getArgOperand(i));
            ArgType.push_back(callInst->getArgOperand(i)->getType());
          }
          auto streamVal = new LoadInst(streamTy, Mod->getNamedGlobal("my_stream"), "", false, callInst);
          Args.push_back(streamVal);
          ArgType.push_back(streamTy);

          FunctionType *memSetAsyncTy = FunctionType::get(callee->getReturnType(), ArgType, false);
          auto memSetAsync = Mod->getOrInsertFunction("cudaMemsetAsync", memSetAsyncTy);
          CallInst *NewCI = CallInst::Create(memSetAsync, Args,"", callInst);
          callInst->replaceAllUsesWith(NewCI);
          instToDelete.push_back(callInst);
        }

        if (callee && callee->getName() == "cudaMemcpy"){
          errs()<<"updating cudaMemcpy calls\n";
          std::vector<Value *> Args;
          std::vector<Type *> ArgType;

          for (unsigned int i = 0; i < callInst->arg_size(); i++) {
            Args.push_back(callInst->getArgOperand(i));
            ArgType.push_back(callInst->getArgOperand(i)->getType());
          }
          auto streamVal = new LoadInst(streamTy, Mod->getNamedGlobal("my_stream"), "", false, callInst);
          Args.push_back(streamVal);
          ArgType.push_back(streamTy);

          FunctionType *memCpyAsyncTy = FunctionType::get(callee->getReturnType(), ArgType, false);
          auto memcpyAsync = Mod->getOrInsertFunction("cudaMemcpyAsync", memCpyAsyncTy);
          CallInst *NewCI = CallInst::Create(memcpyAsync, Args,"", callInst);
          callInst->replaceAllUsesWith(NewCI);
          instToDelete.push_back(callInst);
        }

        if (callee && callee->getName() == "__cudaPushCallConfiguration") {
          errs()<<"updating kernel call with stream variable\n";
          auto streamVal = new LoadInst(streamTy, Mod->getNamedGlobal("my_stream"), "", false, callInst);
          callInst->setOperand(callInst->arg_size()-1 , streamVal);
        }
      }
    }
  }
  for(auto I:instToDelete)
    I->eraseFromParent();
}


struct insertCugraph : PassInfoMixin<insertCugraph> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {

    change_sync_to_async(F);

    if (F.hasName())
	    errs()<<"insertCugraph pass "<<F.getName()<<"\n";
    else
	    errs()<<"function has no name\n";

    LLVMContext &ctx = F.getContext();
    Module *Mod = F.getParent();

    if (F.getName() == "main"){
      Instruction *start_capture = nullptr, *end_capture = nullptr;
      bool kernel_found = false;

      for (auto &BB:F){
        for (auto &I:BB){
          if(auto callInst = dyn_cast<CallInst>(&I)){
            callInst->print(errs());
            errs()<<"\n";
            //errs()<<"call inst "<<callInst->print()<<"\n";
            Function *callee = callInst->getCalledFunction();
            if((callee && callee->getName().contains("cudaLaunchKernel")) ||
               (contains_kernel_call(callee) && !contains_sync(callee))){
              errs()<<"found a kernel call\n";
              kernel_found = true;
              if(nullptr == start_capture){
                start_capture = &I;
                errs()<<"start capture at "<<callee->getName()<<"\n";
              }else{
                 end_capture = &I;
                 errs()<<"end capture after "<<callee->getName()<<"\n";
              }
            }
            else if(( callee && callee->getName().starts_with("cuda") &&
                     callee->getName().contains("Async")) || (!contains_sync(callee) && contains_async(callee))){
              errs()<<"found async call\n";
              if(nullptr == start_capture){
                start_capture = &I;
                errs()<<"start capture at "<<I.getName()<<"\n";
              }else{
                 end_capture = &I;
                 errs()<<"end capture at "<<I.getName()<<"\n";
              }
            }
          }
        }
      }
      /*
      cudaGraph_t graph;
      cudaGraphExec_t instance;
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

      cudaStreamEndCapture(stream, &graph);
      cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
      cudaGraphLaunch(instance, stream);
      cudaStreamSynchronize(stream);

      */
      if (end_capture && kernel_found){
        Type* graphTy = Type::getInt8PtrTy(ctx);
        Type* graphExecTy = Type::getInt8PtrTy(ctx);
        Type *Int32Ty = Type::getInt32Ty(ctx);
        Type *streamTy = Type::getInt8PtrTy(ctx);

        IRBuilder<> builder(ctx);
        builder.SetInsertPoint(start_capture);

        auto graphTyAlloc = builder.CreateAlloca(graphTy, nullptr, "graphTy");

        auto graphExecTyAlloc = builder.CreateAlloca(graphExecTy, nullptr, "graphExecTy");

        auto streamVal = builder.CreateLoad(streamTy, Mod->getNamedGlobal("my_stream"), "streamVal");

        FunctionType *cudaStreamBeginCaptureType = FunctionType::get(Int32Ty, {streamTy, Int32Ty}, false);
        auto cudaStreamBeginCapture = Mod->getOrInsertFunction("cudaStreamBeginCapture", cudaStreamBeginCaptureType);
        builder.CreateCall(cudaStreamBeginCapture, {streamVal, ConstantInt::get(Int32Ty, 0)}, "");


        builder.SetInsertPoint(end_capture->getNextNode());

        FunctionType *cudaStreamEndCaptureType = FunctionType::get(Int32Ty, {streamTy, graphTy}, false);
        auto cudaStreamEndCapture = Mod->getOrInsertFunction("cudaStreamEndCapture", cudaStreamEndCaptureType);
        builder.CreateCall(cudaStreamEndCapture, {streamVal, graphTyAlloc}, "");

        LoadInst *LoadGraphTy = builder.CreateLoad(graphTy, graphTyAlloc, "loadgraph");
        FunctionType *cudaGraphInstantiateType = FunctionType::get(Int32Ty, {graphExecTy, graphTy, Int32Ty}, false);
        auto cudaGraphInstantiate = Mod->getOrInsertFunction("cudaGraphInstantiate", cudaGraphInstantiateType);
        builder.CreateCall(cudaGraphInstantiate, {graphExecTyAlloc, LoadGraphTy, ConstantInt::get(Int32Ty, 0)}, "");

        LoadInst *LoadGraphExecTy = builder.CreateLoad(graphExecTy, graphExecTyAlloc, "loadGraphExec");
        FunctionType *cudaGraphLaunchType = FunctionType::get(Int32Ty, {graphExecTy, streamTy}, false);
        auto cudaGraphLaunch = Mod->getOrInsertFunction("cudaGraphLaunch", cudaGraphLaunchType);
        builder.CreateCall(cudaGraphLaunch, {LoadGraphExecTy, streamVal}, "");

        FunctionType *cudaStreamSynchronizeType = FunctionType::get(Int32Ty, {streamTy}, false);
        auto cudaStreamSynchronize = Mod->getOrInsertFunction("cudaStreamSynchronize", cudaStreamSynchronizeType);
        builder.CreateCall(cudaStreamSynchronize, {streamVal}, "");
      }
    }
  return PreservedAnalyses::all();
  }
};

} // namespace

/* New PM Registration */
llvm::PassPluginLibraryInfo getCugraphPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Cugraph", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, llvm::FunctionPassManager &PM,
                   ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "insertCugraph") {
                    PM.addPass(insertCugraph());
                    return true;
                  }
                  return false;
                });
          }};
}

#ifndef LLVM_BYE_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getCugraphPluginInfo();
}
#endif
