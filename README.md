1. Clone llvm project : git clone https://github.com/llvm/llvm-project.git
2. Copy insertCugraph folder to llvm-project/llvm/examples/insertCugraph
3. build the project:(mkdir build) cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_PROJECTS="clang;lldb"
4. (cd build) cmake --build . -j4
5. verify llvm-project/build/lib/insertCugraph.so exists
6. to compile and run a cuda program go to examples and make (this generates test executable)
7. ./test
