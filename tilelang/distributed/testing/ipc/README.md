## Test of CUDA IPC (Inter-Process Communication)


### Usage
```bash
# build simple set-value kernel
cd ./test_kernel && python setup.py build_ext --inplace && cd - 
# build IPC test example
python setup.py build_ext --inplace
# Run example, 2 GPUs write to each other
python -m torch.distributed.run test_ipc.py --num-processes 2
```