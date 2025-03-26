import ctypes

lib = ctypes.CDLL('./libnvshmem_example.so')

lib.run_simple_shift.restype = ctypes.c_int

def run_simple_shift():
    result = lib.run_simple_shift()
    print(f"message: {result}")
    return result

if __name__ == "__main__":
    run_simple_shift()