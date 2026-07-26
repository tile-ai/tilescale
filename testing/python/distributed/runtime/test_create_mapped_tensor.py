import gc
import weakref

import pytest
import torch
import tilelang.testing
from tilelang.distributed.shared_memory import create_host_device_tensor


@tilelang.testing.requires_cuda
def test_create_host_device_tensor():
    shape = (1024, 1024)
    dtype = torch.float32
    host_tensor, device_tensor = create_host_device_tensor(shape, dtype)

    # test meta-data
    assert device_tensor.device.type == "cuda"
    assert device_tensor.shape == shape, f"{device_tensor.shape=}"
    assert device_tensor.dtype == dtype, f"{device_tensor.dtype=}"
    assert torch.equal(host_tensor, device_tensor.cpu()), f"{host_tensor=}, {device_tensor=}"

    # test modification
    device_tensor.random_()
    assert torch.equal(host_tensor, device_tensor.cpu()), f"{host_tensor=}, {device_tensor=}"

    owner_ref = weakref.ref(host_tensor.untyped_storage()._tilelang_managed_allocation)
    host_alias = host_tensor.detach().reshape(-1)
    device_alias = device_tensor.detach().reshape(-1)
    del host_tensor, device_tensor
    gc.collect()

    assert owner_ref() is not None
    device_alias.fill_(7)
    torch.cuda.synchronize()
    assert torch.all(host_alias == 7)

    del host_alias
    gc.collect()
    assert owner_ref() is not None
    del device_alias
    gc.collect()
    assert owner_ref() is None


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("shape", [(0,), (-1,), (2, 0)])
def test_create_host_device_tensor_rejects_nonpositive_dimensions(shape):
    with pytest.raises(ValueError, match="dimensions must be positive"):
        create_host_device_tensor(shape, torch.float32)


if __name__ == "__main__":
    tilelang.testing.main()
