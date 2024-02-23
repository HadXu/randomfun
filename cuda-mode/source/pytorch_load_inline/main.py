import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

module = torch.utils.cpp_extension.load(
    name="torch_test_cudnn_extension",
    sources=["cudnn-extension.cpp"],
    extra_ldflags=['-lcudnn'],
    verbose=True,
    with_cuda=True,
)

x = torch.randn(100, device="cuda", dtype=torch.float32)
y = torch.zeros(100, device="cuda", dtype=torch.float32)
module.cudnn_relu(x, y)  # y=relu(x)
assert torch.allclose( y, torch.nn.functional.relu(x) )