"""Verify GPU passthrough and genuine PyTorch CUDA computation."""
import torch

print("Torch:", torch.__version__)
print("Wheel CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "The NVIDIA GPU isn't exposed to Docker. Verify Docker Desktop WSL2 integration "
    "and that the container was launched with --gpus all / gpus: all."
)
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print("GPU:", name)
print("Compute capability:", capability)
if "5070" in name:
    assert capability >= (12, 0), "Expected a Blackwell GPU (sm_120)"
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = a @ b
torch.cuda.synchronize()
assert c.is_cuda and torch.isfinite(c).all().item()
print("SUCCESS: CUDA matrix multiply executed on the GPU.")
