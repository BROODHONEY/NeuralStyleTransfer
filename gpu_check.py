import torch

print(torch.version.cuda)             # should NOT be None
print(torch.cuda.is_available())      # should be True
print(torch.cuda.get_device_name(0))  # your GPU name   