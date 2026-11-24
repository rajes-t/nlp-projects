import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

print('PyTorch Version:', torch.__version__)
print('CUDA Build Version:', torch.version.cuda)
print('CUDA Built:', torch.backends.cuda.is_built())
print('CUDA Available:', torch.cuda.is_available())