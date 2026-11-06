import torch
import torch.cuda as cuda

def enable_cuda():
    device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    return device

def synchronize_gpu():
    if cuda.is_available():
        cuda.synchronize()

def check_gpu():
    print(f'CUDA available: {cuda.is_available()}')
    print(f'torch CUDA version: {torch.version.cuda}')
    if cuda.is_available():
        print(f'GPU device: {cuda.get_device_name(0)}')