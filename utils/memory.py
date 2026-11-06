import torch.cuda as cuda

def reset_memory_statistics():
    cuda.reset_peak_memory_stats()
    cuda.empty_cache()

def query_peak_memory_allocation():
    peak_bytes = cuda.max_memory_allocated()

    return _format_bytes(peak_bytes)

def _format_bytes(peak_bytes):
    peak_mebibytes = peak_bytes / (1024 * 1024)
    peak_gibibytes = peak_mebibytes / 1024
    
    return f'{peak_bytes}Bytes | {peak_mebibytes:.2f}MiB {peak_gibibytes:.2f}GiB'