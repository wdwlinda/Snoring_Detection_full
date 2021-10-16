
import torch

def channel_fusion(input_data, method, dim=0, reapeat_times=1):
    if len(input_data.size()) == 2:
        input_data = torch.unsqueeze(input_data, dim)
    if method == 'mean':
        fused_data = torch.mean(input_data, dim, keepdim=True)
    elif method == 'duplicate':
        fused_data = torch.mean(input_data, dim, keepdim=True)
        repeats = [1, 1, 1]
        repeats[dim] = reapeat_times
        fused_data = input_data.repeat(*repeats)
    else:
        raise ValueError('Unknown fusion method.')
    return fused_data