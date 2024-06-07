import gc
import torch


# seeder
def torch_device_seed(seed_num):
    # seed the device if available
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    if is_cuda:
        torch.cuda.manual_seed(seed_num)
    elif is_mps:
        torch.mps.manual_seed(seed_num)
    else:
        torch.manual_seed(seed_num)


# check gpu
def check_gpu():
    # usage = device = check_gpu()
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        if is_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print("GPU not available, CPU used")
    return device


# clear cache
def clear_device_cache():
    # clear cache from the device calculations are running in
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()

    if is_cuda:
        torch.cuda.empty_cache()
    elif is_mps:
        torch.mps.empty_cache()
    
    gc.collect()
