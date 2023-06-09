import torch
def cudaCheck():
    if torch.cuda.is_available():
        print('CUDA avaliable on this system.')
        print('CUDA version: ',torch.version.cuda)

