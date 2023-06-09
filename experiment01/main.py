import utils

import torch

from torch.utils.tensorboard import SummaryWriter

import numpy as np
def exp01():
    log_dir = None
    comment = ""
    writer = SummaryWriter(
        log_dir=log_dir,
        comment=comment,
        )
    return 0



def main():
    log_dir = utils.getResDir()
    comment = "This is first test"
    writer = SummaryWriter(
        log_dir=log_dir,
        comment=comment,
        )
    name = 'First run'
    r = 5
    for i in range(100):
        k = i/r
        writer.add_scalars( 
            main_tag=name,
            tag_scalar_dict = {
                'line1': i*np.sin(k),
                'line2': i*np.cos(k),
                'line3': i*np.tan(k)
                },
            global_step=i
            )
    return 0
if __name__ == '__main__':
    main()