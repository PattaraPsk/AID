def getResDir():
    import os
    from datetime import datetime
    curr_work_dir = os.path.dirname(__file__)
    resDir = os.path.join(curr_work_dir,'results',datetime.now().strftime("%b%d_%H-%M-%S"))
    # print(resDir)
    return resDir