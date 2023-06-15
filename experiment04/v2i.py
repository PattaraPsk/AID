import cv2
import os
import argparse
import sys
parser = argparse.ArgumentParser(description="v2i")
parser.add_argument('--path', type=str, help='file path')
args = parser.parse_args()
PATH = args.path

if PATH == None:
    PATH = "C:/Users/Ribuzari/Desktop/OurDataset"
already = []
print(PATH)
for f in os.listdir(PATH):
    if f[-4:] != ".mp4":
        already.append(f)
        continue
    if f[:-4] in already:
        print(f)
        # sys.exit()
        continue
    vid_path = os.path.join(PATH,f)
    print()
    cap = cv2.VideoCapture(vid_path)
    count = 0
    try:
        os.chdir(vid_path[:-4])
    except:
        os.mkdir(vid_path[:-4])
        os.chdir(vid_path[:-4])

    while 1:
        ret,frame = cap.read()
        if ret:
            cv2.imwrite(filename=str(count)+".jpg",img=frame)
            cv2.waitKey(1)
        else:
            break
        count += 1
    
