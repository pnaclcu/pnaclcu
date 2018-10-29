import cv2
from math import exp
import os
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pic',type=str,help='picture')
args=parser.parse_args()
def softmax(index,list):
    deno=0
    for i in range(len(list)):
        deno+=exp(list[i])
    return (exp(list[index])/deno)
ticks=time.time()
dir_name=[]
logo_name=[]
for root, dirs, files in os.walk("/home/sdu/车标/", topdown=False):
    for name in dirs:
        logo_name.append(name)
        dir_name.append(os.path.join(root, name))
img1_gray = cv2.imread(args.pic)
sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.xfeatures2d.SURF_create(2000)
kp1, des1 = sift.detectAndCompute(img1_gray, None)

brand = [0]*len(dir_name)

for i in range(len(dir_name)):


    for img in os.listdir(dir_name[i]):

        img=os.path.join(dir_name[i]+'/'+img)
        img2_gray = cv2.imread(img)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k = 2)
        goodMatch = []
        hardMatch = []
        for m,n in matches:
            if m.distance < 0.70*n.distance:
                goodMatch.append(m)
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                hardMatch.append(m)
        if len(goodMatch)>=3:
            brand[i]+=1
        if len(hardMatch)>=3:
            brand[i]+=5

result=brand.index(max(brand))
print (logo_name[result],'概率',softmax(result,brand))

ticks1=time.time()
print ('耗时',ticks1-ticks)
