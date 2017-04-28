import cv2
import numpy as np
import time
import os

img = np.loadtxt('workspace/output/winner.csv', delimiter=',')
#score = np.loadtxt('workspace/score.csv')
#print score.shape
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] == 3:
            img[i][j] = 100
        elif img[i][j] == 4:
            img[i][j] = 150
        elif img[i][j] == 5:
            img[i][j] = 200
cv2.imwrite('workspace/output/winner.png', img)
#maximum = np.amax(score)
#score = 255.0/maximum*score
#cv2.imwrite('workspace/score.png', score)

vector = np.loadtxt('workspace/output/vector2.csv', delimiter=',')
img = cv2.imread('workspace/output/img.jpg')
ts = str(time.time())
os.system('cp workspace/output/vector.csv workspace/output/ctx-A/vector'+ts+'.csv')    
os.system('cp workspace/output/img.jpg workspace/output/ctx-A/raw_image'+ts+'.jpg')    
#img = cv2.resize(img, (512, 256))
for i in range(vector.shape[0]):
    #img[2*vector[i], 2*i, :] = (0, 0, 255)
    cv2.circle(img, (i, int(vector[i])), 1, (0, 0, 255), -1)
 

cv2.imwrite('workspace/output/vector.jpg', img)
os.system('cp workspace/output/vector.jpg workspace/output/ctx-A/vector'+ts+'.jpg')    

