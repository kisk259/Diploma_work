import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

SOURCE_IMAGE=sys.argv[1]
img = cv2.imread(SOURCE_IMAGE)

width = img.shape[0]
height = img.shape[1]
channels = img.shape[2]

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

akaze = cv2.AKAZE_create()
keypoints = akaze.detect(gray_img, None)
keypoints, descriptors = akaze.compute(gray_img, keypoints)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary


matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches = matcher.knnMatch(descriptors, descriptors, 2)

#good = []
#for m,n in matches:
#    if m.distance < 0.7*n.distance:
#        good.append(m)

#print(matches)
good = []
counter = 0
matchImg = img;
for m,n in matches:
    print('hello')
    k1 = keypoints[m.trainIdx];
    k2 = keypoints[m.queryIdx];
    print (k1.pt)
    print (k2.pt)
    if k1 == k2:
      continue;
    print (k1.pt)
    print (k2.pt)
    cv2.line(matchImg, (int(k1.pt[0]),int(k1.pt[1])), (int(k2.pt[0]),int(k2.pt[1])), (0,255,0), 2)
    print(counter)
    if counter == 100:
      break;
    counter +=1

outImg	=	cv2.drawMatchesKnn(	gray_img, keypoints, gray_img, keypoints, matches[:10], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS )
#out_img = cv2.drawKeypoints(gray_img, keypoints, descriptors, color=(255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(1)
#plt.imshow(outImg)
plt.imshow(matchImg)
plt.show()
