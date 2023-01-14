# Készítette: Kis Krisztián / R9Z50I
# Szakdolgozat címe: Régiók többszörözésének észlelése a jellemzőpontok megfeleltetésének segítségével
# Szakdolgozat elkészítésének éve: 2020

import cv2
import numpy as np
import math
import time
from matplotlib import pyplot as plt

start_time = time.perf_counter()

# 1) Read in the image
img = cv2.imread("040_F.png")

# Original image parameters: Height, Width, channel
h, w, c = img.shape

# 2) RGB image convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3) Initiate SIFT detector (nfeatures = "keypoints number")
sift = cv2.SIFT_create(nfeatures=100000, nOctaveLayers=3, sigma=1.6)
# 4) Find the keypoints and descriptors with SIFT
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# # Number of the keypoints
# print(len(keypoints))
#
# 4.1) Draw out keypoints, FLAG is for vector size and direction (optional)
# img_keypoints_draw = img.copy()
# img_keypoints_draw = cv2.drawKeypoints(img_gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# # 4.2) Show image or save out in a file (optional)
# cv2.imshow("result", img_keypoints_draw)
# cv2.imwrite('keypoints.png', img_keypoints_draw)

# 5) Create Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# 5.1) Brute-Force with KNN, K the number of best matches
matches = bf.knnMatch(descriptors, descriptors, k=3)

# 5.2) Removing self matching points
better_matches = []
for a, b, c in matches:
    if a.trainIdx == a.queryIdx:
        better_matches.append([b, c])
    elif b.trainIdx == b.queryIdx:
        better_matches.append([a, c])
    elif c.trainIdx == c.queryIdx:
        better_matches.append([a, b])

# 5.3) Apply ratio test
ratio_thresh = 0.5
good_matches = []
for m, n in better_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# 5.4) Affine Transform with RANSAC
MIN_MATCH_COUNT = 3
if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches])

# Affine Transform matrix for forward transform matrix
    retval_forward, inliers_forward = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=100,
                                                           confidence=0.99)
# Affine Transform matrix for backward transform matrix
    retval_backward, inliers_backward = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=100,
                                                             confidence=0.99)

    matchesMask = inliers_forward.ravel().tolist()
else:
    print("Not enough keypoint matches")

# 5.5) Filter with RANSAC
final_matches = []
for i in range(len(good_matches)):
    if matchesMask[i] == 1:
        final_matches.append(good_matches[i])

# # Number of the final matched keypoints
# print(len(final_matches))

# 5.3.1) Draw Lines and circles on the image (optional)
# Grayscale image convert to RGB image
img_Fmatches = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

list_point1 = []
list_point2 = []
for j in final_matches:

    # Get the matching keypoints for each of the images
    point1 = j.trainIdx
    point2 = j.queryIdx

    # Get the coordinates, x - columns, y - rows
    (x1, y1) = keypoints[point1].pt
    (x2, y2) = keypoints[point2].pt

    # Append to each list
    list_point1.append((int(x1), int(y1)))
    list_point2.append((int(x2), int(y2)))

    # Draw a small circle at both co-ordinates: radius 4, colour green, thickness = 1
    # copy keypoints circles
    cv2.circle(img_Fmatches, (int(x1), int(y1)), 4, (0, 255, 0), 1)
    # original keypoints circles
    cv2.circle(img_Fmatches, (int(x2), int(y2)), 4, (0, 255, 0), 1)

    # Draw a line in between the two points, thickness = 1, colour green
    cv2.line(img_Fmatches, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)


# 6) Computing region correlation maps
# 6.1) WrapAffine for forward map
wrapAffine_img_forward = cv2.warpAffine(img_gray, retval_forward, (w, h))
# cv2.imwrite('wrapAffine_img_forward.png', wrapAffine_img_forward)

# 6.1) WrapAffine for backward map
wrapAffine_img_backward = cv2.warpAffine(img_gray, retval_backward, (w, h))
# cv2.imwrite('wrapAffine_img_backward.png', wrapAffine_img_backward)

# 6.2) Black images filled up with zeros
forward_correlation_map = np.zeros((h, w, 1), np.float32)
backward_correlation_map = np.zeros((h, w, 1), np.float32)

# 6.3.1) Computing region correlation map based on Forward affine transform matrix
for y in range(0, h - 4):
    for x in range(0, w - 4):
        window1 = img_gray[y: y + 5, x: x + 5]
        window2 = wrapAffine_img_forward[y: y + 5, x: x + 5]

        I = window1 - window1.mean()
        W = window2 - window2.mean()

        top = np.sum(np.multiply(I, W, dtype=np.float32), dtype=np.float32)
        bottom = math.sqrt(np.sum(np.multiply(np.multiply(I, I, dtype=np.float32), np.multiply(W, W, dtype=np.float32),
                                              dtype=np.float32), dtype=np.float32))

        if bottom != 0.0:
            intensity = (top / bottom)
            forward_correlation_map[y + 2, x + 2] = intensity
        elif bottom == 0.0:
            intensity = 0
            forward_correlation_map[y + 2, x + 2] = intensity

# 6.3.2) Computing region correlation map based on Backward affine transform matrix
for y in range(0, h-4):
    for x in range(0, w-4):
        window1 = img_gray[y: y+5, x: x+5]
        window2 = wrapAffine_img_backward[y: y + 5, x: x + 5]

        I = window1 - window1.mean()
        W = window2 - window2.mean()

        top = np.sum(np.multiply(I, W, dtype=np.float32), dtype=np.float32)
        bottom = math.sqrt(np.sum(np.multiply(np.multiply(I, I, dtype=np.float32), np.multiply(W, W, dtype=np.float32),
                                              dtype=np.float32), dtype=np.float32))

        if bottom != 0.0:
            intensity = (top / bottom)
            backward_correlation_map[y + 2, x + 2] = intensity
        elif bottom == 0.0:
            intensity = 0
            backward_correlation_map[y + 2, x + 2] = intensity

# 7) Post processing
# 7.1) Apply gaussian filter, kernel(7, 7)
gaussian_forward = cv2.GaussianBlur(forward_correlation_map, (7, 7), 0)
gaussian_backward = cv2.GaussianBlur(backward_correlation_map, (7, 7), 0)

# 7.2) Apply binary threshold
# 7.2.1) correlation map max intensity value
minVal_f, maxVal_f, minLoc_f, maxLoc_f = cv2.minMaxLoc(forward_correlation_map)
# print(maxVal_f)
minVal_b, maxVal_b, minLoc_b, maxLoc_b = cv2.minMaxLoc(backward_correlation_map)
# print(maxVal_b)

# 7.2.2) Threshold
ret1, threshold_forward = cv2.threshold(gaussian_forward, (maxVal_f * 0.3), 255, cv2.THRESH_BINARY)
ret2, threshold_backward = cv2.threshold(gaussian_backward, (maxVal_b * 0.3), 255, cv2.THRESH_BINARY)

# 7.3) Remove small isolated regions and close holes
# 7.3.1) Creating struck for morphological operations
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

# 7.3.2) Discard all small isolated regions
# Apply the morphological open operation (erode->dilate)
open_forward = cv2.morphologyEx(threshold_forward, cv2.MORPH_OPEN, struct, iterations=1)
open_backward = cv2.morphologyEx(threshold_backward, cv2.MORPH_OPEN, struct, iterations=1)

# 7.3.3) Apply the morphological closing operation (dilate->erode)
close_forward = cv2.morphologyEx(open_forward, cv2.MORPH_CLOSE, struct, iterations=1)
close_backward = cv2.morphologyEx(open_backward, cv2.MORPH_CLOSE, struct, iterations=1)

# Bitwise operation to get closer to the original shape
bitwise_forward = cv2.bitwise_and(close_forward, threshold_forward)
bitwise_backward = cv2.bitwise_and(close_backward, threshold_backward)
# Close holes after Bitwise
close_forward2 = cv2.morphologyEx(bitwise_forward, cv2.MORPH_CLOSE, struct, iterations=1)
close_backward2 = cv2.morphologyEx(bitwise_backward, cv2.MORPH_CLOSE, struct, iterations=1)

# 7.4) Color the original image where correlate
img_forward = img.copy()
img_forward[close_forward2 > 0] = 0, 255, 255

img_backward = img_forward.copy()
img_backward[close_backward2 > 0] = 255, 0, 0

# Program runtime
end_time = time.perf_counter()
print((end_time - start_time) * 1000.0, "ezredmásodperc.")

# 8) Output
# 8.1) Need to convert BGR to RGB for matplotlib
img_BRG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_detected = cv2.cvtColor(img_backward, cv2.COLOR_BGR2RGB)

# 8.2) Both correlation map in one image
forward_correlation_map_show = cv2.cvtColor(forward_correlation_map, cv2.COLOR_GRAY2RGB)
backward_correlation_map_show = cv2.cvtColor(backward_correlation_map, cv2.COLOR_GRAY2RGB)
correlation_map = np.add(forward_correlation_map_show, backward_correlation_map_show, dtype=np.float32)*255

# 8.3) Final output
plt.subplot(2, 2, 1), plt.title("Original image", fontsize=10), plt.axis('off'), plt.imshow(img_BRG)
plt.subplot(2, 2, 2), plt.title("Matched keypoints", fontsize=10), plt.axis('off'), plt.imshow(img_Fmatches)
plt.subplot(2, 2, 3), plt.title("Correlation maps", fontsize=10), plt.axis('off'), plt.imshow(correlation_map)
plt.subplot(2, 2, 4), plt.title("Detected regions", fontsize=10), plt.axis('off'), plt.imshow(img_detected)
plt.tight_layout()
plt.savefig('final_image.png')
plt.show()
# cv2.imwrite("scale.png", img_backward)
cv2.waitKey(0)
cv2.destroyAllWindows()