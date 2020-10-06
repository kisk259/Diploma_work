import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

# 1) Read in the image
img = cv2.imread("original_golden_bridge.jpg")
img_gray = cv2.imread("original_golden_bridge.jpg", cv2.IMREAD_GRAYSCALE)
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Initiate SIFT detector (nfeatures = "keypoints number")
sift = cv2.SIFT_create(nfeatures=100000)
# find the keypoints and descriptors with SIFT
kp_1, desc_1 = sift.detectAndCompute(img, None)

# Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(desc_1, desc_1, k=3)

# Self matching points remove
better_matches = []
for a, b, c in matches:
    if a.trainIdx == a.queryIdx:
        better_matches.append([b, c])
    elif b.trainIdx == b.queryIdx:
        better_matches.append([a, c])
    elif c.trainIdx == c.queryIdx:
        better_matches.append([a, b])

# Apply ratio test
ratio_thresh = 0.5
good_matches = []
for m, n in better_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# Affine Transform with RANSAC
if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp_1[m.trainIdx].pt for m in good_matches])
    dst_pts = np.float32([kp_1[m.queryIdx].pt for m in good_matches])

    retval, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=100,
                                           confidence=0.99)

    matchesMask = inliers.ravel().tolist()

# Filter with RANSAC
final_matches = []

for i in range(len(good_matches)):
    if matchesMask[i] == 1:
        final_matches.append(good_matches[i])

# Draw Lines and circles on the image
list_point1 = []
list_point2 = []

for j in final_matches:
    # Get the matching keypoints for each of the images
    point1 = j.trainIdx
    point2 = j.queryIdx

    # Get the coordinates, x - columns, y - rows
    (x1, y1) = kp_1[point1].pt
    (x2, y2) = kp_1[point2].pt

    # Append to each list
    list_point1.append((int(x1), int(y1)))
    list_point2.append((int(x2), int(y2)))

    # # Draw a small circle at both co-ordinates: radius 4, colour green, thickness = 1
    # # copy keypoints circles
    # cv2.circle(img, (int(x1), int(y1)), 4, (0, 255, 0), 1)
    # # original keypoints circles
    # cv2.circle(img, (int(x2), int(y2)), 4, (0, 255, 0), 1)
    #
    # # Draw a line in between the two points, thickness = 1, colour green
    # img3 = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

# Template image create
# Original image parameters: Height, Width, channel
h, w, c = img.shape

# Black image filled up with zeros
blank_image = np.zeros((h, w, 3), np.uint8)

# 5x5 kernel filled up with ones
kernel = np.ones((5, 5), np.uint8)

for e in range(0, len(list_point2)):
    x = list_point2[e][0]
    y = list_point2[e][1]
    blank_image[y][x] = 1, 1, 1
    blackwhite_img = cv2.dilate(blank_image, kernel, iterations=15)

template = cv2.multiply(blackwhite_img, img)
temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# WrapAffine
wrapAffine_img = cv2.warpAffine(img_gray, retval, (w, h))

# Template matching for Region Correlation Map
img_correlation_map = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCORR)
print(img_correlation_map)

threshold = 0.8
loc = np.where(img_correlation_map >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
cv2.imwrite('res.png', img)


print(img.size)
print(len(good_matches))
print(len(final_matches))

########################################################################################################################

# print(good_matches) only work with 2 img output
# img3 = cv2.drawMatchesKnn(img,kp_1,img,kp_1,matches,None, flags=2)
# img3 = cv2.drawMatches(img,kp_1,img,kp_1,good_matches,None, flags=2)

# plot img but it's negativ img
# plt.imshow(img3),plt.show()

########################################################################################################################

# plt.imshow(img, interpolation = 'bicubic'),plt.show()
# 3) Output
cv2.imshow("result", img_correlation_map)
#cv2.imshow("corelation", img_correlation_map)
cv2.waitKey(0)
cv2.destroyAllWindows()