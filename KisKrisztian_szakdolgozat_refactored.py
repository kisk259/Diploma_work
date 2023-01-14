"""
Author: Kis Krisztián / R9Z50I
Diploma work title: Régiók többszörözésének észlelése a jellemzőpontok megfeleltetésének segítségével
Date: 2020
"""

import cv2
import numpy as np
import math
import time
from matplotlib import pyplot as plt


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        # Read in the image
        self.original_image = cv2.imread(self.image_path)
        # Original image parameters: Height, Width, channel
        self.h, self.w, self.c = self.original_image.shape

    def grayscale_conversion(self):
        """
        1. Step
        Convert the RGB image to grayscale
        """
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        return self.gray_image

    def sift_detection(self):
        """
        2. Step
        Detect the keypoints and descriptors in the image using SIFT
        """
        sift = cv2.SIFT_create(nfeatures=100000, nOctaveLayers=3, sigma=1.6)
        self.keypoints, self.descriptors = sift.detectAndCompute(self.gray_image, None)
        return self.keypoints, self.descriptors

    def draw_keypoints(self, keypoints, flag=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
        """
        2.1. Step
        (OPTIONAL)
        Draw out keypoints, FLAG is for vector size and direction
        Show image or save out in a file
        """
        img_keypoints_draw = self.original_image.copy()
        img_keypoints_draw = cv2.drawKeypoints(self.original_image, keypoints, None, flags=flag)
        cv2.imshow("result", img_keypoints_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('keypoints.png', img_keypoints_draw)

    def match_keypoints(self, descriptors):
        """
        3. Step
        Find the matches between the keypoints using BFMatcher
        """
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Brute-Force with KNN, K the number of best matches
        matches = self.bf.knnMatch(descriptors, descriptors, k=3)

        # Removing self matching points
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
        return good_matches

    def affine_transform(self, good_matches):
        """
        4. Step
        Apply affine transform to the keypoints using RANSAC
        """
        MIN_MATCH_COUNT = 3
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([self.keypoints[m.trainIdx].pt for m in good_matches])
            dst_pts = np.float32([self.keypoints[m.queryIdx].pt for m in good_matches])

            # Affine Transform matrix for forward transform matrix
            self.retval_forward, self.inliers_forward = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                                             ransacReprojThreshold=3, maxIters=100,
                                                                             confidence=0.99)
            # Affine Transform matrix for backward transform matrix
            self.retval_backward, self.inliers_backward = cv2.estimateAffine2D(dst_pts, src_pts, method=cv2.RANSAC,
                                                                               ransacReprojThreshold=3, maxIters=100,
                                                                               confidence=0.99)
            matches_mask = self.inliers_forward.ravel().tolist()
        else:
            print("Not enough keypoint matches")

        # Filter with RANSAC
        final_matches = []
        for i in range(len(good_matches)):
            if matches_mask[i] == 1:
                final_matches.append(good_matches[i])
        return self.retval_forward, self.retval_backward, final_matches

    def draw_lines_and_circles(self, final_matches):
        """
        4.1. Step
        Draw lines and circles on the image based on final matches
        """
        # Grayscale image convert to RGB image
        img_Fmatches = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2RGB)

        list_point1 = []
        list_point2 = []
        for j in final_matches:
            # Get the matching keypoints for each of the images
            point1 = j.trainIdx
            point2 = j.queryIdx

            # Get the coordinates, x - columns, y - rows
            (x1, y1) = self.keypoints[point1].pt
            (x2, y2) = self.keypoints[point2].pt

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

        return img_Fmatches

    def compute_correlation_maps(self, retval_forward, retval_backward):
        """
        5. Step
        Compute the region correlation maps based on forward and backward affine transform matrices
        """

        # WrapAffine for forward map
        wrapAffine_img_forward = cv2.warpAffine(self.gray_image, self.retval_forward, (self.w, self.h))

        # WrapAffine for backward map
        wrapAffine_img_backward = cv2.warpAffine(self.gray_image, self.retval_backward, (self.w, self.h))

        # Black images filled up with zeros
        forward_correlation_map = np.zeros((self.h, self.w, 1), np.float32)
        backward_correlation_map = np.zeros((self.h, self.w, 1), np.float32)

        # Computing region correlation map based on Forward affine transform matrix
        for y in range(0, self.h - 4):
            for x in range(0, self.w - 4):
                window1 = self.gray_image[y: y + 5, x: x + 5]
                window2 = wrapAffine_img_forward[y: y + 5, x: x + 5]

                I = window1 - window1.mean()
                W = window2 - window2.mean()

                top = np.sum(np.multiply(I, W, dtype=np.float32), dtype=np.float32)
                bottom = math.sqrt(
                    np.sum(np.multiply(np.multiply(I, I, dtype=np.float32), np.multiply(W, W, dtype=np.float32),
                                       dtype=np.float32), dtype=np.float32))

                if bottom != 0.0:
                    intensity = (top / bottom)
                    forward_correlation_map[y + 2, x + 2] = intensity
                elif bottom == 0.0:
                    intensity = 0
                    forward_correlation_map[y + 2, x + 2] = intensity

        # Computing region correlation map based on Backward affine transform matrix
        for y in range(0, self.h - 4):
            for x in range(0, self.w - 4):
                window1 = self.gray_image[y: y + 5, x: x + 5]
                window2 = wrapAffine_img_backward[y: y + 5, x: x + 5]

                I = window1 - window1.mean()
                W = window2 - window2.mean()

                top = np.sum(np.multiply(I, W, dtype=np.float32), dtype=np.float32)
                bottom = math.sqrt(
                    np.sum(np.multiply(np.multiply(I, I, dtype=np.float32), np.multiply(W, W, dtype=np.float32),
                                       dtype=np.float32), dtype=np.float32))

                if bottom != 0.0:
                    intensity = (top / bottom)
                    backward_correlation_map[y + 2, x + 2] = intensity
                elif bottom == 0.0:
                    intensity = 0
                    backward_correlation_map[y + 2, x + 2] = intensity

        return forward_correlation_map, backward_correlation_map

    def post_processing(self, forward_correlation_map, backward_correlation_map):
        """
        6. Step
        Post-processing
        """

        # Apply gaussian filter, kernel(7, 7)
        gaussian_forward = cv2.GaussianBlur(forward_correlation_map, (7, 7), 0)
        gaussian_backward = cv2.GaussianBlur(backward_correlation_map, (7, 7), 0)

        # Apply binary threshold
        # correlation map max intensity value
        minVal_f, maxVal_f, minLoc_f, maxLoc_f = cv2.minMaxLoc(forward_correlation_map)
        # print(maxVal_f)
        minVal_b, maxVal_b, minLoc_b, maxLoc_b = cv2.minMaxLoc(backward_correlation_map)
        # print(maxVal_b)

        # Threshold
        ret1, threshold_forward = cv2.threshold(gaussian_forward, (maxVal_f * 0.3), 255, cv2.THRESH_BINARY)
        ret2, threshold_backward = cv2.threshold(gaussian_backward, (maxVal_b * 0.3), 255, cv2.THRESH_BINARY)

        # Remove small isolated regions and close holes
        # Creating struck for morphological operations
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

        # Discard all small isolated regions
        # Apply the morphological open operation (erode->dilate)
        open_forward = cv2.morphologyEx(threshold_forward, cv2.MORPH_OPEN, struct, iterations=1)
        open_backward = cv2.morphologyEx(threshold_backward, cv2.MORPH_OPEN, struct, iterations=1)

        # Apply the morphological closing operation (dilate->erode)
        close_forward = cv2.morphologyEx(open_forward, cv2.MORPH_CLOSE, struct, iterations=1)
        close_backward = cv2.morphologyEx(open_backward, cv2.MORPH_CLOSE, struct, iterations=1)

        # Bitwise operation to get closer to the original shape
        bitwise_forward = cv2.bitwise_and(close_forward, threshold_forward)
        bitwise_backward = cv2.bitwise_and(close_backward, threshold_backward)
        # Close holes after Bitwise
        close_forward2 = cv2.morphologyEx(bitwise_forward, cv2.MORPH_CLOSE, struct, iterations=1)
        close_backward2 = cv2.morphologyEx(bitwise_backward, cv2.MORPH_CLOSE, struct, iterations=1)

        # Color the original image where correlate
        img_forward = self.original_image.copy()
        img_forward[close_forward2 > 0] = 0, 255, 255

        img_backward = img_forward.copy()
        img_backward[close_backward2 > 0] = 255, 0, 0

        return img_backward

    def output_image(self, forward_correlation_map, backward_correlation_map, img_Fmatches, img_backward):
        """
        7. Step
        Creating the final output image with the results
        """
        # Need to convert BGR to RGB for matplotlib
        img_BRG = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        img_detected = cv2.cvtColor(img_backward, cv2.COLOR_BGR2RGB)

        # Both of the correlation map in one image
        forward_correlation_map_show = cv2.cvtColor(forward_correlation_map, cv2.COLOR_GRAY2RGB)
        backward_correlation_map_show = cv2.cvtColor(backward_correlation_map, cv2.COLOR_GRAY2RGB)
        correlation_map = np.add(forward_correlation_map_show, backward_correlation_map_show, dtype=np.float32) * 255

        # Final output
        plt.subplot(2, 2, 1), plt.title("Original image", fontsize=10), plt.axis('off'), plt.imshow(img_BRG)
        plt.subplot(2, 2, 2), plt.title("Matched keypoints", fontsize=10), plt.axis('off'), plt.imshow(img_Fmatches)
        plt.subplot(2, 2, 3), plt.title("Correlation maps", fontsize=10), plt.axis('off'), plt.imshow(correlation_map)
        plt.subplot(2, 2, 4), plt.title("Detected regions", fontsize=10), plt.axis('off'), plt.imshow(img_detected)
        plt.tight_layout()
        plt.savefig('results/final_image.png')
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # Start timer
    start_time = time.perf_counter()

    # Original image path
    image_path = "040_F.png"
    img_processor = ImageProcessor(image_path)

    # 1. Step
    gray_image = img_processor.grayscale_conversion()

    # 2. Step
    keypoints, descriptors = img_processor.sift_detection()

    # Optional Step
    # img_processor.draw_keypoints(keypoints)

    # 3. Step
    good_matches = img_processor.match_keypoints(descriptors)

    # 4. Step
    retval_forward, retval_backward, final_matches = img_processor.affine_transform(good_matches)

    # 4.1. Step
    img_Fmatches = img_processor.draw_lines_and_circles(final_matches)

    # 5. Step
    forward_correlation_map, backward_correlation_map = img_processor.compute_correlation_maps(retval_forward,
                                                                                               retval_backward)

    # 6. Step
    img_backward = img_processor.post_processing(forward_correlation_map, backward_correlation_map)

    # Stop timer
    end_time = time.perf_counter()
    print("Time taken: ", end_time - start_time)

    # 7. Step
    img_processor.output_image(forward_correlation_map, backward_correlation_map, img_Fmatches, img_backward)


main()
