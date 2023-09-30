import cv2
import numpy as np

# Load images
img1 = cv2.imread("img4.jpg")
img5 = cv2.imread("img5.jpg")

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp5, des5 = sift.detectAndCompute(img5, None)

# FLANN parameters for feature matching
flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)

# Create FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match features using FLANN
matches = flann.knnMatch(des1, des5, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for _ in range(len(matches))]

# Ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

# Draw matches
img_matches = cv2.drawMatchesKnn(img1, kp1, img5, kp5, matches, None, **draw_params)

# Use good matches to estimate the homography using RANSAC
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

print(len(good_matches))
if len(good_matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp5[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Warp img1 to img5 using the estimated homography
    h, w = img1.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, M, (w, h))

    # Resize img5 to match the dimensions of warped_img1
    img5_resized = cv2.resize(img5, (warped_img1.shape[1], warped_img1.shape[0]))

    # Perform stitching with resized img5
    stitched_img = np.where(warped_img1 == 0, img5_resized, warped_img1)


    # Display the stitched image
    cv2.imshow("Stitched Image", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Not enough good matches to compute homography.")
