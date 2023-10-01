import cv2
import numpy as np

img1 = cv2.imread("img4.jpg")
img5 = cv2.imread("img5.jpg")

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp5, des5 = sift.detectAndCompute(img5, None)

flann_index_kdtree = 1
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des5, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

MIN_MATCH_COUNT = 4

if len(good_matches) >= MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp5[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    num_iterations = 1000
    inlier_threshold = 5.0

    best_inliers = 0
    best_homography = None

    for _ in range(num_iterations):
        random_indices = np.random.choice(len(src_pts), 4, replace=False)
        random_src_pts = np.squeeze(src_pts[random_indices])
        random_dst_pts = np.squeeze(dst_pts[random_indices])

        homography, _ = cv2.findHomography(random_src_pts, random_dst_pts, cv2.RANSAC, inlier_threshold)
        transformed_pts = np.matmul(homography, np.hstack((random_src_pts, np.ones((4, 1)))).T).T
        distances = np.linalg.norm(transformed_pts[:, :2] / transformed_pts[:, 2, np.newaxis] - random_dst_pts, axis=1)
        inliers = np.sum(distances < inlier_threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_homography = homography

    h, w = img1.shape[:2]
    warped_img1 = cv2.warpPerspective(img1, best_homography, (w, h))
    img5_resized = cv2.resize(img5, (warped_img1.shape[1], warped_img1.shape[0]))
    stitched_img = np.where(warped_img1 == 0, img5_resized, warped_img1)

    cv2.imshow("Stitched Image", stitched_img)
    cv2.imwrite("StitchedImage.jpg", stitched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Not enough good matches to compute homography.")
