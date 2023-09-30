import cv2
import numpy as np

# Lists to store points
architectural_points = []
flag_points = []


# Mouse callback function to handle mouse events for point selection
def select_points(event, x, y, flags, param):
    global architectural_points, flag_points
    if event == cv2.EVENT_LBUTTONDOWN and len(architectural_points) != 4:
        architectural_points.append((x, y))
        cv2.circle(architectural_image_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Architectural Image", architectural_image_display)
        if len(architectural_points) == 4:
            print("Architectural Points Selected:", architectural_points)
            flag_points = []
            cv2.destroyWindow("Architectural Image")
            #cv2.imshow("Flag Image", flag_image_display)

    elif event == cv2.EVENT_LBUTTONDOWN and len(architectural_points) == 4:
        flag_points.append((x, y))
        cv2.circle(flag_image_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Flag Image", flag_image_display)
        if len(flag_points) == 4:
            print("Flag Points Selected:", flag_points)
            cv2.destroyWindow("Flag Image")


# Read architectural and flag images
architectural_image = cv2.imread("bigben.jpg")
flag_image = cv2.imread("Flag.png")

# Resize images by 50%
architectural_image = cv2.resize(architectural_image, None, fx=0.5, fy=0.5)
flag_image = cv2.resize(flag_image, None, fx=0.5, fy=0.5)

# Make copies of the images for displaying points
architectural_image_display = architectural_image.copy()
flag_image_display = flag_image.copy()

# Create windows and set mouse callback function
cv2.imshow("Architectural Image", architectural_image_display)
cv2.setMouseCallback("Architectural Image", select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Flag Image", flag_image_display)
cv2.setMouseCallback("Flag Image", select_points)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute homography matrix if points are selected
if len(architectural_points) == 4 and len(flag_points) == 4:
    homography_matrix, _ = cv2.findHomography(np.array(flag_points), np.array(architectural_points))

    # Warp and blend images
    flag_warped = cv2.warpPerspective(flag_image, homography_matrix,
                                      (architectural_image.shape[1], architectural_image.shape[0]))
    blended_image = cv2.addWeighted(architectural_image, 0.7, flag_warped, 0.3, 0)

    # Display the final blended image
    cv2.imshow("Blended Image", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Four points are not selected in both images.")
