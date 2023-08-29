import cv2
import numpy as np

# Load the image
img_orig = cv2.imread("Sources/spider.png")
img_hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(img_hsv)

a = 0.3
sigma = 70

# Define the intensity transformation function
def f_x(x):
    return  (a * 128) * np.exp(-((x - 128) ** 2) / (2 * (sigma ** 2))) + x

# Apply the transformation to the saturation channel
s_transformed = np.clip(f_x(s), 0, 255).astype(np.uint8)

# Recombine the planes
img_vibrant_hsv = cv2.merge((h, s_transformed, v))

# Convert HSV image to BGR for display
img_vibrant_bgr = cv2.cvtColor(img_vibrant_hsv, cv2.COLOR_HSV2BGR)

# Display images
cv2.imshow('Original Image', img_orig)
cv2.imshow('Vibrance-Enhanced Image', img_vibrant_bgr)
cv2.imwrite("A04-output.jpg",img_vibrant_bgr)
#cv2.imshow('Intensity Transformation', s_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
