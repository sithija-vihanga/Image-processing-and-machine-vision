import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Sources/jeniffer.jpg')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the image into the three channels
hue, saturation, value = cv2.split(hsv_image)

# Display the original image
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(hue, cmap='gray')
plt.title('Hue Channel')

plt.subplot(2, 3, 3)
plt.imshow(saturation, cmap='gray')
plt.title('Saturation Channel')

plt.subplot(2, 3, 4)
plt.imshow(value, cmap='gray')
plt.title('Value Channel')

threshold_value = 12
# create binary mask
ret, binary_mask = cv2.threshold(saturation, threshold_value, 255, cv2.THRESH_BINARY)
kernel = np.ones((17, 17), np.uint8)

# Dilate the binary mask
binary_mask = cv2.dilate(binary_mask, kernel, iterations=4)
binary_mask = cv2.erode(binary_mask, kernel, iterations=9)

plt.subplot(2, 3, 5)
plt.imshow(binary_mask, cmap='gray')
plt.title('Binary mask')

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered_img = cv2.bitwise_and(image, image, mask=binary_mask)
inverse_mask = np.invert(binary_mask)
plt.subplot(2, 3, 6)
plt.imshow(filtered_img, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()


# Histogram equalization from slides

hist, bins = np.histogram(image.ravel(), 256, [0, 256])

# Calculate the cumulative distribution function (CDF)
cdf = hist.cumsum()

# Normalize the CDF
cdf_normalized = cdf * hist.max()/ cdf.max()

# Plot the CDF and histogram of the original image
plt.plot(cdf_normalized, color='b')
plt.hist(image.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histogram'), loc='upper left')
plt.title('Histogram and CDF of the Original Image')
plt.show()

# Perform histogram equalization using OpenCV
equ = cv2.equalizeHist(image)

# Calculate the histogram and CDF of the equalized image
hist_eq, bins_eq = np.histogram(equ.ravel(), 256, [0, 256])
cdf_eq = hist_eq.cumsum()
cdf_eq_normalized = cdf_eq * hist_eq.max() / cdf_eq.max()

# Plot the CDF and histogram of the equalized image
plt.plot(cdf_eq_normalized, color='b')
plt.hist(equ.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histogram'), loc='upper left')
plt.title('Histogram and CDF of the Equalized Image')
plt.show()

# Create a side-by-side comparison of the original and equalized images
res = np.hstack((image, equ))
plt.axis('off')
plt.imshow(res, cmap='gray')
plt.show()

bg_final = cv2.bitwise_and(image, image, mask=inverse_mask)
fg_final = cv2.bitwise_and(equ, equ, mask=binary_mask)
final_img = bg_final+fg_final
plt.imshow(final_img, cmap='gray')
plt.show()





