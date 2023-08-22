import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv.imread('sources/BrainProtonDensitySlice9.png', cv.IMREAD_GRAYSCALE)

# Calculate the histogram
hist = cv.calcHist([img], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(hist)
plt.xlim([0, 256])
plt.show()


c = np.array([(180, 10), (210, 150),(230, 230), (240, 250)])
#c = np.array([(180, 10), (210, 254),(230, 230), (240, 200)])

t1 = np.linspace(0, c[0, 1], c[0, 0] + 1 - 0).astype('uint8')
t2 = np.linspace(c[0, 1] + 1, c[1, 1], c[1, 0] - c[0, 0]).astype('uint8')
t3 = np.linspace(c[1, 1] + 1, c[2, 1], c[2, 0] - c[1, 0]).astype('uint8')
t4 = np.linspace(c[2, 1] + 1, c[3, 1], c[3, 0] - c[2, 0]).astype('uint8')
t5 = np.linspace(c[3, 1] + 1, 255, 255 - c[3, 0]).astype('uint8')

transform_black = np.concatenate((t1, t2), axis=0).astype('uint8')
transform_black = np.concatenate((transform_black, t3), axis=0).astype('uint8')
transform_black = np.concatenate((transform_black, t4), axis=0).astype('uint8')
transform_black = np.concatenate((transform_black, t5), axis=0).astype('uint8')

transform_white = 255 - transform_black

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(transform_black)
ax1.set_xlabel(r'Input, $f(\mathbf{x})$')
ax1.set_ylabel('Output, $\mathrm{T}[f(\mathbf{x})]$')
ax1.set_xlim(0, 255)
ax1.set_ylim(0, 255)
ax1.set_aspect('equal')

ax2.plot(transform_white)
ax2.set_xlabel(r'Input, $f(\mathbf{x})$')
ax2.set_ylabel('Output, $\mathrm{T}[f(\mathbf{x})]$')
ax2.set_xlim(0, 255)
ax2.set_ylim(0, 255)
ax2.set_aspect('equal')


plt.savefig('A02_transform.png')
plt.show()

cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
cv.imshow("Image", img)
cv.waitKey(0)

image_transformed = cv.LUT(img, transform_black)
cv.imshow("Image", image_transformed)
cv.imwrite('A02_black.jpg', image_transformed)
cv.waitKey(0)

image_transformed = cv.LUT(img, transform_white)
cv.imshow("Image", image_transformed)
cv.imwrite('A02_white.jpg', image_transformed)

cv.waitKey(0)
cv.destroyAllWindows()

