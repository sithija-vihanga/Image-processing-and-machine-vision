import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('Sources/flower.png')

mask = np.zeros(image.shape[:2], np.uint8)

# specify the background and foreground model
# using numpy the array is constructed of 1 row
# and 65 columns, and all array elements are 0
# Data type for the array is np.float64 (default)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

# define the Region of Interest (ROI)
# as the coordinates of the rectangle
# where the values are entered as
# (startingPoint_x, startingPoint_y, width, height)
# these coordinates are according to the input image
# it may vary for different images
rectangle = (30,100, 600, 450)

# apply the grabcut algorithm with appropriate
# values as parameters, number of iterations = 3
# cv2.GC_INIT_WITH_RECT is used because
# of the rectangle mode is used
cv2.grabCut(image, mask, rectangle,
			backgroundModel, foregroundModel,
			3, cv2.GC_INIT_WITH_RECT)

# In the new mask image, pixels will
# be marked with four flags
# four flags denote the background / foreground
# mask is changed, all the 0 and 2 pixels
# are converted to the background
# mask is changed, all the 1 and 3 pixels
# are now the part of the foreground
# the return type is also mentioned,
# this gives us the final mask
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')

# The final mask is multiplied with
# the input image to give the segmented image.
modified_image = image * mask2[:, :, np.newaxis]
bg_img = image - modified_image
bg_blured = cv2.GaussianBlur(bg_img,(9,9),3)

# output segmented image with colorbar
bg_blured = cv2.cvtColor(bg_blured,cv2.COLOR_BGR2RGB)

plt.imshow(bg_blured)
plt.show()

plt.imshow(mask2, cmap='gray')
plt.show()
modified_image = cv2.cvtColor(modified_image,cv2.COLOR_BGR2RGB)
plt.imshow(modified_image)
plt.colorbar()
plt.show()
portrait_img = bg_blured+ modified_image
plt.imshow(portrait_img)
plt.show()
