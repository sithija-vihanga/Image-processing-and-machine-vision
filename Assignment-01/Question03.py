import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_orig = cv.imread('Sources/emma.jpg', cv.IMREAD_COLOR)
gamma = 1.5
table = np.array([(i/255.0)**(gamma)*255.0 for i in np.arange(0, 256)]).astype('uint8')

# Splitting LAB channels
img_orig_lab = cv.cvtColor(img_orig, cv.COLOR_BGR2Lab)
L, a, b = cv.split(img_orig_lab)

# Applying gamma correction to L channel
L_gamma = cv.LUT(L, table)

# Merging LAB channels back together
img_gamma_lab = cv.merge((L_gamma, a, b))

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(cv.cvtColor(img_orig_lab, cv.COLOR_Lab2RGB))
axarr[0, 1].imshow(cv.cvtColor(img_gamma_lab, cv.COLOR_Lab2RGB))

color = ('b', 'g', 'r')
for i, c in enumerate(color):
    hist_orig = cv.calcHist([img_orig_lab], [i], None, [256], [0, 256])
    axarr[1, 0].plot(hist_orig, color=c)
    hist_gamma = cv.calcHist([img_gamma_lab], [i], None, [256], [0, 256])
    axarr[1, 1].plot(hist_gamma, color=c)

plt.show()
