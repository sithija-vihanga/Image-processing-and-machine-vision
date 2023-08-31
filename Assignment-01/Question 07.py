################ Method 01 #################
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv.imread('Sources/einstein.png', cv.IMREAD_GRAYSCALE)
#
# kernel = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [1, 2, 1]], dtype='float32')
# imgc = cv.filter2D(img, -1, kernel)
#
# fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(18, 18))
# axes[0].imshow(img, cmap='gray')
# axes[0].set_title('Original')
# axes[0].set_xticks([]), axes[0].set_yticks([])
# axes[1].imshow(imgc, cmap='gray')
# axes[1].set_title('Sobel Vertical')
# axes[1].set_xticks([]), axes[1].set_yticks([])
# plt.show()
#
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# img = cv.imread('Sources/einstein.png', cv.IMREAD_GRAYSCALE)
# kernel = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [1, 2, 1]], dtype='float32')
#
# def sobel_filter(img, kernel):
#     kernel_h, kernel_w = kernel.shape # get kernel shape
#     img_h, img_w = img.shape
#     print("height :", img_h)
#     print("Width :", img_w)
#
#     offset = int((kernel_w-1)/2)
#     #print(type(img[0:3,0:4]))
#
#     out_img = np.zeros_like(img)
#
#     for i in range(offset,img_w-offset):
#         for j in range(offset,img_h-offset):
#                 value = np.sum(img[j-1:j+2,i-1:i+2]*kernel)
#                 out_img[j][i] = value
#
#     print(out_img)
#     cv.imshow("image", out_img)
#     cv.waitKey(0)
#
#
#
#
#
# sobel_filter(img,kernel)
#

##################### working ########################
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# def sobel_filter(img, kernel):
#     img_blur = cv.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
#     kernel_h, kernel_w = kernel.shape
#     img_h, img_w = img.shape
#
#     offset = (kernel_w - 1) // 2
#
#     out_img = np.zeros_like(img, dtype=np.float32)  # Use float32 data type for the output
#
#     for i in range(offset, img_w - offset):
#         for j in range(offset, img_h - offset):
#             value = np.sum(img_blur[j - offset : j + offset + 1, i - offset : i + offset + 1] * kernel)
#             out_img[j][i] = value
#
#     out_img = cv.normalize(out_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
#
#     plt.imshow(out_img, cmap='gray')
#     plt.show()
#
# img = cv.imread('Sources/einstein.png', cv.IMREAD_GRAYSCALE)
# kernel = np.array([[-1, -2, -1],
#                    [0, 0, 0],
#                    [1, 2, 1]], dtype='float32')
#
# sobel_filter(img, kernel)
#

####################### method 03 ####################
import cv2
import numpy as np
import matplotlib.pyplot as plt


def sobel_filter(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur

    gradient_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # thresholded = np.zeros_like(gradient_magnitude)
    # thresholded[gradient_magnitude > 100] = 255  # Adjust threshold value

    plt.imshow(gradient_magnitude, cmap='gray')
    plt.show()


img = cv2.imread('Sources/einstein.png', cv2.IMREAD_GRAYSCALE)
sobel_filter(img)
