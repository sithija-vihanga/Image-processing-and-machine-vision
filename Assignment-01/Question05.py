import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_orig = cv.imread("Sources/shells.tif", cv.IMREAD_GRAYSCALE)

def hist_equalize(img):
    width, height = img.shape
    print("Width is:", width)
    print("Height is:", height)

    pixelValues = np.zeros(256)
    for i in range(height):
        for j in range(width):
            pixelValues[img[i][j]] += 1

    # Plot the histogram of pixel values
    plt.bar(np.arange(256), pixelValues)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pixel Values")
    plt.show()

    cdf = pixelValues.cumsum()
    plt.bar(np.arange(256), cdf)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Cumulative Sum Pixel Values")
    plt.show()
    cdf = cdf/(width*height)
    lookup_table = np.round((255/width*height)*cdf)
    print(lookup_table)

    for i in range(height):
        for j in range(width):
            img[i][j] = lookup_table[img[i][j]]

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

    cv.imwrite("A05-Output.png", img)
    cv.imshow("Output image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()



hist_equalize(img_orig)
