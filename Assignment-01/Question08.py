import cv2
import numpy as np
import matplotlib.pyplot as plt


def zoom_nearest_neighbor(image, scale_factor):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    zoomed_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_x = int(x / scale_factor)
            src_y = int(y / scale_factor)
            zoomed_image[y, x] = image[src_y, src_x]

    return zoomed_image


def zoom_bilinear_interpolation(image, scale_factor):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    zoomed_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            src_x = x / scale_factor
            src_y = y / scale_factor
            x1, y1 = int(np.floor(src_x)), int(np.floor(src_y))
            x2, y2 = int(np.ceil(src_x)), int(np.ceil(src_y))

            # Ensure coordinates are within the valid range
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            q11 = image[y1, x1]
            q21 = image[y1, x2]
            q12 = image[y2, x1]
            q22 = image[y2, x2]

            x_weight = src_x - x1
            y_weight = src_y - y1

            interpolated_value = (1 - x_weight) * (1 - y_weight) * q11 + \
                                 x_weight * (1 - y_weight) * q21 + \
                                 (1 - x_weight) * y_weight * q12 + \
                                 x_weight * y_weight * q22

            zoomed_image[y, x] = interpolated_value

    return zoomed_image

def calculate_normalized_ssd(image1, image2):
    squared_diff = (image1 - image2) ** 2
    ssd = np.sum(squared_diff)
    normalized_ssd = ssd / (image1.shape[0] * image1.shape[1] * image1.shape[2])
    return normalized_ssd


# Load the input image
image = cv2.imread('Sources/zooming/im08small.png')

# Define the zoom scale factor
scale_factor = 4.0  # You can change this factor

# Zoom using nearest-neighbor method
zoomed_nearest = zoom_nearest_neighbor(image, scale_factor)

# Zoom using bilinear interpolation method
zoomed_bilinear = zoom_bilinear_interpolation(image, scale_factor)

# # Display the original and zoomed images
# cv2.imshow('Original Image', image)
# cv2.imshow('Zoomed (Nearest Neighbor)', zoomed_nearest)
# cv2.imshow('Zoomed (Bilinear Interpolation)', zoomed_bilinear)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.figure(figsize=(10, 4))

# Nearest Neighbor zoomed image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(zoomed_nearest, cv2.COLOR_BGR2RGB))
plt.title('Zoomed (Nearest Neighbor)')

# Bilinear Interpolation zoomed image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(zoomed_bilinear, cv2.COLOR_BGR2RGB))
plt.title('Zoomed (Bilinear Interpolation)')

plt.tight_layout()
plt.show()



# Load the original large image for comparison
original_large_image = cv2.imread('Sources/zooming/im08.png')

# Resize the zoomed images to match the dimensions of the original large image
zoomed_nearest_resized = cv2.resize(zoomed_nearest, (original_large_image.shape[1], original_large_image.shape[0]))
zoomed_bilinear_resized = cv2.resize(zoomed_bilinear, (original_large_image.shape[1], original_large_image.shape[0]))

# Calculate normalized SSD for nearest-neighbor zoom
ssd_nearest = calculate_normalized_ssd(original_large_image, zoomed_nearest_resized)

# Calculate normalized SSD for bilinear interpolation zoom
ssd_bilinear = calculate_normalized_ssd(original_large_image, zoomed_bilinear_resized)

print(f"Normalized SSD (Nearest Neighbor): {ssd_nearest}")
print(f"Normalized SSD (Bilinear Interpolation): {ssd_bilinear}")

