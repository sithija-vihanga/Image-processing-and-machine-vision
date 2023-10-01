import cv2
import numpy as np

# Load the image with reduced color
im = cv2.imread('the_berry_farms_sunflower_field.jpeg', cv2.IMREAD_REDUCED_COLOR_4)

# Convert the image to grayscale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Define a range of sigma values
sigma_values = np.arange(6, 10, 1)

# Initialize a list to store detected circles
all_circles = []

# Loop through different sigma values and perform LoG and extrema detection
for sigma in sigma_values:
    # Apply LoG filter
    log = cv2.GaussianBlur(gray, (3, 3), sigmaX=sigma)
    log = cv2.Laplacian(log, cv2.CV_64F)

    output_filename = f'log_{sigma}.jpg'
    cv2.imwrite(output_filename, log)
    print(f'Saved {output_filename}')

    # Find local maxima as extrema in the scale space
    min_dist = int(sigma)  # Minimum distance between circles
    _, thresh_log = cv2.threshold(np.abs(log), 20, 80, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_log.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) and approximate circles
    circles = []
    for contour in contours:
        if len(contour) >= 10:
            approx = cv2.approxPolyDP(contour, 0.3, closed=True)
            if 35 < len(approx) < 60:  # filter out contours with no. of points to reduce noise
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circles.append((int(x), int(y), int(radius)))

    # Accumulate circles detected at this sigma value
    all_circles.extend(circles)

    # Draw circles on the image
    im_with_circles = im.copy()
    for circle in all_circles:
        cv2.circle(im_with_circles, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)  # Draw circles in green

    # Save the image with circles drawn on it
    output_filename = f'circles_detected_sigma_{sigma}.jpg'
    cv2.imwrite(output_filename, im_with_circles)
    print(f'Saved {output_filename}')

# Find and print details of the largest circle
largest_circle = max(all_circles, key=lambda x: x[2])
print(f'Largest circle: Center coordinates: ({largest_circle[0]}, {largest_circle[1]}), Radius: {largest_circle[2]}')

# Display the image with circles drawn on it
cv2.imshow('Circles Detected', im_with_circles)
cv2.waitKey(0)
cv2.destroyAllWindows()
