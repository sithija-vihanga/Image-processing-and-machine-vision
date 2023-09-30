import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(0)

N = 100
half_n = N // 2
r = 10
x0_gt, y0_gt = 2, 3  # Center
s = r / 16
t = np.random.uniform(0, 2 * np.pi, half_n)
n = s * np.random.randn(half_n)
x, y = x0_gt + (r + n) * np.cos(t), y0_gt + (r + n) * np.sin(t)
X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

s = 1.
m, b = -1, 2
x = np.linspace(-12, 12, half_n)
y = m * x + b + s * np.random.randn(half_n)
X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

X = np.vstack((X_circ, X_line))  # All points

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_line[:, 0], X_line[:, 1], label='Line')
ax.scatter(X_circ[:, 0], X_circ[:, 1], label='Circle')
circle_gt = plt.Circle((x0_gt, y0_gt), r, color='g', fill=False, label='Ground truth circle')
ax.add_patch(circle_gt)
ax.plot(x0_gt, y0_gt, '+', color='g')

x_min, x_max = ax.get_xlim()
x_ = np.array([x_min, x_max])
y_ = m * x_ + b
plt.plot(x_, y_, color='m', label='Ground truth line')
plt.legend()
#plt.show()


# Part (a) - RANSAC for Line Estimation

def calculate_distance(a, b, d, points):
    # Calculate normal distance from points to the line
    distances = np.abs(a * points[:, 0] + b * points[:, 1] - d) / np.sqrt(a**2 + b**2)
    return distances

def ransac_line(points, num_iterations=100, threshold=0.5):
    best_a, best_b, best_d = 0, 0, 0
    best_inliers = []

    for _ in range(num_iterations):
        # Randomly select 2 points for line estimation
        sample_points = points[np.random.choice(points.shape[0], 2, replace=False)]
        p1, p2 = sample_points

        # Calculate line parameters (a, b, d) from two points
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        d = p1[1]*p2[0] - p1[0]*p2[1]

        # Calculate distances from all points to the line
        distances = calculate_distance(a, b, d, points)

        # Count inliers (points within the threshold)
        inliers = points[distances < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_a, best_b, best_d = a, b, d

    return best_a, best_b, best_d, best_inliers

# Estimate line using RANSAC
best_a, best_b, best_d, line_inliers = ransac_line(X)

# Part (c) - Plotting the results
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Plot original points
ax.scatter(X[:, 0], X[:, 1], label='Original Points', alpha=0.5)

# Plot line estimated from RANSAC
x_vals = np.linspace(-15, 15, 2)
y_vals = (-best_a * x_vals - best_d) / best_b
ax.plot(x_vals, y_vals, color='r', label='Estimated Line (RANSAC)')

# Plot line inliers
ax.scatter(line_inliers[:, 0], line_inliers[:, 1], label='Line Inliers', color='g')

ax.set_xlim(-15, 15)
ax.set_ylim(-15, 20)

# Set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()
