import numpy as np
import matplotlib.pyplot as plt

# Source and destination points
source_points = np.array([
    [1.90659, 2.51737],
    [2.20896, 1.1542],
    [2.37878, 2.15422],
    [1.98784, 1.44557],
    [2.83467, 3.41243],
    [9.12775, 8.60163],
    [4.31247, 5.57856],
    [6.50957, 5.65667],
    [3.20486, 2.67803],
    [6.60663, 3.80709],
    [8.40191, 3.41115],
    [2.41345, 5.71343],
    [1.04413, 5.29942],
    [3.68784, 3.54342],
    [1.41243, 2.6001]
])

destination_points = np.array([
    [5.0513, 1.14083],
    [1.61414, 0.92223],
    [1.95854, 1.05193],
    [1.62637, 0.93347],
    [2.4199, 1.22036],
    [5.58934, 3.60356],
    [3.18642, 1.48918],
    [3.42369, 1.54875],
    [3.65167, 3.73654],
    [3.09629, 1.41874],
    [5.55153, 1.73183],
    [2.94418, 1.43583],
    [6.8175, 0.01906],
    [2.62637, 1.28191],
    [1.78841, 1.0149]
])

def compute_homography(src_pts, dst_pts):
    A = []
    for src, dst in zip(src_pts, dst_pts):
        x, y = src
        u, v = dst
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)

    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))

    H = H / H[2, 2] #Normalization

    return H

def compute_residuals(H, src_pts, dst_pts):
    residuals = []
    for src, dst in zip(src_pts, dst_pts):
        src = np.append(src, 1)
        projected_point = np.dot(H, src)
        projected_point = projected_point / projected_point[2]
        error = np.linalg.norm(dst - projected_point[:2])
        residuals.append(error)
    return np.array(residuals)

def compute_weights(residuals, c):
    return 1 / (residuals + c)

def gnc(src_pts, dst_pts, threshold=0.005, max_iterations=1000):
    H = compute_homography(src_pts, dst_pts)
    c = 1.0

    for _ in range(max_iterations):
        residuals = compute_residuals(H, src_pts, dst_pts)

        weights = compute_weights(residuals, c)

        # Weighted least squares
        A = []
        for (src, dst), w in zip(zip(src_pts, dst_pts), weights):
            x, y = src
            u, v = dst
            A.append(w * np.array([x, y, 1, 0, 0, 0, -u*x, -u*y, -u]))
            A.append(w * np.array([0, 0, 0, x, y, 1, -v*x, -v*y, -v]))
        A = np.array(A)

        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape((3, 3))

        H = H / H[2, 2] 

        c = c / 2

        if np.max(residuals) < threshold:
            break

    return H

H_gnc = gnc(source_points, destination_points)

print("Homography Transformation (GNC):")
print(H_gnc)

projected_points = []
for src in source_points:
    src = np.append(src, 1)
    projected_point = np.dot(H_gnc, src)
    projected_point = projected_point / projected_point[2]
    projected_points.append(projected_point[:2])
projected_points = np.array(projected_points)

plt.figure()

plt.scatter(*zip(*source_points), marker='o', color='b', label='Source points')
plt.scatter(*zip(*destination_points), marker='x', color='r', label='Destination points')
plt.scatter(*zip(*projected_points), marker='x', color='g', label='Projected points (GNC)')

for src, dst in zip(source_points, projected_points):
    plt.plot([src[0], dst[0]], [src[1], dst[1]], 'r-')

plt.legend()
plt.show()

