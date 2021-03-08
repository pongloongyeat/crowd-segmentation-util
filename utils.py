import cv2
import numpy as np
from scipy.spatial import KDTree


def gen_density_map_gaussian(im, points, sigma=4):
    """
    Generates a density map based on head coordinate
    points within an image.

    From: https://github.com/ZhengPeng7/CSRNet-Keras/blob/4dbf6eba91f8ecf64f175e494d0cb5a1219aa099/utils_gen.py#L53-L97

    im: Numpy array of the image.
    points: Numpy array of the head coordinates.

    returns: Numpy array of the density map.
    """
    density_map = np.zeros(im.shape[:2], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map
    if sigma == 4:
        # Adaptive sigma in CSRNet.
        leafsize = 2048
        tree = KDTree(points.copy(), leafsize=leafsize)
        distances, _ = tree.query(points, k=4)
    for idx_p, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        gaussian_radius = sigma * 2 - 1
        if sigma == 4:
            # Adaptive sigma in CSRNet.
            sigma = max(int(np.sum(distances[idx_p][1:4]) * 0.1), 1)
            gaussian_radius = sigma * 3
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(int(gaussian_radius*2+1), sigma),
            cv2.getGaussianKernel(int(gaussian_radius*2+1), sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        gaussian_map = gaussian_map[y_up:y_down, x_left:x_right]
        if np.sum(gaussian_map):
            gaussian_map = gaussian_map / np.sum(gaussian_map)
        density_map[
            max(0, p[0]-gaussian_radius):min(h, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(w, p[1]+gaussian_radius+1)
        ] += gaussian_map
    density_map = density_map / (np.sum(density_map / num_gt))
    return density_map

def gen_crowdseg(image, dmap):
    """
    Generates a crowd segmentation by multiplying an
    image and its density map element wise.

    image: Numpy array of the image.
    dmap: Numpy array of the density map.

    returns: Numpy array of the segmented crowd.
    """

    # Convert all valid points to a 1
    valid_points = [i > 0 for i in dmap]

    # Multiply valid points to "segment" image.
    for i in range(0, 3): image[:, :, i] = image[:, :, i] * valid_points

    return image
