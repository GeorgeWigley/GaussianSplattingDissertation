from src.ColmapClasses.Point3D import *
import math
import numpy as np


def distance_between_points_3d(point_1: Point3D, point_2: Point3D) -> float:
    distance = math.sqrt(
        ((point_1.x - point_2.x) ** 2) + ((point_1.y - point_2.y) ** 2) + ((point_1.z - point_2.z) ** 2))
    return distance


def merge_two_points(point_1: Point3D, point_2: Point3D) -> Point3D:
    # average position
    new_transform_x = (point_1.x + point_2.x) / 2
    new_transform_y = (point_1.y + point_2.y) / 2
    new_transform_z = (point_1.z + point_2.z) / 2

    # image references
    image_references = []

    for image_reference in point_1.image_references:
        image_references.append(image_reference)

    for image_reference in point_2.image_references:
        image_references.append(image_reference)


def remove_black_points_from_np(points, colors, pixels):
    # Find indices of all non-black points
    non_black_indices = np.where(np.any(colors > 0, axis=1))[0]

    # Filter out black points
    filtered_points = points[non_black_indices]
    filtered_colors = colors[non_black_indices]
    filtered_pixels = [pixels[i] for i in non_black_indices]

    return filtered_points, filtered_colors, filtered_pixels


def remove_outliers(points, colors, pixels, threshold=2):
    """
    Remove outliers from a 3D point cloud.

    Parameters:
    - points: NumPy array of shape (n, 3), where n is the number of points.
    - threshold: The number of standard deviations from the mean distance
                 for a point to be considered an outlier.

    Returns:
    - A NumPy array of points with outliers removed.
    """
    # Calculate the mean point
    mean_point = np.mean(points, axis=0)

    # Calculate Euclidean distances from the mean point
    distances = np.linalg.norm(points - mean_point, axis=1)

    # Calculate mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Identify points that are within 'threshold' standard deviations
    inliers_mask = distances < (mean_distance + threshold * std_distance)

    # Filter and return inlier points
    inlier_points = points[inliers_mask]
    inlier_colors = colors[inliers_mask]
    inlier_pixels = [pixels[i] for i in inliers_mask]

    return inlier_points, inlier_colors, inlier_pixels