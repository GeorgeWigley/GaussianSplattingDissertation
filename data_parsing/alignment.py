import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyrgb import *

class AlignmentProcessor:
    @staticmethod
    def align_depth_to_rgb_image(depth_img, rgb_img, depth_intrinsic, rgb_intrinsics, depth_to_rgb_extrinsics):
        depth_img_array = np.asarray(depth_img)
        rgb_img_array = np.asarray(rgb_img)
        h, w = np.shape(rgb_img_array)[:2]

        res = cv2.rgbd.registerDepth(
            np.asarray(depth_intrinsic),
            np.asarray(rgb_intrinsics),
            np.array([]),
            depth_to_rgb_extrinsics,
            depth_img_array.astype(np.uint16),
            [w, h],
            depthDilation=True
        )

        return res

    @staticmethod
    def align_depth_to_rgb_custom(depth_image, color_image, depth_intrinsics, color_intrinsics, extrinsics):
        # ------------------------
        #  Alignment Calculation
        # ------------------------
        # Dimensions of the color image
        color_height, color_width = color_image.shape[:2]

        # Project each pixel in the depth image into the color image reference frame
        aligned_depth_image = np.zeros((color_height, color_width), dtype=np.uint16)

        for y in range(depth_image.shape[0]):
            for x in range(depth_image.shape[1]):
                depth_value = depth_image[y, x]

                # Skip pixels with no depth data
                if depth_value == 0:
                    continue

                # Homogeneous pixel coordinates in the depth frame
                depth_pixel = np.array([x, y, 1])

                # Calculate 3D point in depth camera coordinates
                point_3d_depth = depth_value * np.linalg.inv(depth_intrinsics) @ depth_pixel

                # Transform to color camera coordinates
                point_3d_color = extrinsics @ np.array(
                    [point_3d_depth[0], point_3d_depth[1], point_3d_depth[2], 1])

                # Project 3D point into the color image (MAGIC NUMBER)
                color_pixel = (color_intrinsics @ point_3d_color[:3]) / point_3d_color[2]  # * 0.98
                color_x = int(color_pixel[0])
                color_y = int(color_pixel[1])

                # Assign depth to the corresponding color pixel (if within bounds)
                if 0 <= color_x < color_width and 0 <= color_y < color_height:
                    aligned_depth_image[color_y, color_x] = depth_value

        return aligned_depth_image

    @staticmethod
    def gen_mapped_col(depth, dfx, dfy, dcx, dcy,
                         color, cfx, cfy, ccx, ccy,
                         d2c_R, d2c_t):
        mapped_color, valid_mask = gen_mapped_color(
            depth, dfx, dfy, dcx, dcy,
            color, cfx, cfy, ccx, ccy,
            d2c_R, d2c_t,
            ddist_type=None, ddist_param=[],
            cdist_type=None, cdist_param=[],
            cdist_interp='NN',
            missing_color=[0, 0, 0]
        )
        return mapped_color

    @staticmethod
    def depth_rgb_registration(depth_data, rgb_data,
                               fx_d, fy_d, cx_d, cy_d,
                               fx_rgb, fy_rgb, cx_rgb, cy_rgb,
                               extrinsics):

        depth_height, depth_width = depth_data.shape
        rgb_height, rgb_width, _ = rgb_data.shape

        # Aligned will contain X, Y, Z, R, G, B values
        aligned = np.zeros((depth_height, depth_width, 6))
        res = np.zeros((rgb_height, rgb_width))

        # We'll assume a depth scale of 1 for this example
        depth_scale = 1.0

        for v in range(depth_height):
            for u in range(depth_width):
                # Apply depth intrinsics
                z = depth_data[v, u] / depth_scale
                x = (u - cx_d) * z / fx_d
                y = (v - cy_d) * z / fy_d

                # Apply the extrinsics
                transformed = np.dot(extrinsics, np.array([x, y, z, 1]))
                aligned[v, u, 0:3] = transformed[:3]

        for v in range(depth_height):
            for u in range(depth_width):
                # Apply RGB intrinsics
                x = (aligned[v, u, 0] * fx_rgb / aligned[v, u, 2]) + cx_rgb
                y = (aligned[v, u, 1] * fy_rgb / aligned[v, u, 2]) + cy_rgb

                # Check for valid indices
                if (x >= rgb_data.shape[1] - 1 or y >= rgb_data.shape[0] or
                        x < 0 or y < 0 or np.isnan(x) or np.isnan(y)):
                    continue

                res[round(y)][round(x)] = (depth_data[v][u])

        return res.astype(np.uint16)

