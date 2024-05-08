import numpy as np
import cv2


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
            depthDilation=False
        )

        return res
