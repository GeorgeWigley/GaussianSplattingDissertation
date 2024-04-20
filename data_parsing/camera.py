import json
import os.path

import cv2
import open3d as o3d
import numpy as np

import utils
from data_parsing.alignment import AlignmentProcessor


class Camera:

    def __init__(self, camera_path, callibration_path):
        """
        A camera is a single view point of a scene withing a given take. It has a serial ID and references x: RGB, D and
        Segmented images, where x is the number of frames in the capture.
        :param camera_path: the path to the camera data
        """
        self.camera_path = camera_path

        # get serial id
        head, tail = os.path.split(camera_path)
        self.serial_id = tail

        # compute dict from frame_num -> image paths
        self.images = {}
        for filename in os.listdir(camera_path):
            parts = filename.split("-")
            frame_num = int(parts[2])
            frame_type = parts[1]
            # update dict, creating a new entry if frame has not already been processed
            if self.images.get(frame_num) is None:
                self.images[frame_num] = {
                    "colour_images": None,
                    "depth_images": None,
                    "segmented_images": None
                }
            self.images[frame_num][frame_type] = filename

        # compute frame count
        self.frame_count = max(self.images.keys())

        # load extrinsic of depth camera
        with open(callibration_path) as f:
            parsed_data = json.load(f)
            self.camera_callibration = parsed_data["cameras"][self.serial_id]

    def get_available_frames(self):
        return list(self.images.keys())

    def get_image_paths_for_frame(self, frame_number, full_path=False):
        """
        Will return the RGB, D and Segmentation images for a given frame.
        :param frame_number: The frame number to get data for
        :param full_path: should the method return the full path to the image from the context root or just the filename
        :return: a dict of the form
                {
                    "colour_images": filename/path,
                    "depth_images": filename/path,
                    "segmented_images": filename/path
                }
        """
        if frame_number in self.images.keys():
            if full_path:
                result_dict = {}
                for key, value in self.images[frame_number].items():
                    result_dict[key] = os.path.join(self.camera_path, value)
                return result_dict
            else:
                return self.images[frame_number]

        else:
            raise Exception(
                f"Camera ID: {self.serial_id} : Frame requested not available. requested frame {frame_number}, try "
                f"using get_available_frames() first to get a list of available frames")

    def get_depth_w2c(self):
        depth_r = np.reshape(self.camera_callibration["depth_extrinsics"]["orientation"], (3, 3))
        depth_t = self.camera_callibration["depth_extrinsics"]["translation"]
        mat = np.eye(4)
        # loaded row major but data provided as column major
        mat[:3, :3] = depth_r.T
        mat[:3, 3] = depth_t
        return mat

    def get_depth_k(self):
        fx = self.camera_callibration["depth_intrinsics"]["fx"]
        fy = self.camera_callibration["depth_intrinsics"]["fy"]
        ppx = self.camera_callibration["depth_intrinsics"]["ppx"]
        ppy = self.camera_callibration["depth_intrinsics"]["ppy"]
        w = self.camera_callibration["depth_intrinsics"]["width"]
        h = self.camera_callibration["depth_intrinsics"]["height"]

        return fx, fy, ppx, ppy, w, h

    def get_depth_k_mat(self):
        fx = self.camera_callibration["depth_intrinsics"]["fx"]
        fy = self.camera_callibration["depth_intrinsics"]["fy"]
        ppx = self.camera_callibration["depth_intrinsics"]["ppx"]
        ppy = self.camera_callibration["depth_intrinsics"]["ppy"]
        w = self.camera_callibration["depth_intrinsics"]["width"]
        h = self.camera_callibration["depth_intrinsics"]["height"]

        res = [[fx, 0, ppx],
               [0, fy, ppy],
               [0, 0, 1]]

        return res

    def get_c2d(self):
        color_r = np.reshape(self.camera_callibration["colour_to_depth_extrinsics"]["orientation"], (3, 3))
        color_t = self.camera_callibration["colour_to_depth_extrinsics"]["translation"]
        mat = np.eye(4)
        mat[:3, :3] = color_r.T
        mat[:3, 3] = color_t
        return mat

    def get_colour_w2c(self):

        colmap_poses = {
            '000236320812': np.array(
                [[9.99995703e-01, - 1.30696812e-03, - 2.62403348e-03, - 4.85492584e+00],
                 [1.30671747e-03, 9.99999142e-01, - 9.72324683e-05, - 2.76190190e+00],
                 [2.62415831e-03, 9.38031801e-05, 9.99996552e-01, 2.11866640e+00],
                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
            ),
            '000404613112': np.array(
                [[0.53206451, 0.0620448, - 0.8444275, 4.2861101],
                 [-0.06871399, 0.99718605, 0.02997289, - 1.94213824],
                 [0.84391098, 0.04207647, 0.53483066, 0.64024422],
                 [0., 0., 0., 1.]]
            ),
            '000407313112': np.array(
                [[-7.91852643e-02, 1.81986328e-01, - 9.80107479e-01, 6.93644333e+00],
                 [-3.67016104e-01, 9.08810544e-01, 1.98400038e-01, - 1.17361318e+00],
                 [9.26838105e-01, 3.75425588e-01, - 5.17249001e-03, 3.16265359e+00],
                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
            ),
            '000558313112': np.array(
                [[0.91291681, - 0.0491824, 0.40517155, - 7.79868908],
                 [0.08214781, 0.99453931, - 0.06436852, - 1.81735566],
                 [-0.39979324, 0.09204706, 0.91197188, 4.42649365],
                 [0., 0., 0., 1.]]
            ),
            '000951614712': np.array(
                [[0.7988739, 0.17106667, - 0.57665994, - 0.63248427],
                 [-0.24142424, 0.96928485, - 0.04691707, 1.44876046],
                 [0.5509218, 0.17670051, 0.81563601, - 0.10707786],
                 [0., 0., 0., 1.]]
            )
        }

        # colmap_pose = colmap_poses[self.serial_id]
        base = np.linalg.inv(self.get_depth_w2c() @ self.get_c2d())
        # base[:3, :3] = colmap_pose[:3, :3]

        return base

    def get_colour_k(self):
        fx = self.camera_callibration["colour_intrinsics"]["fx"]
        fy = self.camera_callibration["colour_intrinsics"]["fy"]
        ppx = self.camera_callibration["colour_intrinsics"]["ppx"]
        ppy = self.camera_callibration["colour_intrinsics"]["ppy"]
        w = self.camera_callibration["colour_intrinsics"]["width"]
        h = self.camera_callibration["colour_intrinsics"]["height"]

        return fx, fy, ppx, ppy, w, h

    def get_colour_k_mat(self):
        fx = self.camera_callibration["colour_intrinsics"]["fx"]
        fy = self.camera_callibration["colour_intrinsics"]["fy"]
        ppx = self.camera_callibration["colour_intrinsics"]["ppx"]
        ppy = self.camera_callibration["colour_intrinsics"]["ppy"]
        w = self.camera_callibration["colour_intrinsics"]["width"]
        h = self.camera_callibration["colour_intrinsics"]["height"]

        res = [[fx, 0, ppx],
               [0, fy, ppy],
               [0, 0, 1]]

        return res

    def get_rgbd_for_frame_transformed_to_color(self, frame_number, apply_segmentation_mask=True):
        # Load images for frame
        images = self.get_image_paths_for_frame(frame_number, full_path=True)

        depth_img = o3d.io.read_image(images["depth_images"])
        rgb_img = o3d.io.read_image(images["colour_images"])
        mask_img = o3d.io.read_image(images["segmented_images"])

        transformed_depth_image = AlignmentProcessor.align_depth_to_rgb_image(
            np.asarray(depth_img),
            np.asarray(rgb_img),
            self.get_depth_k_mat(),
            self.get_colour_k_mat(),
            np.linalg.inv(self.get_c2d())
        )

        cv2.imwrite("temporary.png", transformed_depth_image)

        # Mask depth image if apply_segmentation_mask is set
        masked_depth = transformed_depth_image
        if apply_segmentation_mask:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            eroded_mask = cv2.erode(np.asarray(mask_img), kernel)
            masked_depth = np.where(np.asarray(eroded_mask) == 0, 0, np.asarray(transformed_depth_image))

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_img),
            o3d.geometry.Image(masked_depth),
            convert_rgb_to_intensity=False,
        )

        return rgbd_img

    def get_colored_point_cloud_color(self, frame_number):
        # Load intrinsics
        rgb_fx, rgb_fy, rgb_ppx, rgb_ppy, rgb_w, rgb_h = self.get_colour_k()
        rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            rgb_w, rgb_h, rgb_fx, rgb_fy, rgb_ppx, rgb_ppy
        )

        rgbd_img = self.get_rgbd_for_frame_transformed_to_color(frame_number, apply_segmentation_mask=False)

        # Create a new point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img,
            rgb_intrinsics,
            self.get_colour_w2c(),
            project_valid_depth_only=True
        )


        return pcd

    def get_point_cloud_depth(self, frame_number, color):
        # Load intrinsics for depth camera
        depth_fx, depth_fy, depth_ppx, depth_ppy, depth_w, depth_h = self.get_depth_k()
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_w, depth_h, depth_fx, depth_fy, depth_ppx, depth_ppy
        )

        # Compute correct transformation
        extrinsic_depth_mat = self.get_depth_w2c()

        # Create a new point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.io.read_image(self.get_image_paths_for_frame(frame_number, True)["depth_images"]),
            depth_intrinsics,
            np.linalg.inv(extrinsic_depth_mat),
            project_valid_depth_only=True
        )
        pcd.paint_uniform_color(color)

        return pcd

    def get_camera_visualisation_line_set(self, color):
        # load intrinsics
        depth_fx, depth_fy, depth_ppx, depth_ppy, depth_w, depth_h = self.get_depth_k()
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_w, depth_h, depth_fx, depth_fy, depth_ppx, depth_ppy
        )

        rgb_fx, rgb_fy, rgb_ppx, rgb_ppy, rgb_w, rgb_h = self.get_colour_k()
        rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            rgb_w, rgb_h, rgb_fx, rgb_fy, rgb_ppx, rgb_ppy
        )

        # Add depth sensor vis
        depth_cam_vis = o3d.geometry.LineSet.create_camera_visualization(
            depth_intrinsics,
            np.linalg.inv(self.get_depth_w2c()),
            scale=0.3
        )
        depth_cam_vis.paint_uniform_color(color)

        # add rgb sensor vis
        col_cam_vis = o3d.geometry.LineSet.create_camera_visualization(
            rgb_intrinsics,
            self.get_colour_w2c(),
            scale=0.3
        )
        col_cam_vis.paint_uniform_color(color)

        return depth_cam_vis, col_cam_vis
