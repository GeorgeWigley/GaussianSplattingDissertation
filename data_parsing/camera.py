import json
import os.path

import cv2
import open3d as o3d
import numpy as np

from data_parsing.alignment import AlignmentProcessor
from utils import compute_final_transform, roll_down_2d_array


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

    def get_rgbd_for_frame_transformed_to_depth(self, frame_number, apply_segmentation_mask=True):
        # Load extrinsic mapping depth -> col
        extrinsic_depth_to_col_mat = self.get_extrinsic_color_to_depth_matrix()

        # Load intrinsics for both Cameras
        depth_fx, depth_fy, depth_ppx, depth_ppy, depth_w, depth_h = self.get_intrinsic_depth()
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_w, depth_h, depth_fx, depth_fy, depth_ppx, depth_ppy
        )
        rgb_fx, rgb_fy, rgb_ppx, rgb_ppy, rgb_w, rgb_h = self.get_intrinsic_color()
        rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            rgb_w, rgb_h, rgb_fx, rgb_fy, rgb_ppx, rgb_ppy
        )

        # Load images for frame
        images = self.get_image_paths_for_frame(frame_number, full_path=True)

        depth_img = o3d.io.read_image(images["depth_images"])
        rgb_img = o3d.io.read_image(images["colour_images"])
        mask_img = o3d.io.read_image(images["segmented_images"])

        # transform RGB to depth frame space
        rgb_aligned_img = AlignmentProcessor.gen_mapped_col(
            np.asarray(depth_img), depth_fx, depth_fy, depth_ppx, depth_ppy,
            np.asarray(rgb_img), rgb_fx, rgb_fy, rgb_ppx, rgb_ppy,
            extrinsic_depth_to_col_mat[:3, :3],
            np.zeros((3))  # extrinsic_depth_to_col_mat[:3, 3]
        )
        mask_aligned_image = AlignmentProcessor.gen_mapped_col(
            np.asarray(depth_img), depth_fx, depth_fy, depth_ppx, depth_ppy,
            np.repeat(np.asarray(mask_img)[..., np.newaxis], 3, axis=-1), rgb_fx, rgb_fy, rgb_ppx, rgb_ppy,
            extrinsic_depth_to_col_mat[:3, :3],
            np.zeros((3))  # extrinsic_depth_to_col_mat[:3, 3]
        )

        # Mask depth image if apply_segmentation_mask is set
        masked_depth = depth_img
        if apply_segmentation_mask:
            masked_depth = np.where(np.asarray(mask_aligned_image).mean(axis=2) == 0, 0, np.asarray(depth_img))

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_aligned_img),
            o3d.geometry.Image(masked_depth),
            convert_rgb_to_intensity=False,
        )

        return rgbd_img

    def get_rgbd_for_frame_transformed_to_color(self, frame_number, apply_segmentation_mask=True):
        # Load images for frame
        images = self.get_image_paths_for_frame(frame_number, full_path=True)

        depth_img = o3d.io.read_image(images["depth_images"])
        rgb_img = o3d.io.read_image(images["colour_images"])
        mask_img = o3d.io.read_image(images["segmented_images"])

        transformation = self.get_extrinsic_color_to_depth_matrix()
        transformation[:3, 3] = -transformation[:3, 3]

        transformed_depth_image = AlignmentProcessor.align_depth_to_rgb_image(
            np.asarray(depth_img),
            np.asarray(rgb_img),
            self.get_intrinsic_depth_mat(),
            self.get_intrinsic_color_mat(),
            transformation
        )

        # Mask depth image if apply_segmentation_mask is set
        masked_depth = transformed_depth_image
        if apply_segmentation_mask:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            eroded_mask = cv2.erode(np.asarray(mask_img), kernel)
            masked_depth = np.where(np.asarray(eroded_mask) == 0, 0, np.asarray(transformed_depth_image))

        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_img),
            o3d.geometry.Image(masked_depth),
            convert_rgb_to_intensity=False,
        )

        return rgbd_img

    def get_extrinsic_depth_matrix(self):
        depth_r = np.reshape(self.camera_callibration["depth_extrinsics"]["orientation"], (3, 3))
        depth_t = self.camera_callibration["depth_extrinsics"]["translation"]
        mat = np.eye(4)
        mat[:3, :3] = depth_r
        mat[:3, 3] = depth_t
        return mat

    def get_intrinsic_depth(self):
        fx = self.camera_callibration["depth_intrinsics"]["fx"]
        fy = self.camera_callibration["depth_intrinsics"]["fy"]
        ppx = self.camera_callibration["depth_intrinsics"]["ppx"]
        ppy = self.camera_callibration["depth_intrinsics"]["ppy"]
        w = self.camera_callibration["depth_intrinsics"]["width"]
        h = self.camera_callibration["depth_intrinsics"]["height"]

        return fx, fy, ppx, ppy, w, h

    def get_intrinsic_depth_mat(self):
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

    def get_extrinsic_color_to_depth_matrix(self):
        color_r = np.reshape(self.camera_callibration["colour_to_depth_extrinsics"]["orientation"], (3, 3))
        color_t = self.camera_callibration["colour_to_depth_extrinsics"]["translation"]
        mat = np.eye(4)
        mat[:3, :3] = color_r
        mat[:3, 3] = color_t
        return mat

    def get_intrinsic_color(self):
        fx = self.camera_callibration["colour_intrinsics"]["fx"]
        fy = self.camera_callibration["colour_intrinsics"]["fy"]
        ppx = self.camera_callibration["colour_intrinsics"]["ppx"]
        ppy = self.camera_callibration["colour_intrinsics"]["ppy"]
        w = self.camera_callibration["colour_intrinsics"]["width"]
        h = self.camera_callibration["colour_intrinsics"]["height"]

        return fx, fy, ppx, ppy, w, h

    def get_intrinsic_color_mat(self):
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

    def get_colored_point_cloud_depth(self, frame_number):
        # Load intrinsics for depth camera
        depth_fx, depth_fy, depth_ppx, depth_ppy, depth_w, depth_h = self.get_intrinsic_depth()
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_w, depth_h, depth_fx, depth_fy, depth_ppx, depth_ppy
        )

        # Compute correct transformation
        extrinsic_depth_mat = self.get_extrinsic_depth_matrix()
        depth_final_transform = np.eye(4)
        depth_final_transform[:3, :3] = extrinsic_depth_mat[:3, :3].transpose()
        depth_final_transform[:3, 3] = extrinsic_depth_mat[:3, 3]

        # Transform images to depth camera and produce RGBD image
        rgbd_img = self.get_rgbd_for_frame_transformed_to_depth(frame_number, apply_segmentation_mask=True)

        # Create a new point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img,
            depth_intrinsics,
            np.eye(4),
            project_valid_depth_only=True
        )
        pcd.transform(depth_final_transform)

        return pcd

    def get_colored_point_cloud_color(self, frame_number):
        # Load intrinsics
        rgb_fx, rgb_fy, rgb_ppx, rgb_ppy, rgb_w, rgb_h = self.get_intrinsic_color()
        rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            rgb_w, rgb_h, rgb_fx, rgb_fy, rgb_ppx, rgb_ppy
        )

        # Compute correct transformations
        depth_transform, rgb_transform = compute_final_transform(
            self.get_extrinsic_depth_matrix(),
            self.get_extrinsic_color_to_depth_matrix()
        )

        # Transform images to depth camera and produce RGBD image
        rgbd_img = self.get_rgbd_for_frame_transformed_to_color(frame_number, apply_segmentation_mask=True)

        # Create a new point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_img,
            rgb_intrinsics,
            np.eye(4),
            project_valid_depth_only=True
        )
        pcd.transform(rgb_transform)

        return pcd

    def get_point_cloud_depth(self, frame_number, color):
        # Load intrinsics for depth camera
        depth_fx, depth_fy, depth_ppx, depth_ppy, depth_w, depth_h = self.get_intrinsic_depth()
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_w, depth_h, depth_fx, depth_fy, depth_ppx, depth_ppy
        )

        # Compute correct transformation
        extrinsic_depth_mat = self.get_extrinsic_depth_matrix()
        depth_final_transform = np.eye(4)
        depth_final_transform[:3, :3] = extrinsic_depth_mat[:3, :3].transpose()
        depth_final_transform[:3, 3] = extrinsic_depth_mat[:3, 3]

        # Create a new point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.io.read_image(self.get_image_paths_for_frame(frame_number, True)["depth_images"]),
            depth_intrinsics,
            np.eye(4),
            project_valid_depth_only=True
        )
        pcd.transform(depth_final_transform)
        pcd.paint_uniform_color(color)

        return pcd

    def get_camera_visualisation_line_set(self, color):
        # Compute correct transformations
        depth_transform, rgb_transform = compute_final_transform(
            self.get_extrinsic_depth_matrix(),
            self.get_extrinsic_color_to_depth_matrix()
        )

        # load intrinsics
        depth_fx, depth_fy, depth_ppx, depth_ppy, depth_w, depth_h = self.get_intrinsic_depth()
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            depth_w, depth_h, depth_fx, depth_fy, depth_ppx, depth_ppy
        )
        rgb_fx, rgb_fy, rgb_ppx, rgb_ppy, rgb_w, rgb_h = self.get_intrinsic_color()
        rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            rgb_w, rgb_h, rgb_fx, rgb_fy, rgb_ppx, rgb_ppy
        )

        # Add depth sensor vis
        depth_cam_vis = o3d.geometry.LineSet.create_camera_visualization(
            depth_intrinsics,
            np.eye(4),
            scale=0.3
        )
        depth_cam_vis.transform(depth_transform)
        depth_cam_vis.paint_uniform_color(color)

        # add rgb sensor vis
        col_cam_vis = o3d.geometry.LineSet.create_camera_visualization(
            rgb_intrinsics,
            np.eye(4),
            scale=0.3
        )
        col_cam_vis.transform(rgb_transform)
        col_cam_vis.paint_uniform_color(color)

        return depth_cam_vis, col_cam_vis
