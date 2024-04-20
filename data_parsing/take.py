import os.path

import cv2
import quaternion
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

import utils
from ColmapClasses.COLMAPScene import COLMAPScene
from ColmapClasses.ColmapCamera import ColmapCamera
from ColmapClasses.Image import Image
from ColmapClasses.Point3D import Point3D, PointImageReference
from camera import Camera
from utils import rotmat2qvec


class Take:

    def __init__(self, take_path, callibration_path):
        """
        A take is a single recording of a scene containing n cameras
        :param take_path: path to the folder containing a Take
        """
        # set take name
        self.take_name = os.path.split(take_path)[-1]

        # load a camera for each take
        self.cameras = {}
        list_subfolders_with_paths = [f.path for f in os.scandir(take_path) if f.is_dir()]
        for camera in list_subfolders_with_paths:
            loaded_camera = Camera(camera, callibration_path)
            self.cameras[loaded_camera.serial_id] = loaded_camera

        self.frame_count = len(self.cameras[list(self.cameras.keys())[0]].get_available_frames())

    def serialize_to_colmap_txt_transformed_to_color(self, frame_number, folder_path):
        # because of the many-to-many relationships in the COLMAP format, the scene must be created as follows:
        # 1 - Add cameras
        # 2 - Add point 3ds
        # 3 - Add images
        # 4 - Update References ensuring that the image id remains consistent
        # 5 - Serialize to either Text or Bin

        # ID for each component of the COLMAP scene
        point_3d_id = 1
        camera_id = 1
        image_id = 1

        # create a colmap scene for serialization
        scene = COLMAPScene()

        # 1 - Add cameras
        camera_ids = list(self.cameras.keys())
        temp_camera_ids = ['000236320812', '000404613112', '000407313112', '000558313112', '000951614712']

        for x, camera in enumerate(temp_camera_ids):
            rgb_fx, rgb_fy, rgb_ppx, rgb_ppy, rgb_w, rgb_h = self.cameras[camera].get_colour_k()
            rgb_image_filename = self.cameras[camera].get_image_paths_for_frame(frame_number, False)["colour_images"]
            rgb_img = o3d.io.read_image(
                self.cameras[camera].get_image_paths_for_frame(frame_number, True)["colour_images"])
            mask_img = o3d.io.read_image(
                self.cameras[camera].get_image_paths_for_frame(frame_number, True)["segmented_images"])

            # Apply the mask
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            # eroded_mask = cv2.erode(np.asarray(mask_img), kernel)
            # masked_rgb_img = np.asarray(rgb_img).copy()
            # masked_rgb_img[np.asarray(eroded_mask) == 0] = (0, 0, 0)
            # masked_rgb_img = o3d.geometry.Image(masked_rgb_img)
            masked_rgb_img = rgb_img
            # masked_rgb_img, rgb_w, rgb_h, rgb_fx, rgb_fy, rgb_ppx, rgb_ppy = utils.remove_rectification_border(np.asarray(rgb_img), rgb_fx, rgb_fy, rgb_ppx, rgb_ppy)
            # masked_rgb_img = utils.remove_rectification_border(np.asarray(rgb_img))

            cam = ColmapCamera(
                camera_id,
                "PINHOLE",
                rgb_w,
                rgb_h,
                rgb_fx,
                rgb_fy,
                rgb_ppx,
                rgb_ppy
            )

            scene.add_camera(cam)

            # Create point cloud
            pcd = self.cameras[camera].get_colored_point_cloud_color(frame_number).voxel_down_sample(0.03)

            # reproject points to determine which pixel they were projected from
            # pcd_pixel_indices = []
            # intrinsic_matrix = self.cameras[camera].get_colour_k_mat()
            # for point in pcd.points:
            #     uv = intrinsic_matrix @ point  # Project 3D point to 2D
            #     u = int(uv[0] / uv[2])
            #     v = int(uv[1] / uv[2])
            #     pcd_pixel_indices.append([u, v])

            # 2 - Add point 3ds with tracks
            for p in range(len(pcd.points)):
                point_3d = Point3D(
                    point_3d_id,
                    pcd.points[p][0],
                    pcd.points[p][1],
                    pcd.points[p][2],
                    int(pcd.colors[p][0] * 255),
                    int(pcd.colors[p][1] * 255),
                    int(pcd.colors[p][2] * 255),
                )
                # image_ref = PointImageReference(
                #     image_id,
                #     pcd_pixel_indices[x][0],
                #     pcd_pixel_indices[x][1]
                # )
                # point_3d.add_image_reference(image_ref)
                scene.add_3d_point(point_3d)

                point_3d_id += 1

            # 3 - Add images
            w2c = self.cameras[camera].get_colour_w2c()

            # Convert rotation matrix to quaternion
            quaternion_obj = quaternion.from_rotation_matrix(w2c[:3, :3])
            # Extract COLMAP quaternion components
            qw = quaternion_obj.w
            qx = quaternion_obj.x
            qy = quaternion_obj.y
            qz = quaternion_obj.z

            translation = w2c[:3, 3]

            image = Image(
                image_id,
                qw,
                qx,
                qy,
                qz,
                translation[0],
                translation[1],
                translation[2],
                camera_id,
                rgb_image_filename
            )
            scene.add_image(image, np.asarray(masked_rgb_img))

            camera_id += 1
            image_id += 1

        # 4 - Update References
        scene.update_references()

        # 5 - Serialize
        scene.serialize_scene_to_text(folder_path, True)

    def render_scene_visualisation(self, frame_number, use_color=True):
        sample_cols = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0.5, 0], [0.5, 0, 1],
                       [0.5, 1, 0], [0, 0.5, 0.5]]
        camera_ids = list(self.cameras.keys())
        temp_camera_ids = ['000236320812', '000404613112', '000407313112', '000558313112', '000951614712']
        idz = ['000951614712']

        pcds = []
        cameras = []
        labels = []

        for x, camera in enumerate(idz):
            # Create camera line sets for visualisation
            camera_visualisation = self.cameras[camera].get_camera_visualisation_line_set(sample_cols[x])
            cameras.append(camera_visualisation[0])
            cameras.append(camera_visualisation[1])
            labels.append([camera_visualisation[0].get_center(), f"{camera}"])

            # Create point cloud for visualisation
            if use_color:
                pcd = self.cameras[camera].get_colored_point_cloud_color(frame_number)
            else:
                pcd = self.cameras[camera].get_point_cloud_depth(frame_number, sample_cols[x])
            pcds.append(pcd)

        # combine
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in pcds:
            combined_pcd += pcd

        combined_pcd = combined_pcd.voxel_down_sample(0.05)
        # Setup visualization
        o3d.visualization.gui.Application.instance.initialize()

        window = o3d.visualization.gui.Application.instance.create_window("George Wigley - Viewer", 1400, 800)

        scene = o3d.visualization.gui.SceneWidget()
        scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

        window.add_child(scene)

        scene.scene.add_geometry(
            "axis",
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25),
            o3d.visualization.rendering.MaterialRecord()
        )

        scene.scene.add_geometry("pcd", combined_pcd, o3d.visualization.rendering.MaterialRecord())
        o3d.visualization.RenderOption.point_size = 1.0

        # for x, cam in enumerate(cameras):
        #     scene.scene.add_geometry(f"cam_{x}", cam, o3d.visualization.rendering.MaterialRecord())

        bounds = combined_pcd.get_axis_aligned_bounding_box()
        scene.setup_camera(60, bounds, bounds.get_center())

        # for label in labels:
        #     scene.add_3d_label(label[0], label[1])

        # TEMP add cameras for colmap poses
        # def quaternion_to_rotation_matrix(qw, qx, qy, qz, tx, ty, tz):
        #     # Compute rotation matrix
        #     R = np.array([
        #         [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        #         [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        #         [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
        #     ])
        #
        #     # Create translation vector
        #     T = np.array([tx, ty, tz])
        #
        #     # Construct 4x4 extrinsic matrix
        #     M = np.eye(4)
        #     M[:3, :3] = R
        #     M[:3, 3] = T
        #
        #     return M
        #
        # colmap_to_o3d = np.array([
        #     [0.99, 0., 0.1, 4.85],
        #     [-0.01, 1., 0.01, 2.79],
        #     [-0.1, -0.01, 0.99, -2.13],
        #     [0., 0., 0., 1.]
        # ])
        #
        # with open(r"C:\Users\georg\Documents\DissertationGeorgeWigley\colmap_w_me2\rembg_attempt\sparse\0\images.txt",
        #           'r') as f:
        #     for x, line in enumerate(f):
        #         if x % 2 == 0:
        #             tokens = line.split(" ")
        #             qw, qx, qy, qz, tx, ty, tz = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7]),
        #             extrinsic = quaternion_to_rotation_matrix(qw, qx, qy, qz, tx, ty, tz)
        #             intrinsics = o3d.camera.PinholeCameraIntrinsic(
        #                 1440, 2560, 1228.4296521734339, 1211.2472899099009, 737, 1270
        #             )
        #             colmap_cam_vis = o3d.geometry.LineSet.create_camera_visualization(
        #                 intrinsics,
        #                 extrinsic,
        #                 scale=0.3
        #             )
        #             colmap_cam_vis.paint_uniform_color([0, 0, 0])
        #             scene.scene.add_geometry(f"colmap_cam_{x}", colmap_cam_vis, o3d.visualization.rendering.MaterialRecord())
        #             print(tokens[-1])
        #             print(extrinsic)



        o3d.visualization.gui.Application.instance.run()  # Run until user closes window


test = Take(r"C:\Users\georg\Documents\DissertationGeorgeWigley\data\take2",
            r"C:\Users\georg\Documents\DissertationGeorgeWigley\data\calibration.json")
# test.serialize_to_colmap_txt_transformed_to_color(180, "C:/Users/georg/Documents/DissertationGeorgeWigley/test3")
test.render_scene_visualisation(11, True)
