import cv2

from ColmapClasses import Image
from ColmapClasses import ColmapCamera
from ColmapClasses.Point2D import Point2D
from ColmapClasses import Point3D
import os
from tqdm import tqdm
import open3d as o3d


# This class must be run in the following order:
# Add cameras
# Add point 3ds
# Optionally Merge points (unimplemented)
# Add images
# Update References ensuring that the image id remains consistent
# Serialize to either Text or Bin


class COLMAPScene:

    def __init__(self):
        self.name = ""
        self.cameras = []
        self.points_3d = []
        self.images = []
        self.images_data = []

    def add_camera(self, camera):
        self.cameras.append(camera)

    def add_image(self, image, transformed_image_object=None):
        self.images.append(image)
        if transformed_image_object is not None:
            self.images_data.append(transformed_image_object)

    def add_3d_point(self, point_3d):
        # should be a 3d point containing standard information AND all image references
        self.points_3d.append(point_3d)

    def update_references(self):
        # for each 3D point
        for point_3d in tqdm(self.points_3d):
            # Update the corresponding 2D image
            for image_ref in point_3d.image_references:
                # create new point 2D for image
                new_point_2d = Point2D(image_ref.x, image_ref.y, point_3d.point_3d_id)
                # add it to image
                self.images[image_ref.image_id - 1].points_2d.append(new_point_2d)
                # point 2d idx is the count of the point in that image
                point_2d_idx = len(self.images[image_ref.image_id - 1].points_2d) - 1
                # update point 3d to point to correct point 2d
                image_ref.point_2d_idx = point_2d_idx

    def serialize_scene_to_text(self, output_dir, serialize_images=False):
        # create paths
        spare_model_path = os.path.join(output_dir, "sparse/0")
        images_path = os.path.join(output_dir, "images")

        # create dirs
        os.makedirs(spare_model_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)

        # write cameras
        with open(os.path.join(spare_model_path, "cameras.txt"), 'w') as f:
            for camera in self.cameras:
                f.write(camera.serialize_text() + '\n')

        # write images
        with open(os.path.join(spare_model_path, "images.txt"), 'w') as f:
            # for image in self.images:
            for image in self.images:
                line_one, line_two = image.serialize_text()
                f.write(line_one + '\n' + line_two + '\n')

        # write points
        with open(os.path.join(spare_model_path, "points3D.txt"), 'w') as f:
            for point in self.points_3d:
                f.write(point.serialize_text() + '\n')

        # Serialize transformed images
        if serialize_images:
            for x, image_object in enumerate(self.images_data):
                cv2.imwrite(os.path.join(images_path, self.images[x].file_name),
                            cv2.cvtColor(image_object, cv2.COLOR_BGR2RGB))


