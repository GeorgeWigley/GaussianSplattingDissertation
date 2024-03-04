
class Image:
    def __init__(self, image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, file_name):
        self.image_id = image_id
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.camera_id = camera_id
        self.file_name = file_name
        self.points_2d = []

    def serialize_text(self):
        line_one = f"{self.image_id} {self.qw} {self.qx} {self.qy} {self.qz} {self.tx} {self.ty} {self.tz} {self.camera_id} {self.file_name}"
        line_two = ""
        for point in self.points_2d:
            line_two += f"{point.x} {point.y} {point.point_3d_id} "
        return line_one, line_two


    def serialize_bin(self):
        raise NotImplementedError
        return None
