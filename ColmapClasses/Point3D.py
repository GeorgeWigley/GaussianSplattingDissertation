class PointImageReference:
    def __init__(self, image_id, x, y, point_2d_idx=-1):
        self.image_id = image_id
        self.x = x
        self.y = y
        self.point_2d_idx = point_2d_idx


class Point3D:
    def __init__(self, point_3d_id, x, y, z, r, g, b):
        self.point_3d_id = point_3d_id
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.b = b
        self.g = g
        self.image_references = []

    def add_image_reference(self, image_reference):
        self.image_references.append(image_reference)

    def serialize_text(self):
        result = f"{self.point_3d_id} {self.x} {self.y} {self.z} {self.r} {self.g} {self.b} 0 "
        for ref in self.image_references:
            result += f"{ref.image_id} {ref.point_2d_idx}"
        return result
