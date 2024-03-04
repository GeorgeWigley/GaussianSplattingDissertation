
class ColmapCamera:
    def __init__(self, cameraId, camera_type, sensor_width, sensor_height, fx, fy, cx, cy):
        self.camera_id = cameraId
        self.camera_type = camera_type
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def serialize_text(self):
        result = f"{self.camera_id} {self.camera_type} {self.sensor_width} {self.sensor_height} {self.fx} {self.fy} {self.cx} {self.cy}"
        return result

    def serialize_bin(self):
        raise NotImplementedError
        return None
