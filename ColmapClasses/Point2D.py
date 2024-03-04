class Point2D:
    def __init__(self, x, y, point_3d_id=-1):
        """
        Defines a feature point on an image plane and its corresponding 3D point
        :param x: x coordinate of point
        :param y: y coordinate of point
        :param point_3d_id: ID of the corresponding 3D point
        """
        self.x = x
        self.y = y
        self.point_3d_id = point_3d_id

    def set_3d_point(self, point_3d_id):
        self.point_3d_id = point_3d_id
