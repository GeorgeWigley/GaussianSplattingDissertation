import numpy as np


# Open3D uses openGL RH:  X right, Y up, Z inward
def convert_beween_opencv_opengl_view_mat(mat):
    element_wise_conversion_mat = np.array([[-1, -1, -1, -1],
                                            [1, 1, 1, 1],
                                            [-1, -1, -1, -1],
                                            [1, 1, 1, 1]])
    return np.multiply(mat, element_wise_conversion_mat)


def compute_final_transform(depth, depth_to_rgb):
    depth_final_transform = np.eye(4)
    depth_final_transform[:3, :3] = depth[:3, :3]
    depth_final_transform[:3, 3] = depth[:3, 3]

    depth_to_rgb_final_transform = np.eye(4)
    depth_to_rgb_final_transform[:3, :3] = depth_to_rgb[:3, :3]
    depth_to_rgb_final_transform[:3, 3] = depth_to_rgb[:3, 3]

    rgb_final_transform = depth_final_transform @ depth_to_rgb_final_transform

    return depth_final_transform, rgb_final_transform


def roll_down_2d_array(array, num_pixels):
    """Rolls a 2D NumPy array down by a specified number of pixels.

  Args:
    array: The 2D NumPy array to be rolled.
    num_pixels: The number of pixels to roll the array down.

  Returns:
    A new 2D NumPy array with the rolled elements.
  """

    if num_pixels == 0 or array.shape[0] == 0:
        return array.copy()  # No rolling needed or empty array

    height, width = array.shape

    # Handle cases where num_pixels is larger than the array height
    if num_pixels >= height:
        num_pixels = num_pixels % height  # Wrap around if necessary

    # Get the bottom and top parts of the array
    bottom_part = array[-num_pixels:, :]
    top_part = array[:-num_pixels, :]

    # Combine the bottom and top parts to form the rolled array
    rolled_array = np.concatenate((bottom_part, top_part), axis=0)

    return rolled_array


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
