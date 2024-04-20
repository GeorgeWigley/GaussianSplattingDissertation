import numpy as np
import cv2


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


def crop_black_border(image, fx, fy, ppx, ppy):
    """
  Crops a black border from an image.

  Args:
      image: The input image.
      fx: The focal length x.
      fy: The focal length y.
      ppx: The principal point x.
      ppy: The principal point y.

  Returns:
      A tuple containing:
         - cropped_image: The cropped image.
         - w: Width of the cropped region
         - h: Height of the cropped region
         - cropped_fx: The updated focal length x (single value).
         - cropped_fy: The updated focal length y (single value).
         - cropped_ppx: The updated principal point x (single value).
         - cropped_ppy: The updated principal point y (single value).
  """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find non-zero pixels (assuming black border is 0)
    mask = cv2.findNonZero(gray)

    # Get bounding rectangle of non-zero pixels
    x, y, w, h = cv2.boundingRect(mask)

    # Crop the image
    cropped_image = image[y:y + h, x:x + w]

    # Update intrinsics
    cropped_fx = fx * (w / image.shape[1])
    cropped_fy = fy * (h / image.shape[0])
    cropped_ppx = ppx - x
    cropped_ppy = ppy - y

    return cropped_image, w, h, cropped_fx, cropped_fy, cropped_ppx, cropped_ppy


def remove_rectification_border(image, fx, fy, ppx, ppy):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    center_x = np.shape(thresh)[1] // 2
    center_y = np.shape(thresh)[0] // 2

    tl_x = 0
    tl_y = 0
    br_x = np.shape(thresh)[1]
    br_y = np.shape(thresh)[0]

    # find rightmost col containing black pixel for left half of image
    for x in range(center_x - int(0.75 * center_x), 0, -1):
        if np.any(thresh[center_y - int(center_y * 0.75): center_y + int(center_y * 0.75), x] == 0):
            tl_x = x
            break

    # find leftmost col containing black pixel for right half of image
    for x in range(center_x + int(0.75 * center_x), np.shape(thresh)[1]):
        if np.any(thresh[center_y - int(center_y * 0.75): center_y + int(center_y * 0.75), x] == 0):
            br_x = x
            break

    # find lowest row containing black pixel for top half of image
    for y in range(center_y - int(0.75 * center_y), 0, -1):
        if np.any(thresh[y, center_x - int(center_x * 0.75): center_x + int(center_x * 0.75)] == 0):
            tl_y = y
            break

    # find highest row containing black pixel for bot half of image
    for y in range(center_y + int(0.75 * center_y), np.shape(thresh)[0]):
        if np.any(thresh[y, center_x - int(center_x * 0.75): center_x + int(center_x * 0.75)] == 0):
            br_y = y
            break

    # crop to bounding box calculated from previous steps
    cropped_image = image[tl_y:br_y, tl_x:br_x]

    # Update intrinsics
    cropped_fx = fx * ((br_x - tl_x) / image.shape[1])
    cropped_fy = fy * ((br_y - tl_y) / image.shape[0])
    cropped_ppx = ppx - tl_x
    cropped_ppy = ppy - tl_y

    return cropped_image, np.shape(cropped_image)[1], np.shape(cropped_image)[0], cropped_fx, cropped_fy, cropped_ppx, cropped_ppy


