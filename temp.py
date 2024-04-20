import numpy as np

o3d_mat = np.array([
    [0.99, 0.00, 0.10, 0.00],
    [-0.01, 1.00, 0.01, 0.03],
    [-0.10, -0.01, 0.99, -0.01],
    [0.00, 0.00, 0.00, 1.00],
])

colmap_mat = np.array([
    [1.00, 0.00, 0.00, -4.85],
    [0.00, 1.00, 0.00, -2.76],
    [0.00, 0.00, 1.00, 2.12],
    [0.00, 0.00, 0.00, 1.00]
])


def compute_relative_pose(M_A, M_B):
    # Extract rotation and translation components
    R_A, T_A = M_A[:3, :3], M_A[:3, 3]
    R_B, T_B = M_B[:3, :3], M_B[:3, 3]

    # Compute inverse of R_A
    P = np.linalg.inv(R_A)

    # Construct relative pose matrix
    M_rel = np.eye(4)
    M_rel[:3, :3] = np.dot(R_B, P)
    M_rel[:3, 3] = np.dot(P, -T_A) + T_B

    return M_rel


colmap_to_o3d = np.array([
    [0.99, 0., 0.1, 4.85],
    [-0.01, 1., 0.01, 2.79],
    [-0.1, -0.01, 0.99, -2.13],
    [0., 0., 0., 1.]
])

print(compute_relative_pose(colmap_mat, o3d_mat))
