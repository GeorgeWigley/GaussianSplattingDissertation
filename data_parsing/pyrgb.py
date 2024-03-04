import numpy as np


def unproject(u, v, d, fx, fy, cx, cy):
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    # for scalar and tensor
    return np.stack([x, y, d], axis=-1)


def project(x, y, z, fx, fy, cx, cy, with_depth=True):
    u = x * fx / z + cx
    v = y * fy / z + cy
    # for scalar and tensor
    if with_depth:
        return np.stack([u, v, z], axis=-1)
    return np.stack([u, v], axis=-1)


def depth2pc(depth, fx, fy, cx, cy, color=None, ignore_zero=True,
             keep_image_coord=False,
             distortion_type=None, distortion_param=[],
             distortion_interp='NN'):
    if depth.ndim != 2:
        raise ValueError()
    with_color = color is not None
    with_distortion = distortion_type is not None
    if ignore_zero:
        valid_mask = depth > 0
    else:
        valid_mask = np.ones(depth.shape, dtype=np.bool)
    invalid_mask = np.logical_not(valid_mask)
    h, w = depth.shape
    u = np.tile(np.arange(w), (h, 1))
    v = np.tile(np.arange(h), (w, 1)).T

    pc = unproject(u, v, depth, fx, fy, cx, cy)
    pc[invalid_mask] = 0

    pc_color = None
    if with_color:
        # interpolation for float uv by undistort_pixel
        if with_distortion and distortion_interp == 'NN':
            v, u = np.rint(v), np.rint(u)
        elif with_distortion:
            raise NotImplementedError('distortion_interp ' +
                                      distortion_interp +
                                      ' is not implemented')

        # 1) Make UV valid mask for color
        v_valid = np.logical_and(0 <= v, v < h)
        u_valid = np.logical_and(0 <= u, u < w)
        uv_valid = np.logical_and(u_valid, v_valid)

        # 2) Set stub value for outside of valid mask
        v[v < 0] = 0
        v[(h - 1) < v] = h - 1
        u[u < 0] = 0
        u[(w - 1) < u] = w - 1
        pc_color = color[v, u]

        # 3) Update valid_mask and invalid_mask
        valid_mask = np.logical_and(valid_mask, uv_valid)
        invalid_mask = np.logical_not(valid_mask)

        pc_color[invalid_mask] = 0

    # return as organized point cloud keeping original image shape
    if keep_image_coord:
        return pc, pc_color

    # return as a set of points
    return pc[valid_mask], pc_color[valid_mask]


def gen_mapped_color(depth, dfx, dfy, dcx, dcy,
                     color, cfx, cfy, ccx, ccy,
                     d2c_R, d2c_t,
                     ddist_type=None, ddist_param=[],
                     cdist_type=None, cdist_param=[],
                     cdist_interp='NN',
                     missing_color=[0, 0, 0]):
    # point cloud in depth camera coordinate
    dpc, _ = depth2pc(depth, dfx, dfy, dcx, dcy,
                      keep_image_coord=True,
                      distortion_type=ddist_type,
                      distortion_param=ddist_param)
    # Valid region is depth > 0
    valid_mask = dpc[..., 2] > 0
    dpc = dpc[valid_mask]

    # point cloud in color camera coordinate
    cpc = (d2c_R @ dpc.T).T + d2c_t

    # Project to color camera coordinate
    img_p = project(cpc[..., 0], cpc[..., 1], cpc[..., 2],
                    cfx, cfy, ccx, ccy, with_depth=False)
    u, v = img_p[..., 0], img_p[..., 1]

    v, u = np.rint(v).astype(int), np.rint(u).astype(int)

    dh, dw = depth.shape
    ch, cw, cc = color.shape

    # Guard if point cloud is projected to outside of color image
    # 1) Make UV valid mask for color
    v_valid = np.logical_and(0 <= v, v < ch)
    u_valid = np.logical_and(0 <= u, u < cw)
    uv_valid = np.logical_and(u_valid, v_valid)
    uv_invalid = np.logical_not(uv_valid)

    # 2) Set stub value for outside of valid mask
    v[v < 0] = 0
    v[(ch - 1) < v] = ch - 1
    u[u < 0] = 0
    u[(cw - 1) < u] = cw - 1

    # 3) Get color (with stub value)
    pc_color = color[v, u]

    # 4) Set missing_color for invalid region
    pc_color[uv_invalid] = missing_color

    # Prepare mapped color image
    mapped_color = np.zeros([dh, dw, cc], np.uint8)
    mapped_color[..., :] = missing_color

    # Update valid_mask
    # Set False for region where projection to color was failed
    all_false = np.zeros([dh, dw], bool)
    all_false[valid_mask] = uv_valid
    valid_mask = all_false

    # Set color to depth image coordinate
    # Note that this coordinate is BEFORE UNDISTORTION
    mapped_color[valid_mask] = pc_color[uv_valid]

    return mapped_color, valid_mask
