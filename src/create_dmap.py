import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


def create_dmap(mesh):
    """Process a 3D mesh (.stl) to create a 1000x1000 depth map image"""

    # Render mesh (FOR TESTING PURPOSES -- TODO: REMOVE)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh], window_name = "Hailpad Mesh", width = 750, height = 750)

    # Create point cloud from mesh (set an arbitrary seed for reproducibility)
    np.random.seed(0)
    pcd = mesh.sample_points_uniformly(number_of_points=1000000)

    # Fit minimal bounding box
    bb = pcd.get_minimal_oriented_bounding_box()

    # Translate by bounding box center
    pcd.translate(-1.0 * bb.center)

    # Rotate by bounding box rotation matrix (take the transpose so the rotation aligns the pcd with the axes)
    pcd.rotate(bb.R.T)

    # Remove hidden (not visible) points
    diameter = np.linalg.norm(np.asarray(
        pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    camera = [0, 0, diameter]
    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    pcd = pcd.select_by_index(pt_map)

    # Render point cloud of hail pad surface (FOR TESTING PURPOSES -- TODO: REMOVE)
    o3d.visualization.draw_geometries([pcd],
                                      zoom = 0.7,
                                      front = [0, 0, 1],
                                      lookat = [0, 0, 0],
                                      up = [0, 1, 0],
                                      window_name = "Hailpad Point Cloud",
                                      width = 750, height = 750)

    # Convert to numpy array
    pts = np.asarray(pcd.points)

    # Visualize raw data points in the traditional xyz plane (FOR TESTING PURPOSES -- TODO: REMOVE)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()

    N = 1000  # Depth map resolution (for creating an N x N grid)

    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    max_range = max(x_range, y_range)
    N_x = int(N * x_range / max_range)
    N_y = int(N * y_range / max_range)

    x = np.linspace(pts[:, 0].min(), pts[:, 0].max(), N_x)
    y = np.linspace(pts[:, 1].min(), pts[:, 1].max(), N_y)

    # Linearly interpolate over grid to create depth map image
    interp = LinearNDInterpolator(list(zip(pts[:, 0], pts[:, 1])), pts[:, 2])

    z = np.zeros((N_y, N_x))  # Initialize N x N grid for z-coordinates

    for i in range(N_y):
        for j in range(N_x):
            z[i, j] = interp(x[j], y[i])

    # Fix NAN values of areas outside of pad since pad is not a perfect square
    z = np.nan_to_num(z, nan=-9999.0)
    z[z == -9999.0] = z.max()

    # Normalize depth values between 0-1.0 (required for using as an image)
    z -= z.min()
    dmap = z / z.max()

    # Show depth image (FOR TESTING PURPOSES -- TODO: REMOVE)
    # cv2.imshow("Depth Map", dmap)
    # cv2.waitKey(0)

    # Save as regular image (convert from float to byte)
    dmap *= 255.0
    dmap = dmap.astype(np.uint8)
    dmap = cv2.flip(dmap, 1)

    # Create a 1000x1000 white image
    white_img = np.full((1000, 1000), 255, dtype=np.uint8)

    # Calculate the start indices for pasting the depth map image onto the white image
    start_x = (1000 - dmap.shape[1]) // 2
    start_y = (1000 - dmap.shape[0]) // 2

    # Paste the depth map image onto the white image
    white_img[start_y:start_y+dmap.shape[0], start_x:start_x+dmap.shape[1]] = dmap

    cv2.imwrite("src/images/dmap.png", white_img)

    return white_img
