import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

def main():
    # # Import mesh
    # mesh = o3d.io.read_triangle_mesh("src/meshes/hailpad.stl")

    # # Render mesh
    # # mesh.compute_vertex_normals()
    # # o3d.visualization.draw_geometries(
    # #     [mesh], window_name = "Hailpad Mesh", width = 750, height = 750)

    # # Create point cloud from mesh (set an arbitrary seed for reproducibility)
    # np.random.seed(0)
    # pcd = mesh.sample_points_uniformly(number_of_points = 1000000)

    # # Fit minimal bounding box
    # bb = pcd.get_minimal_oriented_bounding_box()

    # # Translate by bounding box center
    # pcd.translate(-1.0 * bb.center)

    # # Rotate by bounding box rotation matrix (take the transpose so the rotation aligns the pcd with the axes)
    # pcd.rotate(bb.R.T)

    # # Remove hidden (not visible) points
    # diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    # camera = [0, 0, diameter]
    # radius = diameter * 100
    # _, pt_map = pcd.hidden_point_removal(camera, radius)

    # pcd = pcd.select_by_index(pt_map)

    # # Render point cloud of hail pad surface
    # # o3d.visualization.draw_geometries([pcd],
    # #                                   zoom = 0.7,
    # #                                   front = [0, 0, 1],
    # #                                   lookat = [0, 0, 0],
    # #                                   up = [0, 1, 0],
    # #                                   window_name = "Hailpad Point Cloud",
    # #                                   width = 750, height = 750)
    
    # # Convert to numpy array
    # pts = np.asarray(pcd.points)

    # # Visualize raw data points in the traditional xyz plane
    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection = '3d')
    # # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # ax.set_zlabel('Z')

    # # plt.show()

    # N = 1000 # Depth map resolution (for creating an N x N grid)
    # x = np.linspace(pts[:, 0].min(), pts[:, 0].max(), N)
    # y = np.linspace(pts[:, 1].min(), pts[:, 1].max(), N)

    # # Linearly interpolate over grid to create depth map image
    # interp = LinearNDInterpolator(list(zip(pts[:, 0], pts[:, 1])), pts[:, 2])

    # z = np.zeros((N,N)) # Initialize N x N grid for z-coordinates

    # for i in range(N):
    #     for j in range(N):
    #         z[i,j] = interp(x[j], y[i])

    # # Fix NAN values of areas outside of pad since pad is not a perfect square
    # z = np.nan_to_num(z, nan = -9999.0)
    # z[z == -9999.0] = z.max()

    # # Normalize depth values between 0-1.0 (required for using as an image)
    # z -= z.min()
    # dmap = z / z.max()

    # # Show depth image
    # # cv2.imshow("Depth Map", dmap)
    # # cv2.waitKey(0)

    # # Save as regular image (convert from float to byte)
    # dmap *= 255.0
    # dmap = dmap.astype(np.uint8)
    # dmap = cv2.flip(dmap, 1)
    # cv2.imwrite("src/images/dmap.png", dmap)

    # Load image
    img = cv2.imread("src/images/dmap.png", cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur (preprocessing to improve efficiency/accuracy)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold (old)
    # threshold = cv2.threshold(blurred, 0, 255,
    #                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # CLAHE preprocessing (increase contrast; clipLimit is contrast limit for equalization) (TESTING)
    # clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    # cl1 = clahe.apply(blurred)

   # xx block_size = 15 # Pixel neighbourhood size
   # xx c = 1 # Lower c corresponds to more lenient thresholding
    
    # Threshold using blackhat transform
    filter_size = (50,50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)

    blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    cv2.imshow("blackhat", blackhat_img)
    cv2.waitKey(0)

    # Filter out shallow artifacts by depth (e.g., small or non-hail indents)
    depth_threshold = 20
    filtered_blackhat = np.where(blackhat_img < depth_threshold, 0, blackhat_img)

    cv2.imshow("blackhat (filtered)", filtered_blackhat)
    cv2.waitKey(0)

    # Apply adaptive threshold
    # xx threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c)
    
    # TESTING
    # xx cv2.imshow("threshold (adaptive)", threshold)
    cv2.waitKey(0)

    # Apply binary threshold
    _, binary_img = cv2.threshold(filtered_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply erosion to remove the edges of the hailpad
    eroded_binary = cv2.erode(binary_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations = 1)

    cv2.imshow("eroded", eroded_binary)
    cv2.waitKey(0)

    # Apply the component analysis function
    analysis = cv2.connectedComponentsWithStats(eroded_binary, 4, cv2.CV_32S)

    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to store all the output components
    output = np.zeros(img.shape, dtype = "uint8")

    # Set the area, width, and height bounds for component filtering
    min_area = 140
    max_area = 700
    max_width = 200
    max_height = 200

    # Loop through each component
    for i in range(1, totalLabels):
        # Get the area, width, and height of the i-th connected component
        area = values[i, cv2.CC_STAT_AREA]
        width = values[i, cv2.CC_STAT_WIDTH]
        height = values[i, cv2.CC_STAT_HEIGHT]

        # Filter the connected components to the specified bounds
        if area > min_area:
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)

    cv2.imshow("Filtered Components", output)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
