import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


def analyze_dmap(img):
    """Apply CV algorithms to analyze the depth map and identify hail indents"""

    # Apply Gaussian blur (preprocessing to improve efficiency/accuracy)
    # blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold using blackhat transform
    filter_size = (50, 50)
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)

    blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, blackhat_kernel)

    # Increase blackhat contrast to improve detection of finer details
    accentuated = cv2.convertScaleAbs(blackhat_img, alpha=2, beta=0)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("blackhat", blackhat_img)
    cv2.waitKey(0)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("accentuated", accentuated)
    cv2.waitKey(0)

    # Filter out shallow artifacts by depth (e.g., small or non-hail indents)
    depth_threshold = 0
    filtered_blackhat = np.where(
        blackhat_img < depth_threshold, 0, blackhat_img)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("blackhat (filtered)", filtered_blackhat)
    cv2.waitKey(0)

    # Apply binary threshold
    _, binary_img = cv2.threshold(
        accentuated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # TODO?: Change to filtered_blackhat

    # Subtract difference between dilation and erosion transforms to remove noise and hailpad edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    binary_img = cv2.morphologyEx(binary_img,
                                  cv2.MORPH_OPEN,
                                  kernel,
                                  iterations=1)
    

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("noise removal (dilation - erosion)", binary_img)
    cv2.waitKey(0)

    binary_img = cv2.erode(binary_img, kernel, iterations=2)
    binary_img = cv2.dilate(binary_img, kernel, iterations=2)
    cv2.imshow("fr", binary_img)
    cv2.waitKey(0)

    # sure_bg = cv2.dilate(binary_img, kernel, iterations=3)

    # # FOR TESTING PURPOSES (TODO: REMOVE)
    # cv2.imshow("sure bg", sure_bg)
    # cv2.waitKey(0)

    # dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(
    #     dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
    # sure_fg = sure_fg.astype(np.uint8)
    # unknown = cv2.subtract(sure_bg, sure_fg)

    # # FOR TESTING PURPOSES (TODO: REMOVE)
    # cv2.imshow("unknown area", unknown)
    # cv2.waitKey(0)

    # Apply watershed algorithm
    ret, markers = cv2.connectedComponents(binary_img) # TODO: Change to sure_bg
    markers += 1
    # markers[unknown == 255] = 0
    # Convert back to 3-channel BGR image for watershed
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # FOR TESTING PURPOSES (TODO: REMOVE)
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(markers, cmap="tab20b")
    # ax.axis('off')
    # plt.show()

    # Get the unique labels from the markers
    labels = np.unique(markers)

    # Initialize a list to store the component statistics
    component_stats = []

    # Loop through each label to get the component statistics
    for label in labels:
        marker_binary = np.where(markers == label, 255, 0).astype('uint8')
        num_labels, label_im, stats, centroid = cv2.connectedComponentsWithStats(
            marker_binary, 4, cv2.CV_32S)
        component_stats.append((num_labels, label_im, stats, centroid))

    indents = []

    # Output the statistics of each component
    for component in component_stats:
        num_labels, label_im, stats, centroid = component

        area = stats[1, cv2.CC_STAT_AREA]
        width = stats[1, cv2.CC_STAT_WIDTH]
        height = stats[1, cv2.CC_STAT_HEIGHT]

        # Calculate major and minor axes by determining the longest and shortest distances

        # TODO: Remove (write new algo to calculate major and minor axes)
        # major_axis = 0
        # minor_axis = 0

        # # Identify major and minor axes based on component width and height values
        # if width > height:
        #     major_axis = width
        #     minor_axis = height
        # else:
        #     major_axis = height
        #     minor_axis = width

        # Calculate the depth value at the centroid of the component
        y = int(centroid[1][0])
        x = int(centroid[1][1])
        cent_depth = img[y, x][0] / 255.0  # Normalize depth value between 0-1

        # Calculate the average and largest depth values of the component
        mask = label_im == 1
        avg_depth = img[mask].mean() / 255.0
        max_depth = img[mask].max() / 255.0

        indents.append({
            "area": area,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "centroid": {
                "y": y,
                "x": x
            },
            "depth_at_centroid": cent_depth,
            "avg_depth": avg_depth,
            "max_depth": max_depth
        })

    return indents
