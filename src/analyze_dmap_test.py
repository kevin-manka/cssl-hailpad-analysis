import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from scipy.interpolate import LinearNDInterpolator


def analyze_dmap_test(img):
    """Apply CV algorithms to analyze the depth map and identify hail indents"""

    # Apply blackhat transform
    filter_size = (50, 50)
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)
    blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, blackhat_kernel)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("blackhat", blackhat_img)
    cv2.waitKey(0)

    # Separate out shallow artifacts by depth
    depth_threshold = 17 # TODO: Make adjustable
    blackhat_bg = np.where(
        blackhat_img > depth_threshold, 0, blackhat_img)
    
    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("blackhat bg", blackhat_bg)
    cv2.waitKey(0)

    _, binary_bg = cv2.threshold(blackhat_bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("bg -- binary", binary_bg)
    cv2.waitKey(0)

    # Erode binary background to isolate large artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    eroded_bg = cv2.erode(binary_bg, kernel, iterations=1)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("bg -- eroded", eroded_bg)
    cv2.waitKey(0)

    # Filter out small artifacts from eroded background
    contours, _ = cv2.findContours(eroded_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(eroded_bg.shape, dtype="uint8") * 255

    for contour in contours:
        area = cv2.contourArea(contour)
        width = cv2.boundingRect(contour)[2]
        height = cv2.boundingRect(contour)[3]
        if area < 500:
            cv2.drawContours(mask, [contour], -1, 0, -1)

    eroded_filtered_bg = cv2.bitwise_and(eroded_bg, eroded_bg, mask=mask)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("bg -- eroded filtered", eroded_filtered_bg)
    cv2.waitKey(0)

    # Dilate the filtered background
    dilated_bg = cv2.dilate(eroded_filtered_bg, kernel, iterations=1)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("bg -- dilated", dilated_bg)
    cv2.waitKey(0)

    # Apply contrast limited adaptive histogram equalization to even out values
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # clahe_img = clahe.apply(blackhat_img)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    # cv2.imshow("clahe", clahe_img)
    # cv2.waitKey(0)

    # Apply adaptive thresholding
    adaptive_binary = cv2.adaptiveThreshold(blackhat_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -4)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("adaptive binary", adaptive_binary)
    cv2.waitKey(0)

    # Subtract the dilated background from the adaptive binary image
    adaptive_binary_filtered = cv2.subtract(adaptive_binary, dilated_bg)
    
    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("adaptive binary filtered", adaptive_binary_filtered)
    cv2.waitKey(0)

    # Apply morphological operations to remove noise and hailpad edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_img = cv2.morphologyEx(
        adaptive_binary_filtered, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("binary img", binary_img)
    cv2.waitKey(0)

    # Watershed data vis (TODO: Remove)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    def imshow(img, ax=None):
        if ax is None:
            ret, encoded = cv2.imencode(".jpg", img)
            display(Image(encoded))
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')

    sure_bg = cv2.dilate(binary_img, kernel, iterations=3)
    imshow(sure_bg, axes[0, 0])
    axes[0, 0].set_title("Sure Background")

    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    imshow(dist, axes[0, 1])
    axes[0, 1].set_title("Distance Transform")

    ret, sure_fg = cv2.threshold(
        dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    imshow(sure_fg, axes[1, 0])
    axes[1, 0].set_title("Sure Foreground")

    unknown = cv2.subtract(sure_bg, sure_fg)
    imshow(unknown, axes[1, 1])
    axes[1, 1].set_title("Unknown Area")

    plt.show()

    # ret, markers = cv2.connectedComponents(sure_fg)
    # markers += 1
    # markers[unknown == 255] = 0
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(markers, cmap='tab20b')
    # ax.axis('off')
    # ax.set_title("Markers")
    # plt.show()

    # # Apply watershed algorithm
    # ret, markers = cv2.connectedComponents(binary_img)
    # markers += 1
    # # markers[unknown == 255] = 0
    # # Convert back to 3-channel BGR image for watershed
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]

    # Apply watershed algorithm
    ret, markers = cv2.connectedComponents(binary_img)
    markers += 1
    # markers[unknown == 255] = 0
    # Convert back to 3-channel BGR image for watershed
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("watershed", img)
    cv2.waitKey(0)

    # Get the unique labels from the markers
    labels = np.unique(markers)

    # Initialize a list to store the component statistics
    component_stats = []

    count = 1

    # Loop through each label to get the component statistics
    for label in labels:
        marker_binary = np.where(markers == label, 255, 0).astype('uint8')
        num_labels, label_im, stats, centroid = cv2.connectedComponentsWithStats(
            marker_binary, 4, cv2.CV_32S)
        # component_stats.append((num_labels, label_im, stats, centroid))

        # Find the coordinates of the points in the component
        y, x = np.where(label_im == 1)
        points = np.column_stack((x, y))

        # Fit an ellipse to the component
        if points.shape[0] > 5:
            ellipse = cv2.fitEllipse(points)
            component_stats.append(
                (num_labels, label_im, stats, centroid, ellipse))
            print(f"Ellipse {count} fitted:" + str(ellipse))
            count += 1
        else:
            component_stats.append(
                (num_labels, label_im, stats, centroid, None))

    indents = []

    # Output the statistics of each component
    for component in component_stats:
        num_labels, label_im, stats, centroid, ellipse = component

        area = stats[1, cv2.CC_STAT_AREA]

        # Calculate the depth value at the centroid of the component
        y = int(centroid[1][0])
        x = int(centroid[1][1])
        # cent_depth = img[y, x][0] / 255.0  # Normalize depth value between 0-1

        # Calculate the average and largest depth values of the component
        mask = label_im == 1
        avg_depth = img[mask].mean() / 255.0
        max_depth = img[mask].max() / 255.0

        minor_axis = None
        major_axis = None

        if ellipse is not None:
            if ellipse[1][0] > ellipse[1][1]:
                major_axis = ellipse[1][0]
                minor_axis = ellipse[1][1]
            else:
                major_axis = ellipse[1][1]
                minor_axis = ellipse[1][0]

        indents.append({
            "area": area,
            "angle": np.deg2rad(ellipse[2]) if ellipse is not None else None,
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "centroid": {
                "y": y,
                "x": x
            },
            "avg_depth": avg_depth,
            "max_depth": max_depth
        })

        # Draw ellipses on the image (FOR TESTING PURPOSES -- TODO: REMOVE)
        # If an ellipse was fitted, draw it on the image
        if ellipse is not None:
            cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    # display the image with ellipses
    cv2.imshow("ellipses", img)
    cv2.waitKey(0)

    return indents
