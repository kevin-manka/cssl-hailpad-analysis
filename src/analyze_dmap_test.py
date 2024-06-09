import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
from scipy.interpolate import LinearNDInterpolator


def analyze_dmap_test(img):
    """Apply CV algorithms to analyze the depth map and identify hail indents"""

    # Apply Gaussian blur (preprocessing to improve efficiency/accuracy)
    # blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold using blackhat transform
    filter_size = (50, 50)
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)

    blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, blackhat_kernel)

    # Increase contrast of the blackhat image to accentuate the hail indents
    # TODO: Make alpha value adjustable
    accentuated = cv2.convertScaleAbs(blackhat_img, alpha=6, beta=0)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("blackhat", blackhat_img)
    cv2.waitKey(0)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("accentuated", accentuated)
    cv2.waitKey(0)

    # Apply preliminary morphological operations to remove noise and hailpad edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    accentuated_refined = cv2.morphologyEx(
        accentuated, cv2.MORPH_OPEN, kernel, iterations=2)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("prelim noise removal", accentuated_refined)
    cv2.waitKey(0)
    
    # Separate out shallow artifacts by depth
    depth_threshold = 15
    filtered_blackhat = np.where(
        blackhat_img < depth_threshold, 0, accentuated_refined)
    filtered_blackhat_bg = np.where(
        blackhat_img > depth_threshold, 0, accentuated_refined)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("blackhat (filtered) -- main", filtered_blackhat)
    cv2.waitKey(0)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("blackhat (filtered) -- bg", filtered_blackhat_bg)
    cv2.waitKey(0)

    # Accentuate shallow artifacts
    filtered_accentuated = cv2.convertScaleAbs(filtered_blackhat_bg, alpha=4, beta=0)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("accentuated -- bg", filtered_accentuated)
    cv2.waitKey(0)

    _, filtered_accentuated_binary = cv2.threshold(filtered_accentuated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("accentuated -- bg -- binary", filtered_accentuated_binary)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(filtered_accentuated_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(filtered_accentuated.shape, dtype="uint8") * 255

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            cv2.drawContours(mask, [contour], -1, 0, -1)

    filtered_accentuated_refined = cv2.bitwise_and(filtered_accentuated, filtered_accentuated, mask=mask)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("accentuated -- bg -- refined", filtered_accentuated_refined)

    # Combine the accentuated hail indents with the filtered/accentuated/refined shallow artifacts
    accentuated_refined = cv2.add(filtered_accentuated_refined, filtered_blackhat)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("accentuated (combined)", accentuated_refined)
    cv2.waitKey(0)

    # Apply binary threshold
    ret, binary_img = cv2.threshold(
        accentuated_refined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("binary", binary_img)
    cv2.waitKey(0)

    # Apply morphological operations to remove noise and hailpad edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary_img = cv2.morphologyEx(
        binary_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # FOR TESTING PURPOSES (TODO: REMOVE)
    cv2.imshow("noise removal", binary_img)
    cv2.waitKey(0)

    # Watershed prep (TODO: Remove?)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))  # TODO: Remove

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

    # Perform connected components analysis
    # ret, markers = cv2.connectedComponents(binary_img)

    # -- OLD MESS --
    # # Filter out shallow artifacts by depth (e.g., small or non-hail indents)
    # # depth_threshold = 0
    # # filtered_blackhat = np.where(
    # #     blackhat_img < depth_threshold, 0, blackhat_img)
    # filtered_blackhat = accentuated # TODO: Remove

    # # FOR TESTING PURPOSES (TODO: REMOVE)
    # cv2.imshow("blackhat (filtered)", filtered_blackhat)
    # cv2.waitKey(0)

    # # Apply binary threshold
    # _, binary_img = cv2.threshold(
    #     accentuated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # TODO?: Change to filtered_blackhat

    # # Subtract difference between dilation and erosion transforms to remove noise and hailpad edges
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # binary_img = cv2.morphologyEx(binary_img,
    #                               cv2.MORPH_OPEN,
    #                               kernel,
    #                               iterations=1)

    # # FOR TESTING PURPOSES (TODO: REMOVE)
    # cv2.imshow("noise removal (dilation - erosion)", binary_img)
    # cv2.waitKey(0)

    # binary_img = cv2.erode(binary_img, kernel, iterations=2)
    # binary_img = cv2.dilate(binary_img, kernel, iterations=2)
    # cv2.imshow("fr", binary_img)
    # cv2.waitKey(0)

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

    # # Apply watershed algorithm
    # ret, markers = cv2.connectedComponents(binary_img) # TODO: Change to sure_bg
    # markers += 1
    # # markers[unknown == 255] = 0
    # # Convert back to 3-channel BGR image for watershed
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # markers = cv2.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]
    # -- OLD MESS --

    # FOR TESTING PURPOSES (TODO: REMOVE)
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(markers, cmap="tab20b")
    # ax.axis('off')
    # plt.show()

    # Apply watershed algorithm
    ret, markers = cv2.connectedComponents(binary_img)
    markers += 1
    # markers[unknown == 255] = 0
    # Convert back to 3-channel BGR image for watershed
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

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
        width = stats[1, cv2.CC_STAT_WIDTH]
        height = stats[1, cv2.CC_STAT_HEIGHT]

        # Calculate major and minor axes by determining the longest and shortest distances

        # TODO: Remove (use ellipse fitting algo to calculate major and minor axes)
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
        # cent_depth = img[y, x][0] / 255.0  # Normalize depth value between 0-1

        # Calculate the average and largest depth values of the component
        mask = label_im == 1
        avg_depth = img[mask].mean() / 255.0
        max_depth = img[mask].max() / 255.0

        indents.append({
            "area": area,
            "major_axis": ellipse[1][0] if ellipse is not None else None,
            "minor_axis": ellipse[1][1] if ellipse is not None else None,
            "centroid": {
                "y": y,
                "x": x
            },
            # "depth_at_centroid": cent_depth,
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
