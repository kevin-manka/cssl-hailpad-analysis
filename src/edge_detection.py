import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('src/images/dmapog.png', cv2.IMREAD_GRAYSCALE)
# binary = cv2.imread('src/images/binary.png', cv2.IMREAD_GRAYSCALE)

# Apply blackhat transform
# filter_size = (50, 50)
# blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filter_size)
# blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, blackhat_kernel)

# accentuated = cv2.convertScaleAbs(blackhat_img, alpha=8, beta=4)
# cv2.imshow("accentuated", accentuated)
# cv2.waitKey(0)

edge_img = cv2.Canny(img, 15, 15)
cv2.imshow("canny", edge_img)
cv2.waitKey(0)

# Dilate the edge_img
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated_edge_img = cv2.dilate(edge_img, kernel, iterations=1)
cv2.imshow("dilated", dilated_edge_img)
cv2.waitKey(0)

# Find contours in the dilated edge image
contours, _ = cv2.findContours(dilated_edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

ellipses = []

# Check if there are enough contours found
if contours:
    # Loop through each contour
    for cnt in contours:
        # Ensure the contour has at least 5 points
        if len(cnt) >= 5:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(cnt)
            
            ellipses.append(ellipse)
else:
    print("No contours found")

# Display the countours found
output = img.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
cv2.imshow("contours", output)
cv2.waitKey(0)

# Draw ellipses from list of ellipses on image
output = img.copy()
for ellipse in ellipses:
    cv2.ellipse(output, ellipse, (0, 255, 0), 2)
cv2.imshow("output", output)
cv2.waitKey(0)

# # Fit edges to ellipses
# ellipses = cv2.fitEllipse(dilated_edge_img)

# # Draw ellipses on image
# output = img.copy()
# cv2.ellipse(output, ellipses, (0, 255, 0), 2)
# cv2.imshow("output", output)
# cv2.waitKey(0)

# # Perform bitwise and between binary and edge_img
# and_img = cv2.bitwise_and(binary, edge_img)
# cv2.imshow("and", and_img)
# cv2.waitKey(0)
