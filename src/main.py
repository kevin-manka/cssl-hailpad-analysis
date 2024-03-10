import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

from create_dmap import create_dmap
from analyze_dmap import analyze_dmap

def main():
    """Main function to run the pipeline for hailpad analysis"""

    # Import mesh
    mesh = o3d.io.read_triangle_mesh("src/meshes/hailpad.stl") # TODO: Replace with user file input

    # Create depth map from mesh
    img = create_dmap(mesh)

    # Output indent values
    indents = analyze_dmap(img,
                           min_area=140,
                           max_area=0,
                           min_minor=0,
                           max_minor=0,
                           min_major=0,
                           max_major=0)
    
    print(indents)

if __name__ == '__main__':
    main()
