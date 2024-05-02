import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import base64
import json

from create_dmap import create_dmap
from analyze_dmap import analyze_dmap


def main():
    """Main function to run the pipeline for hailpad analysis"""

    # Import mesh
    # TODO: Replace with user file input
    mesh = o3d.io.read_triangle_mesh("src/meshes/hailpad.stl")

    # Create depth map from mesh
    img = create_dmap(mesh)

    # Convert image to base64 for JSON output
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Get indent values (and convert numpy types to standard Python types)
    indents = analyze_dmap(img)
    indents = [{k: v.item() if isinstance(v, np.generic) else v for k,
                v in indent.items()} for indent in indents]

    output = {
        "indents": indents,
        "img": img_base64
    }

    with open('output.json', 'w') as f:
        json.dump(output, f)


if __name__ == '__main__':
    main()
