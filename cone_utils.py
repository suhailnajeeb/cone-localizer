import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpltPath

from PIL import Image
import numpy as np
import os
import random

def load_image(image_path):
    """
    Load an image from a specified file path.

    Args:
        image_path (str): The path to the image file to be loaded.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    try:
        image = Image.open(image_path)
        return image
    except IOError:
        print(f"Error opening image file {image_path}. Please check the file path and file integrity.")
        return None

def load_annotations(filename):
    """ Load annotations from a file. """
    with open(filename, 'r') as file:
        data = file.read().strip().split('\n')
    all_points = []
    for line in data:
        points = line.split(' ')[1:]  # skip the first number if it's a class label or similar
        points = np.array(points, dtype=np.float32).reshape(-1, 2)
        all_points.append(points)
    return all_points

def scale_points(points, img_width, img_height):
    """ Scale normalized points to image dimensions. """
    scaled_points = []
    for point_set in points:
        scaled_point_set = np.copy(point_set)
        scaled_point_set[:, 0] *= img_width
        scaled_point_set[:, 1] *= img_height
        scaled_points.append(scaled_point_set)
    return scaled_points

def get_tip_box(points, tip, box_scale=0.2):
    """
    Calculate the bounding box dimensions and position for the cone tip.

    Parameters:
        points (np.array): Array of points that define the cone's polygon.
        tip (tuple): The (x, y) coordinates of the cone tip.
        box_scale (float): Factor to scale the bounding box relative to the cone size.

    Returns:
        tuple: Returns (x, y, width, height) of the bounding box.
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width = max_x - min_x
    height = max_y - min_y

    tip_box_width = width * box_scale
    tip_box_height = height * box_scale

    tip_box_x = tip[0] - tip_box_width / 2
    tip_box_y = tip[1]

    return (tip_box_x, tip_box_y, tip_box_width, tip_box_height)

def plot_cone_tip(ax, points, tip, plot_box=False):
    """ Plot a cone tip as a cross and optionally a bounding box around it on a given matplotlib axes. """
    # Plot the tip as a cyan cross
    ax.scatter(tip[0], tip[1], color='brown', marker='x', s=100, linewidth = 5)  # s is the size of the marker

    if plot_box:
        # Assuming get_tip_box() is a function that computes the bounding box dimensions based on the points and the tip
        tip_box_x, tip_box_y, tip_box_width, tip_box_height = get_tip_box(points, tip)
        # Draw the bounding box with the tip at the bottom
        tip_box = patches.Rectangle((tip_box_x, tip_box_y), tip_box_width, tip_box_height, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(tip_box)

    return ax

def plot_annotations(image, annotations, cone_tips=None, plot_box=False):
    """ Plot image and overlay annotations. Optionally, plot cone tips as crosses and bounding boxes if provided and requested. """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw polygons for each annotation
    for points in annotations:
        poly = patches.Polygon(points, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(poly)

    # Plot cone tips and optionally draw bounding boxes around them if they are provided
    if cone_tips is not None:
        for points, tip in zip(annotations, cone_tips):
            # Plot the tip as a cyan cross
            ax.scatter(tip[0], tip[1], color='cyan', marker='x', s=50)  # s is the size of the marker

            if plot_box:
                tip_box_x, tip_box_y, tip_box_width, tip_box_height = get_tip_box(points, tip)
                # Draw the bounding box with the tip at the bottom
                tip_box = patches.Rectangle((tip_box_x, tip_box_y), tip_box_width, tip_box_height, linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(tip_box)

    ax.axis('off')
    plt.show()

def find_cone_tip(points):
    """ Find the tip of the cone by locating the point with the minimum y-value and taking the median of all such points in the input set, returning integer coordinates. """
    # Find the minimum y-value in the set of points
    min_y = np.min(points[:, 1])
    # Get all points that have this minimum y-value
    tip_candidates = points[points[:, 1] == min_y]
    # Choose the median x-value of these candidates as the tip x-coordinate
    tip_x = np.median(tip_candidates[:, 0])
    # Convert coordinates to integer
    tip_x_int = int(round(tip_x))
    min_y_int = int(round(min_y))
    # The tip is the median x-value and the minimum y-value, both as integers
    cone_tip = (tip_x_int, min_y_int)
    return cone_tip

def find_cone_tips(annotations):
    """ Find the tip of the cone by locating the point with the minimum y-value and taking the median of all such points, returning integer coordinates. """
    cone_tips = []
    for points in annotations:
        # Find the minimum y-value in the set of points
        min_y = np.min(points[:, 1])
        # Get all points that have this minimum y-value
        tip_candidates = points[points[:, 1] == min_y]
        # Choose the median x-value of these candidates as the tip x-coordinate
        tip_x = np.median(tip_candidates[:, 0])
        # Convert coordinates to integer
        tip_x_int = int(round(tip_x))
        min_y_int = int(round(min_y))
        # The tip is the median x-value and the minimum y-value, both as integers
        cone_tips.append((tip_x_int, min_y_int))
    return cone_tips

def create_mask_from_polygon(image_size, polygon):
    """
    TODO: Make this faster, maybe skip the mask generation
    Create a mask for the given polygon.

    Args:
    image_size: tuple of (width, height) for the image dimensions.
    polygon: numpy array of shape (n, 2), where n is the number of vertices.

    Returns:
    A numpy array representing the mask, with 1s inside the polygon and 0s outside.
    """
    width, height = image_size
    x, y = np.meshgrid(np.arange(width), np.arange(height))  # Create a grid of coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    path = mpltPath.Path(polygon)
    grid = path.contains_points(points)
    grid = grid.reshape((height, width))
    return grid.astype(int)

def plot_mask(image, mask):
    """Plot the image and overlay the mask."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='jet')  # Overlay mask with transparency
    plt.title('Image with Mask Overlay')
    plt.axis('off')
    plt.show()

def get_average_color(image, mask):
    image = np.array(image)
    masked_image = image * mask[:, :, None]
    nonzero_pixels = masked_image[masked_image.any(axis=2)]
    average_color = np.mean(nonzero_pixels, axis=0)
    return average_color

# Predefined ideal colors
ideal_colors = {
    'yellow': np.array([255, 255, 0]),
    'green': np.array([128, 255, 0]),
    'purple': np.array([128, 0, 128]),
    #'orange': np.array([255, 165, 0])
    'red': np.array([255, 0, 0]),
}

def calculate_color_distance(sampled_color, ideal_colors):
    distances = {color: np.linalg.norm(sampled_color - np.array(ideal_colors[color])) for color in ideal_colors}
    return min(distances, key=distances.get)

# Additional functions by GPT

import cv2

# def sample_points_within_polygon(polygon, num_samples=10):
#     """Sample random points within the given polygon."""
#     rect = cv2.boundingRect(polygon.astype(int))
#     min_x, min_y, width, height = rect
#     samples = []

#     while len(samples) < num_samples:
#         random_point = np.array([random.randint(min_x, min_x + width), random.randint(min_y, min_y + height)])
#         if cv2.pointPolygonTest(polygon, (random_point[0], random_point[1]), False) >= 0:
#             samples.append(random_point)
#     return np.array(samples)

def sample_points_within_polygon(polygon, num_samples=10):
    """Sample random points within the given polygon."""
    # Ensure the polygon is a NumPy array of type int
    if not isinstance(polygon, np.ndarray) or polygon.dtype != np.int32:
        polygon = np.array(polygon, dtype=np.int32)
    
    # Compute the bounding rectangle to limit the random point generation area
    rect = cv2.boundingRect(polygon)
    min_x, min_y, width, height = rect
    samples = []

    while len(samples) < num_samples:
        # Generate random point within the bounding rectangle
        random_point = (random.randint(min_x, min_x + width), random.randint(min_y, min_y + height))
        
        # Check if the point is inside the polygon
        if cv2.pointPolygonTest(polygon, random_point, False) >= 0:
            samples.append(random_point)
    return np.array(samples, dtype=np.int32)

def rgb_to_hue(rgb):
    """Convert an RGB numpy array to HSV and return the hue component."""
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)
    return hsv[0][0][0]

def calculate_average_hue(points, image):
    """Calculate the average hue from the list of points."""
    hues = [rgb_to_hue(image[pt[1], pt[0]]) for pt in points]
    return np.mean(hues)

def calculate_closest_hue(average_hue, ideal_hues):
    """ Calculate the closest hue by considering the cyclic nature of hues. """
    min_difference = float('inf')  # Start with an infinitely large difference
    closest_color = None

    for color, hue in ideal_hues.items():
        # Calculate absolute difference considering the circular nature of hues
        diff = min(abs(hue - average_hue), 360 - abs(hue - average_hue))

        # Update the closest color if a new smaller difference is found
        if diff < min_difference:
            min_difference = diff
            closest_color = color

    return closest_color

# Example usage in your script:
#closest_color = calculate_closest_hue(average_hue, ideal_hues)
