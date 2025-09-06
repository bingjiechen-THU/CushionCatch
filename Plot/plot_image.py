import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plot3d(data, file_path, point=None):
    """
    Plots a 3D curve and saves it to a file.

    Args:
        data (list or tuple): A container with three elements (e.g., list of lists,
                              tuple of arrays) representing the x, y, and z coordinates
                              of the curve.
        file_path (str): The path where the plot image will be saved.
        point (tuple, optional): A tuple (x, y, z) for a specific point to be
                                 highlighted on the plot. Defaults to None.
    """
    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')  # Set the figure background to white
    ax.set_facecolor('white')  # Set the axes background to white

    # Plot the 3D curve
    ax.plot3D(data[0], data[1], data[2], linewidth=1.5, color="Purple")
    if point is not None:
        x, y, z = point
        ax.scatter(x, y, z, color="Red", s=60, label='Catch Point')

    # Add title and labels with a font size of 16
    ax.set_title('3D Curve', fontsize=16)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.set_zlabel('Z', fontsize=16)

    # Set the font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, pad=0)

    # Set the color of the axis panes to white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Set the axis grid style and color
    grid_color = (0.8, 0.8, 0.8, 1.0)
    ax.xaxis._axinfo['grid'].update(linestyle="dashed", color=grid_color)
    ax.yaxis._axinfo['grid'].update(linestyle="dashed", color=grid_color)
    ax.zaxis._axinfo['grid'].update(linestyle="dashed", color=grid_color)

    # Save the figure, removing extra white space
    plt.savefig(file_path, bbox_inches='tight')

    # Display the plot
    plt.show()