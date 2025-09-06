import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import numpy as np


def swift_scene():
    """
    Initializes and configures a Swift simulation environment with a robot and surrounding objects.

    Returns:
        tuple: A tuple containing:
            - env (swift.Swift): The Swift simulation environment.
            - frankie (rtb.models.Frankie): The robot model instance.
    """
    # Initialize the Swift environment.
    # If realtime=False, the simulation runs as fast as possible, otherwise it runs in real time.
    env = swift.Swift()
    env.launch(realtime=True)

    # Define and add static environment objects (floor and walls).
    floor = sg.Cuboid(scale=[14, 10, 0.002], base=sm.SE3(0, 0, 0.002), color=(0.5, 0.5, 0.5, 0.8))
    left_wall = sg.Cuboid(scale=[14, 0.1, 2], base=sm.SE3(0, -5.05, 1), color=(0.647, 0.482, 0.227, 1.0))
    right_wall = sg.Cuboid(scale=[14, 0.1, 2], base=sm.SE3(0, 5.05, 1), color=(0.647, 0.482, 0.227, 1.0))
    top_wall = sg.Cuboid(scale=[0.1, 10, 2], base=sm.SE3(-7.05, 0, 1), color=(0.647, 0.482, 0.227, 1.0))
    bottom_wall = sg.Cuboid(scale=[0.1, 10, 2], base=sm.SE3(7.05, 0, 1), color=(0.647, 0.482, 0.227, 1.0))

    shape_scene = [floor, left_wall, right_wall, top_wall, bottom_wall]

    for shape in shape_scene:
        env.add(shape)

    # Add coordinate axes for visualization.
    ax_goal = sg.Axes(0.1)
    env.add(ax_goal)

    # Add a second set of coordinate axes.
    ax_goal2 = sg.Axes(0.1)
    env.add(ax_goal2)

    # Add the robot to the scene.
    # .q represents the joint angles of the robot arm.
    # .qr is a nominal joint configuration (e.g., ready pose).
    # .qd represents the joint velocities.
    frankie = rtb.models.Frankie()
    frankie.q = frankie.qr
    env.add(frankie)

    # Set the camera pose for a specific viewpoint.
    env.set_camera_pose([-5, 3, 0.7], [-2, 0.0, 0.5])

    return env, frankie


# Main execution block for testing.
if __name__ == '__main__':
    _, _ = swift_scene()

    # Keep the simulation window open.
    while True:
        pass