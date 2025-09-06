import copy

import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import numpy as np

def swift_scene():
    # 添加swift环境
    env = swift.Swift()
    # realtime=False的话，仿真是会根据计算频率在一瞬间完成的
    env.launch(realtime=True)


    cu3 = sg.Cuboid(scale=[14, 10, 0.002], base=sm.SE3(0, 0, 0.002), color=(0.5, 0.5, 0.5, 0.8))            # 地板
    cu4 = sg.Cuboid(scale=[14, 0.1, 2], base=sm.SE3(0, -5.05, 1), color=(0.647, 0.482, 0.227, 1.0))     # 左围墙
    cu5 = sg.Cuboid(scale=[14, 0.1, 2], base=sm.SE3(0, 5.05, 1), color=(0.647, 0.482, 0.227, 1.0))          # 右围墙    
    cu6 = sg.Cuboid(scale=[0.1, 10, 2], base=sm.SE3(-7.05, 0, 1), color=(0.647, 0.482, 0.227, 1.0))         # 上围墙
    cu7 = sg.Cuboid(scale=[0.1, 10, 2], base=sm.SE3(7.05, 0, 1), color=(0.647, 0.482, 0.227, 1.0))       # 下围墙

    shape_scene = [cu3, cu4, cu5, cu6, cu7]

    for shape in shape_scene:
        env.add(shape)

    # -------------------------------------------------------------------------------------------增加坐标轴
    # 增加坐标轴，可以通过 .T 在swift环境中显示
    ax_goal = sg.Axes(0.1)
    env.add(ax_goal)

    # 增加第二个坐标轴
    ax_goal2 = sg.Axes(0.1)
    env.add(ax_goal2)

    # --------------------------------------------------------------------------------------------添加机器人
    # .q即为该机械臂的关节角，qr可能是一个特殊的关节角数值，修改.q的值即可改变机械臂的状态。机械臂关节角的速度为qd
    frankie = rtb.models.Frankie()
    frankie.q = frankie.qr
    env.add(frankie)


    env.set_camera_pose([-5, 3, 0.7], [-2, 0.0, 0.5])  # 确定观测视角


    return env, frankie


# 测试
if __name__ == '__main__':
    _,_ = swift_scene()

    while(True):
        pass