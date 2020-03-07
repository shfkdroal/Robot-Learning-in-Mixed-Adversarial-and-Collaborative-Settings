import mujoco_py
import numpy as np
import inspect
import time
import matplotlib.pyplot as plt
import inspect

fullpath = "/home/tolga/neural_ws/baxter_RL/baxter/master.xml"
model = mujoco_py.load_model_from_path(fullpath)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
step = 0
# print(dir(sim._render_context_window))
# print(inspect.getargspec(mujoco_py.MjViewer))
# print(dir(viewer) == dir(sim._render_context_window))

# time.sleep(2)
# exit()

def useful_functions():
    funcs = ("sim.data.get_body_xpos",
             "sim.data.ctrl",
             "sim.model.joint_name2id",
             "sim.model.body_name2id")
    return funcs

    


while True:
    sim.step()
    if step == 3:
        print(dir(sim.data))
        # print(inspect.getsource(sim.data))
        # print(sim.data.get_body_xpos("kase"))
        print(sim.data.get_joint_qvel("left_e0"))
        print(len(sim.data.ctrl))
        print(sim.model.joint_name2id("left_e0"))
        print(len(sim.model.joint_names))
        #break
    step += 1
    act = np.random.normal(size=len(sim.data.ctrl))
    # sim.data.ctrl[0] += -0.0001
    # sim.data.ctrl[1] -= 0.0006
    # sim.data.ctrl[2] -= 0.0001
    # sim.data.ctrl[3] += 0.0005
    sim.data.ctrl[:] += act/100

    if step%3000 == 1:
        # rgb, depth = viewer.read_pixels(120, 120, True)
        # plt.imshow(np.flip(rgb, 0), cmap="magma_r")

        rgb, depth = sim.render(120, 120, camera_name="head_camera_rgb", depth=True)
        plt.imshow(np.flip(depth, 0), cmap="magma_r")        
        plt.show()
    
    
    # sim.data.ctrl[7] += -1
    # sim.data.ctrl[8] += 10    
    # sim.data.qfrc_applied[0] = -0.01
    viewer.render()
