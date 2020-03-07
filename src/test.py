from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import numpy as np

model = load_model_from_path("../xmls/Baxter/baxter/master.xml")
sim = MjSim(model)

viewer = MjViewer(sim)


sim_state = sim.get_state()

while True:
    sim.set_state(sim_state)

    for i in range(2):

        sim.data.ctrl[:] = 0.03
        target_jacp = np.zeros(3 * sim.model.nv)
        # sim.forward()
        sim.data.get_site_jacp('grip', jacp=target_jacp)
        print(target_jacp[-7:-1])
        sim.data.ctrl[:] = np.ones(sim.model.nu) * 1e9
        sim.step()

        viewer.render()

    if os.getenv('TESTING') is not None:
        break
