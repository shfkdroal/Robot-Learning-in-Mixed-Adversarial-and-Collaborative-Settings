# Robot Learning in Mixed Settings IROS

<div>
<img src="new_cube.xml_predict_9_1Lifted.jpg" width="24%">
<img src="new_cube.xml_predict_9_2Perturbed.jpg" width="24%">
  <img src="new_cube.xml_predict_15_1Lifted.jpg" width="24%">
<img src="new_cube.xml_predict_15_2Perturbed.jpg" width="24%">
</div>
  
This project is based on [public grasp IROS](https://github.com/davidsonic/grasp_public) and maintained by Seung Hee Yoon, with the following new features added:

1. **Gneralized force classifier based on the top camera images**
2. **Robot learning framework with the classifier**
2. **Functionality to Capture the ground truth of robustness**



## Usage Examples

A number of examples available under examples folder:
- Ground map of robustness: One can collect ground truth (numby matrix) of robustness by setting paramerter 'training_R_table_ground_truth' in training/train_init_ik.py true.  
- [`inverse_kinmatics.py`](./examples/inverse_kinmatics.py): generates an qpos vector, which can be copied to sim.data.ctrl[:] to generate controlling
- robotics and objects: new objects can be added in xmls/Baxter/baxtermaster.xml
- discrete perturbation: 2D plane remapped to N uniformly discretized circular points
- camera and image recording: a new camera named "top_camera_rgb" defined in master.xml, change to camera id 1 to capture grasping images


## Results

- 3-state visualization images are saved in training/logs/images/bonues-test6


## Development

Install grasp environment and self-brewed-mujoco-py:

```
python setup.py build
pip install -e . 
```
Run training:

```
python training/train_init_ik.py
```


## Changelog

- 03/07/2019: Uploaded the public source code.

## Credits

`Grasping IROS` is maintained by the ICAROS team. Contributors include:

- Seug Hee Yoon
- Stefanos Nikolaidis
