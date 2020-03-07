
class Gripper(object):

    def __init__(self):
        super().__init__()



    @property
    def init_qpos(self):
        raise NotImplementedError


    @property
    def dof(self):
        raise NotImplementedError


    @property
    def joints(self):
        raise NotImplementedError



    def contact_geoms(self):
        return []


    @property
    def left_finger_geoms(self):
        raise NotImplementedError


    @property
    def right_finger_geoms(self):
        raise NotImplementedError


    