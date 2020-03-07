from .two_finger_gripper import TwoFingerGripper, LeftTwoFingerGripper



def gripper_factory(name):

    if name == 'TwoFingerGripper':
        return TwoFingerGripper

    if name == 'LeftTwoFingerGripper':
        return LeftTwoFingerGripper

    raise ValueError('Unknown gripper name')