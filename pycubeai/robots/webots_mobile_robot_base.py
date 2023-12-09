"""module webots_mobile_robot_base. Base class for
ground mobile robots from Webots simulator

"""
from abc import ABC

from pycubeai.robots import MobileRobotPose
from mobile_robot import MobileRobotBase
from robots_specification_enum_type import RobotSpecificationEnumType

class WebotsMobileRobotBase(MobileRobotBase, ABC):

    def __init__(self, name: str, pose: MobileRobotPose):
        super(WebotsMobileRobotBase, self).__init__(name=name,
                                                   specification=RobotSpecificationEnumType.WEBOTS_MOBILE_ROBOT_SPECIFICATION.name.upper(),
                                                    pose=pose)


