from enum import Enum, auto
from collections import namedtuple

class Lane(Enum):
    ONE = auto()
    TWO = auto()
    
class EventPlace(Enum):
    STOP_LINE = auto()
    
    U_TURN = auto()
    SPEED_BUMP = auto()

# 해당 Enum 클래스 수정 시 controller.h의 enum도 수정할 것
class EventType(Enum):
    STOP = 0
    DECELERATION = 1
    DRIVE = 2

    def __int__(self):
        return self.value

class TrafficLightStatus(Enum):
    RED_3 = auto()
    RED_LEFT_3 = auto()
    YELLOW_3 = auto()
    GREEN_3 = auto()
    RED_4 = auto()
    RED_LEFT_4 = auto()
    YELLO_4 = auto()
    RED_YELLO_4 = auto()
    GREEN_LEFT_4 = auto()
    GREEN_4 = auto()

class Constant:
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise Exception('변수에 값을 할당할 수 없습니다.')
        self.__dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            raise Exception('변수를 삭제할 수 없습니다.')
        
const = Constant()

# 이 값은 optimal_velocity 에서 가져다 쓰는 함수에 따라 다름

const.AVOIDANCE_SUCCESS_RAIDUS = 1.0   #1.75 default
# const.AVOIDANCE_SWITCH_RADIUS = 0.5
# const.OBSTACLE_RAIDUS = 2.0 >> obst on path에서 따로 관리
const.DELIVERY_SUCCESS_RADIUS = 1.5 
const.BLOCK_CHECK_LD = 8.0
const.LANE_WIDTH = 2.0 # 원래 : 3.5
const.LANE_WIDTH_SMALL = 0.75# 1.0 #FMTC : 0.75 MORAI : 1.3
const.EVENT_THRESHOLD_DIST = 10.0
