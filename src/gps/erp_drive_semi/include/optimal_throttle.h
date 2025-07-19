#ifndef __OPTIMAL_THROTTLE_H__
#define __OPTIMAL_THROTTLE_H__

// ros 관련 헤더
#include <ros/ros.h>

// 좌표계 및 기하 연산 관련 헤더
#include <cmath>
#include <tf/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <std_msgs/Int16.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

// Path 관련 커스텀 메세지
#include <erp_drive_semi/PathReference.h>
#include <erp_drive_semi/PathReferenceElement.h>

// 제어 관련 메세지
#include <race/drive_values.h>
#include <morai_msgs/CtrlCmd.h>

// velocity 등 parameter 서버를 위한 헤더 
#include <dynamic_reconfigure/server.h>
#include <erp_drive_semi/CtrlParamConfig.h>

#include <vector>
#include <string>

typedef enum {
    DRIVE = 1,
    
    STOP = 4,
    STOP_LINE = 5,
    STOP_PARKING = 6,
    STOP_DELIVERY = 7,
    
    DECEL = 8,
    DECEL_U_TURN_START = 9,
    DECEL_U_TURN_END = 10,
    DECEL_SPEED_BUMP_START = 11,
    DECEL_SPEED_BUMP_END = 12
} EventType;

namespace TrafficLightStatus {
    const std::string RED3 = "3red";
    const std::string YELLOW3 = "3yellow";
    const std::string GREEN3 = "3green";
    const std::string RED4 = "4red";
    const std::string RED_LEFT4 = "4redleft";
    const std::string YELLO4 = "4yellow";
    const std::string RED_YELLO4 = "4redyellow";
    const std::string GREEN_LEFT4 = "4greenleft";
    const std::string GREEN4 = "4green";
    const std::string NONE = "";
};

struct EventPlaceInfo {
public:
    double x, y;
    EventType type;

    EventPlaceInfo(const double& _x, const double& _y, const EventType& _type) {
        x = _x;
        y = _y;
        type = _type;
    }
};

class EventPlaceInfoManager {
public:
    EventPlaceInfo& getCurrEventPlaceInfo() {
        return *iter;
    }

    bool isEventPlaceInfoEnd() {
        return iter == event_place.end();
    }

    bool nextEventPlaceInfo() {
        if(this->isEventPlaceInfoEnd()) return false;
        iter++;
        return true;
    }

    double calcCurrPoseToEventPlaceDist(const geometry_msgs::Pose& curr_pose){
        double ret = pow(curr_pose.position.x - getCurrEventPlaceInfo().x, 2);
        ret += pow(curr_pose.position.y - getCurrEventPlaceInfo().y, 2);
        return sqrt(ret);
    }

private:
    // 530, 590, 700, 925, 1490, 1915, 2040, 3470, 3655, 
    std::vector<EventPlaceInfo> event_place = {
        // 배탈 A 체크
        EventPlaceInfo(935549.5145976712, 1915871.4294302394, EventType::STOP_DELIVERY),

        // traffic 1
        EventPlaceInfo(935553.8989157581, 1915885.608052522, EventType::STOP_LINE),

        // 그냥 정지선 1
        EventPlaceInfo(935531.6553283043, 1915951.8808367082, EventType::STOP_LINE),

        // 그냥 정지선 2
        EventPlaceInfo(935539.5026992218, 1915963.0705893768, EventType::STOP_LINE),

        // traffic 2
        EventPlaceInfo(935562.5859266134, 1915954.1952024777, EventType::STOP_LINE),

        // traffic 3
        EventPlaceInfo(935599.9336382326, 1915972.0595697816, EventType::STOP_LINE),

        // traffic 3
        EventPlaceInfo(935599.9336382326, 1915972.0595697816, EventType::STOP_LINE),

        // traffic 4 (배달 직전)
        EventPlaceInfo(935650.1697879703, 1916096.433997373, EventType::STOP_LINE),

        // 배달 시작 지점
        // EventPlaceInfo(935654.2636029524, 1916124.8271699022, EventType::STOP_LINE),

        // 좌회전 traffic (1)
        EventPlaceInfo(935649.197701739, 1916202.2291346425, EventType::STOP_LINE),

        // 좌회전 traffic (2)
        EventPlaceInfo(935603.9153716592, 1916237.9461188167, EventType::STOP_LINE),

        // 그냥 정지선
        EventPlaceInfo(935626.7522067935, 1916113.72147557, EventType::STOP_LINE),
        
        // traffic
        EventPlaceInfo(935610.8443009052, 1916007.955912431, EventType::STOP_LINE),

        // 그냥 정지선
        EventPlaceInfo(935591.0916574165, 1915966.4912939603, EventType::STOP_LINE),

        // 주차 시작점
        // EventPlaceInfo(935545.2449255498, 1915882.5901543524, EventType::STOP_LINE),

        // Last Element
        EventPlaceInfo(0., 0., EventType::STOP_LINE),
    };

    std::vector<EventPlaceInfo>::iterator iter = event_place.begin();
};

class OptimalThrottle
{
public:
    const double DISTANCE = 15.0;
    const double STOP_TIME = 3.0;
    const double MAX_VELOCITY = 15.0;
    const double DECEL_VELOCITY = 5.0;

    // ns를 s단위로 바꾸기 위함
    const double NANO_TO_SEC = 1000000000.0;

    bool shouldStop, shouldDecel;
    double throttle;

    EventPlaceInfoManager eventPlaceInfoManager;
    geometry_msgs::Pose eventPose;

public:
    double calcTanhVelocity(geometry_msgs::Pose curr_pose);
    double calcLinearVelocity(geometry_msgs::Pose curr_pose);
};

#endif