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
#include <erp_drive/PathReference.h>
#include <erp_drive/PathReferenceElement.h>

// 제어 관련 메세지
#include <race/drive_values.h>
#include <morai_msgs/CtrlCmd.h>

// velocity 등 parameter 서버를 위한 헤더 
#include <dynamic_reconfigure/server.h>
#include <erp_drive/CtrlParamConfig.h>

#include <vector>
#include <string>

typedef enum {
    STOP_LINE = 9,
    STOP_DELIVERY_A = 11,
    STOP_TRAFFIC = 13,
    DECEL_BIG = 14
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
    int trafficSignNum;
    int index;
    EventType type;

    EventPlaceInfo(const int& _index, const EventType& _type){
        index = _index;
        type = _type;
    }
};

class EventPlaceInfoManager {
public:
    EventPlaceInfo& getCurrEventPlaceInfo() const {
        return *iter;
    }

    void init(int g_curr_idx){
        if(initialized) return;

        while(!this->isEventPlaceInfoEnd() && this->getCurrEventPlaceInfo().index < g_curr_idx)
            this->nextEventPlaceInfo();

        initialized = true;
    }

    bool isEventPlaceInfoEnd() {
        return iter->index == 99999999;
    }

    bool nextEventPlaceInfo() {
        if(this->isEventPlaceInfoEnd()) return false;
        iter++;
        return true;
    }

    double calcCurrPoseToEventPlaceDist(const geometry_msgs::Pose& curr_pose, const geometry_msgs::Pose& event_pose){
        double ret = pow(curr_pose.position.x - event_pose.position.x, 2);
        ret += pow(curr_pose.position.y - event_pose.position.y, 2);
        return sqrt(ret);
    }

private:
    bool initialized = false;
    const int END_NUM = 99999999;

    // @TODO 신호등 위치마다 번호 적기. 증가하는 순으로
    // 530, 590, 700, 925, 1490, 1915, 2040, 3470, 3655, 
    std::vector<EventPlaceInfo> event_place = {

        // EventPlaceInfo(END_NUM, EventType::STOP_LINE),

        // 배탈 A 체크
        // EventPlaceInfo(935549.5145976712, 1915871.4294302394, EventType::STOP_DELIVERY),
        EventPlaceInfo(220 , EventType::STOP_DELIVERY_A),

        // 그냥 정지선 1
        // EventPlaceInfo(935531.6553283043, 1915951.8808367082, EventType::STOP_LINE),
        // EventPlaceInfo(425, EventType::STOP_TRAFFIC),

        // 그냥 정지선 2
        // EventPlaceInfo(935539.5026992218, 1915963.0705893768, EventType::STOP_LINE),
        // EventPlaceInfo(605, EventType::STOP_TRAFFIC),

        // traffic 1
        // EventPlaceInfo(935562.5859266134, 1915954.1952024777, EventType::STOP_LINE),
        // EventPlaceInfo(1045, EventType::STOP_TRAFFIC),

        // traffic 2
        // EventPlaceInfo(935599.9336382326, 1915972.0595697816, EventType::STOP_LINE),
        EventPlaceInfo(1470, EventType::STOP_LINE),

        // Big Obstacle deceleration
        // EventPlaceInfo(1050/*tmp*/, EventType::DECEL_BIG),

        // delivery 1st stop sign 4 (배달 직전)
        // EventPlaceInfo(935650.1697879703, 1916096.433997373, EventType::STOP_LINE),
        EventPlaceInfo(1570, EventType::STOP_LINE),

        // 수정 시 delivery_planner.py 에서도 수정
        // 배달 시작 지점 >> 중복인데 넌 누구냐?
        // EventPlaceInfo(1348, EventType::STOP_DELIVERY_B),

        // 좌회전 traffic (1)
        // EventPlaceInfo(2650, EventType::STOP_TRAFFIC),

        // 좌회전 traffic (2)
        // EventPlaceInfo(1854, EventType::STOP_TRAFFIC),

        // // 그냥 정지선
        // EventPlaceInfo(2443, EventType::STOP_LINE),
        
        // // traffic
        // EventPlaceInfo(2914, EventType::STOP_TRAFFIC),

        // // 그냥 정지선
        // EventPlaceInfo(3072, EventType::STOP_LINE),

        // Last Element
        EventPlaceInfo(END_NUM, EventType::STOP_LINE),
    };

    std::vector<EventPlaceInfo>::iterator iter = event_place.begin();
};

class OptimalThrottle {
private:
    double max_velocity;

public:
    const double DISTANCE = 10.0;
    const double STOP_TIME = 3.0;
    const double DECEL_VELOCITY = 5.0;

    bool shouldStop, shouldDecel;

    EventPlaceInfoManager eventPlaceInfoManager;
    geometry_msgs::Pose eventPose;

// @TODO
// 지금은 brake PID고 뭐고 생각하기 어려우니까 일단 주행하다가 일정거리 이내에 들어오면 throttle을 0을 줌
// 그리고 GPS Velocity 를 기반으로 속도가 너무 감속했다 -> 그때부터는 throttle을 감속 속도를 부여함 (ex: 10 이면 10)
// 이때 brake 를 살짝 섞어서 진행해야 함

public:
    void setMaxVelocity(double _velocity){
        this->max_velocity = _velocity;
    }

    double calcLinearVelocity(geometry_msgs::Pose curr_pose){
        // if(!shouldStop) return this->max_velocity;
        
        // double dist_to_stop = this->eventPlaceInfoManager.calcCurrPoseToEventPlaceDist(curr_pose);
        // // ROS_INFO("calcTanhVelocity\ndist_to_stop : %lf", dist_to_stop);

        // return 15.0 / 5. * dist_to_stop;
        return this->max_velocity;
    }

    double calcTanhVelocity(geometry_msgs::Pose curr_pose){
        // if(!shouldStop && !shouldDecel) return MAX_VELOCITY;
        
        // // double dist_to_stop = this->eventPlaceInfoManager.calcCurrPoseToEventPlaceDist(curr_pose);
        // // // ROS_INFO("calcTanhVelocity\ndist_to_stop : %lf", dist_to_stop);
        
        // // if(shouldDecel && this->eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::DECEL_U_TURN_END){
        // //     return DECEL_VELOCITY;
        // // }

        // // return std::min(15.0 * tanh(dist_to_stop / 5.0), MAX_VELOCITY);
        if(shouldDecel) return this->max_velocity / 3;
        
        if(shouldStop) {
            return 0.;
        }

        return this->max_velocity;
    }
};

#endif