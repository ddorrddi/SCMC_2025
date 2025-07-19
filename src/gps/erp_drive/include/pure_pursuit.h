#ifndef PURE_PURSUIT_H
#define PURE_PURSUIT_H

// #include <cmath>
// #define DEG2RAD(x) ((x) * M_PI / 180.0)

#include "controller.h"

// 나중에 mode용 클래스? 파일 추가하기
class PurePursuit: public Controller{
public:
    PurePursuit(char control_target);

    // path 추종이 끝났는지
    // bool is_finish;

    // // OBJ AVOID steering control flag
    // bool left_flag = false;
    // bool right_flag = false;
    // bool just_finished_local_path = false;    

    // // cb
    // void leftFlagCB(const std_msgs::Int16::ConstPtr& msg);
    // void rightFlagCB(const std_msgs::Int16::ConstPtr& msg);
    // void localPathDoneCB(const std_msgs::Int16::ConstPtr& msg);


    // lookahead distance | rosparam 으로 관리될 값
    double lookahead_distance;
    double global_lookahead_distance;
    double local_lookahead_distance;
    
    void calcSteering(const geometry_msgs::Pose& curr_pose);
    
    // 현재 차량의 위치에 대한 상대적인 좌표 구하기
    geometry_msgs::Pose calcRelativeCoordinateAboutCurr(const geometry_msgs::Pose& pt, const geometry_msgs::Pose& curr_pose);

    // lookahead distance에 있는 Waypoint의 idx 와 그걸 계산하는 함수
    int g_next_idx, l_next_idx, delivery_next_idx;
    int calcGlobalNextWaypoint(const geometry_msgs::Pose& curr_pose);
    int calcLocalNextWaypoint(const geometry_msgs::Pose& curr_pose);
    int calcDeliveryNextWaypoint(const geometry_msgs::Pose& curr_pose);
};


#endif