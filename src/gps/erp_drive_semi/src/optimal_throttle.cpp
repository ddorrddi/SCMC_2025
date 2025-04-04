#include "optimal_throttle.h"

double OptimalThrottle::calcLinearVelocity(geometry_msgs::Pose curr_pose)
{
    if(!shouldStop) return MAX_VELOCITY;
    
    double dist_to_stop = this->eventPlaceInfoManager.calcCurrPoseToEventPlaceDist(curr_pose);
    // ROS_INFO("calcTanhVelocity\ndist_to_stop : %lf", dist_to_stop);

    return 15.0 / 5. * dist_to_stop;

    // GPS_VELOCITY 먼저 이걸로 진짜 속도를 측정함
    // PID 제어 (입력 - 속도 / 출력 - 브레이크)
}

double OptimalThrottle::calcTanhVelocity(geometry_msgs::Pose curr_pose)
{
    if(!shouldStop && !shouldDecel) return MAX_VELOCITY;
    
    double dist_to_stop = this->eventPlaceInfoManager.calcCurrPoseToEventPlaceDist(curr_pose);
    // ROS_INFO("calcTanhVelocity\ndist_to_stop : %lf", dist_to_stop);
    
    if(shouldDecel && this->eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::DECEL_U_TURN_END){
        return DECEL_VELOCITY;
    }

    return std::min(15.0 * tanh(dist_to_stop / 5.0), MAX_VELOCITY);
}
