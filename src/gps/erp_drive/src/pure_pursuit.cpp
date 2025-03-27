#include "pure_pursuit.h"

PurePursuit::PurePursuit(char control_target):
    Controller(control_target), g_next_idx(0), l_next_idx(0), delivery_next_idx(0),
    lookahead_distance(0.), global_lookahead_distance(0.), local_lookahead_distance(0.) {}

geometry_msgs::Pose PurePursuit::calcRelativeCoordinateAboutCurr(const geometry_msgs::Pose& pt, const geometry_msgs::Pose& curr_pose){
    tf::Transform inverse;
    tf::poseMsgToTF(curr_pose, inverse);
    tf::Transform transform = inverse.inverse();

    tf::Pose p;
    poseMsgToTF(pt, p);

    tf::Pose tf_p = transform * p;
    geometry_msgs::Pose ret;
    poseTFToMsg(tf_p, ret);

    return ret;
}

void PurePursuit::calcSteering(const geometry_msgs::Pose& curr_pose) {
    geometry_msgs::Pose curr_waypoint;
    geometry_msgs::Pose next_waypoint;

    calcGlobalCurrWaypoint(curr_pose);
    calcGlobalNextWaypoint(curr_pose);

    if(this->is_local_path_set){
        calcLocalCurrWaypoint(curr_pose);
        calcLocalNextWaypoint(curr_pose);

        curr_waypoint = this->l_path.path.at(this->l_curr_idx).pose;
        next_waypoint = this->l_path.path.at(this->l_next_idx).pose;    
    }

    else if(this->is_delivery_path_set){
        ROS_INFO("is_delivery_path_set");

        calcDeliveryCurrWaypoint(curr_pose);
        calcDeliveryNextWaypoint(curr_pose);

        curr_waypoint = this->delivery_path.path.at(this->delivery_curr_idx).pose;
        next_waypoint = this->delivery_path.path.at(this->delivery_next_idx).pose;    
    }
    
    else {
        curr_waypoint = this->g_path.path.at(this->g_curr_idx).pose;
        next_waypoint = this->g_path.path.at(this->g_next_idx).pose;
    }

    // 감속해야 하는데, U_TURN 임.
    // if(this->optimal_throttle.shouldDecel 
    //     && this->optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::DECEL_U_TURN_END) {
    //     // 바로 왼쪽 풀 조향
    //     this->steering = 30.0;
    // }

    // else {
    double denominator = pow(getDistance(curr_pose, next_waypoint), 2);
    double numerator = 2 * this->calcRelativeCoordinateAboutCurr(next_waypoint, curr_pose).position.y;

    if(denominator == 0)
        this->steering = 0.;

    else
        this->steering = atan(this->WHEEL_BASE * numerator / denominator);
    // }
}

int PurePursuit::calcGlobalNextWaypoint(const geometry_msgs::Pose& curr_pose) {
    if(this->g_path_size == 0){
        ROS_ERROR("calcNextWaypoint(): no path!");
        return this->g_next_idx = -1;
    }

    this->calcGlobalCurrWaypoint(curr_pose);
    for(int i = this->g_curr_idx; i < this->g_path_size - 1; i++){
        if(i == this->g_path_size - 2){
            // this->is_finish = true;
            // ROS_INFO("calcNextWaypoint(): Finish Path Tracking");
            return this->g_next_idx = i + 1;
        }

        if(getDistance(this->g_path.path[i].pose, curr_pose) >= this->lookahead_distance)
            return this->g_next_idx = i;
    }

    return this->g_next_idx;
}

int PurePursuit::calcLocalNextWaypoint(const geometry_msgs::Pose& curr_pose) {
    if(this->l_path_size == 0){
        ROS_ERROR("calcNextWaypoint(): no path!");
        return this->l_next_idx = -1;
    }
    
    this->calcLocalCurrWaypoint(curr_pose);
    for(int i = this->l_curr_idx; i < this->l_path_size - 1; i++){
        if(i == this->l_path_size - 2)
            return this->l_next_idx = i + 1;

        if(getDistance(this->l_path.path[i].pose, curr_pose) >= this->lookahead_distance)
            return this->l_next_idx = i;
    }

    return this->l_next_idx;
}

int PurePursuit::calcDeliveryNextWaypoint(const geometry_msgs::Pose& curr_pose) {
    if(this->delivery_path_size == 0){
        ROS_ERROR("calcNextWaypoint(): no path!");
        return this->l_next_idx = -1;
    }
    
    this->calcDeliveryCurrWaypoint(curr_pose);
    for(int i = this->delivery_curr_idx; i < this->delivery_path_size - 1; i++){
        if(i == this->delivery_path_size - 2)
            return this->delivery_next_idx = i + 1;

        if(getDistance(this->delivery_path.path[i].pose, curr_pose) >= this->lookahead_distance)
            return this->delivery_next_idx = i;
    }

    return this->delivery_next_idx;
}

