#include <stanley_method.h>

StanleyMethod::StanleyMethod(char control_target): Controller(control_target) {}


//조향각 theta = psi + artan( k * e / v) 
// psi : path_yaw - ego_yaw
// k : gain
// e: lateral error > 현재 추종하는 차선의 중심 위치와 차량의 앞바퀴 사이의 최단 거리 
// ** warning : 차선과 앞바퀴 좌표 A와 앞바퀴와 최단거리인 차선의 좌표 B사이의 직선은 
//              B좌표에서의 곡률 각도와 수직이다.
// v : velocity 
double StanleyMethod::calc_error_front_axle(const geometry_msgs::Pose& curr_pose){
    double roll, pitch, yaw;
    this->quat_to_euler(curr_pose, roll, pitch, yaw); 

    double fx = curr_pose.position.x + WHEEL_BASE * cos(yaw);
    double fy = curr_pose.position.y + WHEEL_BASE * sin(yaw);

    geometry_msgs::Pose stanley_pose;
    stanley_pose.position.x = fx;
    stanley_pose.position.y = fy;
    stanley_pose.position.z = 0;
    stanley_pose.orientation.x = curr_pose.orientation.x;
    stanley_pose.orientation.y = curr_pose.orientation.y;
    stanley_pose.orientation.z = curr_pose.orientation.w;
    stanley_pose.orientation.w = curr_pose.orientation.z;

    this->calcGlobalCurrWaypoint(stanley_pose);

    const int& gpx = this->g_path.path[this->g_curr_idx].pose.position.x;
    const int& gpy = this->g_path.path[this->g_curr_idx].pose.position.y;
    
    // return (fx - gpx) * (-cos(yaw + M_PI_2)) + (fy - gpy) * (-sin(yaw + M_PI_2));
    return (fx - gpx) * (fx - gpx) + (fy - gpy) * (fy - gpy); // 어차피 전역좌표 끼리 거리 차이(크기)만 구하면 되는거 아님?
}

void StanleyMethod::calcSteering(const geometry_msgs::Pose& curr_pose){
    double error_front_axle = calc_error_front_axle(curr_pose);

    double roll, pitch, curr_yaw, g_path_yaw;
    this->quat_to_euler(this->curr_pose, roll, pitch, curr_yaw);
    this->quat_to_euler(this->curr_pose, roll, pitch, g_path_yaw);

    double theta_e = g_path_yaw - curr_yaw;
    theta_e = fmodf64(theta_e + M_PI, M_PI * 2) - M_PI;

    double theta_d = atan2(0.5 * error_front_axle, 2.0);
    
    this->steering = theta_e + theta_d;

}
