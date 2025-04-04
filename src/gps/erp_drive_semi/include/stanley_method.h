#ifndef STANLEY_METHOD_H
#define STANLEY_METHOD_H

#include "controller.h"

class StanleyMethod: public Controller {
public:
    StanleyMethod(char control_target);

    double calc_error_front_axle(const geometry_msgs::Pose& curr_pose);

    void calcSteering(const geometry_msgs::Pose& curr_pose);

    void quat_to_euler(const geometry_msgs::Pose& pose, double& roll, double& pitch, double& yaw){
        tf::Quaternion q(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        );

        tf::Matrix3x3 m(q);
        m.getRPY(roll, pitch, yaw);
    }
};

#endif