#ifndef PURE_PURSUIT_H
#define PURE_PURSUIT_H

#include "controller.h"

class PurePursuit: public Controller{
public:
    PurePursuit(char control_target);

    double lookahead_distance;
    double global_lookahead_distance;
    double local_lookahead_distance;
    double parking_local_lookahead_distance;

    int g_next_idx;
    int l_next_idx;

    void calcSteering(const geometry_msgs::Pose& curr_pose);
    geometry_msgs::Pose calcRelativeCoordinateAboutCurr(const geometry_msgs::Pose& pt, const geometry_msgs::Pose& curr_pose);
    int calcGlobalNextWaypoint(const geometry_msgs::Pose& curr_pose);
    int calcLocalNextWaypoint(const geometry_msgs::Pose& curr_pose);
};

#endif