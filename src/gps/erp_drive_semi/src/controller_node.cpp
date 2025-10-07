#include "pure_pursuit.h"

void paramCallback(erp_drive_semi::CtrlParamSemiConfig &config, PurePursuit* control) {
    control->global_velocity = config.global_velocity;
    control->global_lookahead_distance = config.global_lookahead_distance;
    control->local_velocity = config.local_velocity;
    control->local_lookahead_distance = config.local_lookahead_distance;
    control->parking_local_lookahead_distance = config.parking_local_lookahead_distance;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pure_pursuit");

    ROS_INFO("controller_node init");

    PurePursuit control(argv[1][0]);

    ros::Rate loop_rate(control.LOOP_RATE);

    ros::Publisher target_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/target_waypoint", 1);
    ros::Publisher curr_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/curr_waypoint", 1);

    dynamic_reconfigure::Server<erp_drive_semi::CtrlParamSemiConfig> server;
    dynamic_reconfigure::Server<erp_drive_semi::CtrlParamSemiConfig>::CallbackType f;
    
    f = boost::bind(&paramCallback, _1, &control);
    server.setCallback(f);

    std_msgs::Bool msg;

    while(ros::ok()){
        ros::spinOnce();

        if(control.flag && control.start_yaw != -400){
            int _diff = control.curr_yaw - control.start_yaw;
            
            if(_diff >= 180)
                _diff = abs(_diff - 360);
            
            else if(_diff <= -180)
                _diff = abs(_diff + 360);
        
            if(abs(_diff) >= 70) {
                control.flag = false;
                continue;
            }

        }

        switch(control.current_state) {
            case VehicleState::GLOBAL_PATH_DRIVING:
                if(!control.is_finish && control.is_path_set && control.is_pose_set){
                    ROS_INFO_STREAM("State: GLOBAL_PATH_DRIVING");
                    control.lookahead_distance = control.global_lookahead_distance;

                    if (110 < control.g_curr_idx && control.g_curr_idx < 230 && !control.is_parking_complete) {
                        control.brake = 0.0;
                        control.velocity = 10.0;
                    } else if (1100 < control.g_curr_idx && !control.uturn_done) {
                        control.brake = 0.0;
                        control.velocity = 10.0;
                    } else {
                        control.brake = 0.0;
                        control.velocity = control.global_velocity;
                    }
                    
                    control.calcSteering(control.curr_pose);
                    control.publishControlMsg();
                }
                break;

            case VehicleState::LOCAL_PATH_DRIVING:
                if(!control.is_finish && control.is_local_path_set){
                    ROS_INFO_STREAM("State: LOCAL_PATH_DRIVING");
                    control.lookahead_distance = control.local_lookahead_distance;
                    control.velocity = control.local_velocity;
                    control.calcSteering(control.curr_pose);
                    control.publishControlMsg();
                }
                break;
            
            case VehicleState::PARKING_PATH_DRIVING:
                if(!control.is_finish && control.is_local_path_set){
                    ROS_INFO_STREAM("State: PARKING_PATH_DRIVING");
                    control.lookahead_distance = control.parking_local_lookahead_distance;
                    control.velocity = control.local_velocity;
                    control.calcSteering(control.curr_pose);
                    control.publishControlMsg();
                }
                break;

            case VehicleState::PARKING_DONE_STOPPING:
                ROS_INFO_STREAM("State: PARKING_DONE_STOPPING (3 seconds)");
                control.velocity = 0.0;
                control.brake = 1.0;
                control.publishControlMsg();
                break;

            case VehicleState::PARKING_REVERSING:
                if(control.is_local_path_set){
                    ROS_INFO_STREAM("State: PARKING_REVERSING");
                    control.lookahead_distance = control.parking_local_lookahead_distance;
                    control.brake = 0.0;
                    control.velocity = 5.0;
                    control.calcSteering(control.curr_pose);
                    control.is_moving_back = true;
                    control.publishControlMsg();
                }
                break;

            case VehicleState::REVERSE_DONE_STOPPING:
                ROS_INFO_STREAM("State: REVERSE_DONE_STOPPING (1 second)");
                control.velocity = 0.0;
                control.brake = 1.0;
                control.is_moving_back = false;
                control.is_parking_complete = true;
                control.publishControlMsg();
                break;
        }

        if(control.is_finish){
            ROS_INFO_STREAM("IS_FINISH");

            if(int(control.velocity) > 0){
                control.velocity = 0.;
                // control.brake = 1.;
                control.publishControlMsg();
            }

            msg.data = control.is_finish;
            control.global_path_finish_pub.publish(std_msgs::Bool(msg));
            break;
        }
        
        msg.data = control.is_finish;
        control.global_path_finish_pub.publish(std_msgs::Bool(msg));
        loop_rate.sleep();
    }

    return 0;
}
