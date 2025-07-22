#include "pure_pursuit.h"

void paramCallback(erp_drive::CtrlParamConfig &config, PurePursuit* control) {
    control->global_velocity = config.global_velocity;
    control->global_lookahead_distance = config.global_lookahead_distance;
    control->local_velocity = config.local_velocity;
    control->local_lookahead_distance = config.local_lookahead_distance;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pure_pursuit");

    ROS_INFO("controller_node init");

    PurePursuit control(argv[1][0]);

    ros::Rate loop_rate(control.LOOP_RATE);

    ros::Publisher target_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/target_waypoint", 1);
    ros::Publisher curr_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/curr_waypoint", 1);

    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig> server;
    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig>::CallbackType f;
    
    f = boost::bind(&paramCallback, _1, &control);
    server.setCallback(f);
    
    ros::Time message_received_time;
    bool message_received = false;
    bool already_initialized = false;

    double prev_brake = 0.0;
    double dist_to_last_index = 0.0;
    control.brake = 0;

    bool test_flag = false;

    while(ros::ok()){

        ros::spinOnce(); // 콜백 함수 호출

        if(control.is_path_set && control.is_pose_set && control.start_yaw != -400){
            int diff = control.curr_yaw - control.start_yaw;

            if(diff >= 180)
                diff = abs(diff - 360);

            else if(diff <= -180)
                diff = abs(diff + 360);

            ROS_INFO_STREAM("diff : " << diff << ", start_yaw : " << control.start_yaw << ", curr_yaw : " << control.curr_yaw);

            if(!test_flag){
                if(diff >= 60){
                    if(!message_received){
                        message_received_time = ros::Time::now();
                        message_received = true;
                    }

                    while(message_received && (ros::Time::now() - message_received_time).toSec() <= 3){
                        control.steering = 0.0;
                        control.velocity = 0.0;
                        control.brake = 1.0;
                        control.publishControlMsg();
                    }
                    
                    test_flag = true;
                    control.brake = 0.0;

                } else {
                    control.steering = 30.0;
                    control.velocity = 5.0;
                    control.publishControlMsg();
                }
            } else {
                if(diff <= 5){
                    // test_flag = true;
                    control.steering = 0.0;
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();
                } else {
                    control.steering = -30.0;
                    control.velocity = 5.0;
                    control.publishControlMsg();
                }
            }
        }


        if(control.is_local_path_set){
            control.lookahead_distance = control.local_lookahead_distance;
            control.velocity = control.local_velocity;
            control.calcSteering(control.curr_pose);
            control.publishControlMsg();
        }

        else if(!control.is_finish && control.is_path_set && control.is_pose_set){
            control.lookahead_distance = control.global_lookahead_distance;
            control.velocity = control.global_velocity;

            control.calcSteering(control.curr_pose);
            control.publishControlMsg();
        }

        else if(control.is_finish){
            control.velocity = 0.;
            control.brake = 1.;

            control.publishControlMsg();
        }

        loop_rate.sleep();
    }

    return 0;
}
