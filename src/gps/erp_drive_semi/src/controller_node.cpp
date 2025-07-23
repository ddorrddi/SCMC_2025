#include "pure_pursuit.h"

void paramCallback(erp_drive_semi::CtrlParamConfig &config, PurePursuit* control) {
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

    dynamic_reconfigure::Server<erp_drive_semi::CtrlParamConfig> server;
    dynamic_reconfigure::Server<erp_drive_semi::CtrlParamConfig>::CallbackType f;
    
    f = boost::bind(&paramCallback, _1, &control);
    server.setCallback(f);

    std_msgs::Bool msg;

    // lanenet 주행 속도 7 -> 15
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

            ROS_INFO_STREAM("start_yaw : " << control.start_yaw << ", curr_yaw : " << control.curr_yaw << ", _diff : " << _diff);
        }

        // 유턴 path 용
        if(!control.is_finish && control.is_local_path_set && control.flag){
            ROS_INFO_STREAM("local PATH");
            control.lookahead_distance = control.local_lookahead_distance;
            control.velocity = control.local_velocity;
            control.calcSteering(control.curr_pose);
            control.publishControlMsg();
        }

        // global path 용
        else if(!control.is_finish && control.is_path_set && control.is_pose_set){
            ROS_INFO_STREAM("GLOBAL PATH");

            control.lookahead_distance = control.global_lookahead_distance;
            control.velocity = control.global_velocity;
            control.calcSteering(control.curr_pose);


            control.publishControlMsg();
        }

        else if(control.is_finish){
            ROS_INFO_STREAM("IS_FINISH");

            if(int(control.velocity) > 0){
                control.velocity = 0.;
                // control.brake = 1.;
                control.publishControlMsg();
            }

            break;
        }
        
        msg.data = control.is_finish;
        control.global_path_finish_pub.publish(std_msgs::Bool(msg));
        loop_rate.sleep();
    }

    return 0;
}
