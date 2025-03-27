#include "pure_pursuit.h"

void paramCallback(erp_drive::CtrlParamConfig &config, PurePursuit* control) {
    control->global_velocity = config.global_velocity;
    control->global_lookahead_distance = config.global_lookahead_distance;
    control->local_velocity = config.local_velocity;
    control->local_lookahead_distance = config.local_lookahead_distance;
    control->parking_velocity = config.parking_velocity;
    control->parking_lookahead_distance = config.parking_lookahead_distance;
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
        // ROS_INFO_STREAM("it's fine : " << control.is_path_set << ", " << control.is_pose_set << ", "<<control.is_finish<< ", " );
        // ROS_INFO_STREAM("velocity is " << control.parking_velocity << " ");
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

        // control.launch
        // 현재 imu 값 -180
        // -90 될 때까지 좌 조향
        // 잠깐 stop
        // 그리고 다시 -180 될 때까지 우 조향
        // stop

        // if(!control.is_parking_finish && control.is_parking_in_path_set && control.is_parking_start_point){
        //     dist_to_last_index = control.getDistance(control.parking_in_path.path.rbegin()->pose, control.curr_pose);
            
        //     if(control.parking_in_curr_idx == control.parking_in_path_size - 1){ 
        //         // 주차 완료 후 3초간 정지하는 로직 //
        //         // 인덱스는 다 끝까지 갔지만 거리 조건에 안맞는 경우 더 끝까지 후진하도록 
        //         if (!message_received){
        //             message_received_time = ros::Time::now();
        //             message_received = true;
        //         }
   
        //         ros::Time curr_time = ros::Time::now();
        //         ros::Duration elapsed_time = curr_time - message_received_time;
        //         if((elapsed_time.toNSec() / 1e9) <= 3.5){
        //             ROS_INFO_STREAM(" SUCCESSFULLY PARKED - STOP FOR 3.5 SECS");
        //             control.velocity = 0.0;
        //             control.brake = 1.0;
        //             control.publishControlMsg();    
        //         }
        //         //////////////////////////////

        //         // 여기서 부터 전진 //
        //         else {
        //             // if (!already_initialized){
        //             //     control.srv.request.request = true;
        //             //     ROS_INFO_STREAM("POSE INITIALIZED");
        //             //     control.client.call(control.srv);
        //             //     already_initialized = true;
        //             // }

        //             // geometry_msgs::PointStamped data1;
        //             // data1.header.frame_id = "map";
        //             // data1.header.stamp = ros::Time::now();
        //             // data1.point.x = control.parking_out_path.path[control.parking_out_curr_idx].pose.position.x;
        //             // data1.point.y = control.parking_out_path.path[control.parking_out_curr_idx].pose.position.y;
        //             // data1.point.z = control.parking_out_path.path[control.parking_out_curr_idx].pose.position.z;

        //             // data1.point.x -= 935531.2068552896;
        //             // data1.point.y -= 1915844.4607388575;

        //             // curr_waypoint_pub.publish(data1);


        //             // geometry_msgs::PointStamped data2;
        //             // data2.header.frame_id = "map";
        //             // data2.header.stamp = ros::Time::now();
        //             // data2.point.x = control.parking_out_path.path[control.out_parking_next_idx].pose.position.x;
        //             // data2.point.y = control.parking_out_path.path[control.out_parking_next_idx].pose.position.y;
        //             // data2.point.z = control.parking_out_path.path[control.out_parking_next_idx].pose.position.z;

        //             // data2.point.x -= 935531.2068552896;
        //             // data2.point.y -= 1915844.4607388575;

        //             // target_waypoint_pub.publish(data2);

        //             // ROS_INFO_STREAM(" BACK AGAIN - MOVING FORWARD");
        //             control.lookahead_distance = control.parking_lookahead_distance + 2.0;
        //             control.velocity = control.parking_velocity;
        //             control.brake = 0.0;
        //             control.is_moving_front = false;
        //             control.calcSteering(control.curr_pose);
        //             control.publishControlMsg();
        //         }
        //         ///////////////////
        //     }

        //     // 후진 path 주행
        //     else {
        //         // ROS_INFO_STREAM("MOVING BACKWARD");
        //         control.lookahead_distance = control.parking_lookahead_distance;
        //         control.velocity = control.parking_velocity;
        //         if (control.is_erp){control.velocity *= -1;}
        //         control.brake = 0.0;
        //         control.is_moving_front = true;
        //         control.calcSteering(control.curr_pose);
        //         control.publishControlMsg();
        //     }
        // }

        // else if(control.is_local_path_set){
        //     control.lookahead_distance = control.local_lookahead_distance;
        //     control.velocity = control.local_velocity;
        //     control.calcSteering(control.curr_pose);
        //     control.publishControlMsg();
        // }

        // else if(!control.is_finish && control.is_path_set && control.is_pose_set){
        //     control.lookahead_distance = control.global_lookahead_distance;
        //     control.velocity = control.global_velocity;
        //     // if (!parking_flag){
        //     //     parking_flag = control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_PARKING;
        //     // }
            
        //     // if(control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_PARKING){
        //     //     double tmp1 = std::min(1 - (control.velocity / 15), 0.4);
        //     //     double tmp2 = 1 - std::min(3 * tanh(control.optimal_throttle.eventPlaceInfoManager.calcCurrPoseToEventPlaceDist(control.curr_pose) / 35), 1.);
                
        //     //     if(control.optimal_throttle.eventPlaceInfoManager.calcCurrPoseToEventPlaceDist(control.curr_pose) < 0.5){
        //     //         if(prev_brake == 0.0) prev_brake = 0.4;
        //     //         prev_brake = control.brake;
        //     //         control.brake = prev_brake + 0.06;
        //     //     }

        //     //     else {
        //     //         control.brake = tmp1;
        //     //     }
        //     // }

        //     // ROS_INFO_STREAM("Controller_Node\nvelocity : " << control.velocity << ", brake : " << control.brake);

        //     control.calcSteering(control.curr_pose);
        //     control.publishControlMsg();
        // }

        // else if(control.is_finish){
        //     control.velocity = 0.;
        //     control.brake = 1.;

        //     control.publishControlMsg();
        // }

        loop_rate.sleep();
    }

    return 0;
}
