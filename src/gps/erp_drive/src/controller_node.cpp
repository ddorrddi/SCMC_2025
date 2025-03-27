#include "pure_pursuit.h"

void paramCallback(erp_drive::CtrlParamConfig &config, PurePursuit* control) {
    control->optimal_throttle.setMaxVelocity(config.global_velocity);
    control->global_lookahead_distance = config.global_lookahead_distance;

    control->local_velocity = config.local_velocity;
    control->local_lookahead_distance = config.local_lookahead_distance;
}

void clearConsole();
void logging(const PurePursuit& control);

int main(int argc, char** argv) {
    ros::init(argc, argv, "pure_pursuit");

    ROS_INFO("controller_node init");

    PurePursuit control(argv[1][0]);
    int parking_space_num = atoi(argv[3]);
    control.parking_space_num = parking_space_num;
    bool isLogging = true; //argv[2][0] == 't';

    ros::Rate loop_rate(control.LOOP_RATE);

    ros::Publisher target_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/target_waypoint", 1);
    ros::Publisher curr_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/curr_waypoint", 1);

    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig> server;
    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig>::CallbackType f;
    
    f = boost::bind(&paramCallback, _1, &control);
    server.setCallback(f);
    
    ros::Time message_received_time;

    // 주차 공간 각도 349
    bool flag = false;
    double parking_mid_angle = 50;
    
    int parking_phase = -1; // ~ 4? 각 기능은 ROS_INFO 참고

    double prev_x = -1.0, prev_y = -1.0;

    int delivery_phase = -1;  // parking_phase와 유사하게 초기화

    while(ros::ok()){
        ros::spinOnce();

        if(!control.is_pose_set) continue;
    
        if(prev_x == -1 && prev_y == -1){
            prev_x = control.curr_pose.position.x;
            prev_y = control.curr_pose.position.y;
        }

        double dist = sqrt(pow(prev_x - control.curr_pose.position.x, 2) + pow(prev_y - control.curr_pose.position.y, 2));
        if(dist > 0.5){
            ROS_INFO("Pose Initialized : Found Index");
            break;
        }

        control.steering = 0.0;
        control.lookahead_distance = control.global_lookahead_distance;
        control.velocity = control.optimal_throttle.calcTanhVelocity(control.curr_pose);
        control.publishControlMsg();
    }

    control.calcGlobalCurrWaypoint(control.curr_pose);
    control.optimal_throttle.eventPlaceInfoManager.init(control.g_curr_idx);

    while(ros::ok()){
        ros::spinOnce();
        control.flowControl(control.curr_pose);

        if(isLogging && control.is_path_set && control.is_pose_set){
            logging(control);
        }

        if(control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_PARKING && control.optimal_throttle.shouldStop){
            int diff = control.curr_yaw - control.start_yaw;
            
            if(diff >= 180)
                diff = abs(diff - 360);
            
            else if(diff <= -180)
                diff = abs(diff + 360);
            
            // 주차 공간 인식 후 이 위치에 멈출 수 있도록 함.
            if(parking_phase == -1){
                control.steering = 0.0;
                control.velocity = 0.0;
                control.brake = 1.0;
                control.publishControlMsg();

                control.timer.start();
                if(!control.timer.isTimeout()){
                    ROS_INFO("[STOP PARKING] Vehicle Stabilization");
                    continue;
                }

                control.start_yaw = -400;
                control.timer.resetTimer(0.5);
                parking_phase += 1;
            }

            // [STOP PARKING] Back Moving (1)
            else if(parking_phase == 0){
                if(control.start_yaw == -400) continue;

                ROS_INFO("[STOP PARKING] Back Moving (1)");

                if(diff >= parking_mid_angle){
                    control.steering = 0.0;
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();

                    control.timer.start();
                    if(!control.timer.isTimeout()){
                        // ROS_INFO("[STOP PARKING] Vehicle Stabilization");
                        continue;
                    }

                    control.brake = 0.0;

                    control.timer.resetTimer(3.0);
                    parking_phase += 1;
                }
                
                else {
                    control.steering = 30.0;
                    control.velocity = -5.0;
                    control.brake = 0.0; // 0.2;
                    control.publishControlMsg();
                }
            }
            
            // [STOP PARKING] Back Moving (2)
            else if(parking_phase == 1){
                ROS_INFO("[STOP PARKING] Back Moving (2)");
                if(diff <= 3){
                    control.steering = 0.0;
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();

                    control.timer.start();
                    if(!control.timer.isTimeout()){
                        // ROS_INFO("[STOP PARKING] Vehicle Stabilization");
                        continue;
                    }

                    control.start_yaw = -400;

                    control.timer.resetTimer(3.5);
                    parking_phase += 1;
                } 
                
                else {
                    control.steering = -30.0;
                    control.velocity = -5.0;
                    control.brake = 0.0; // 0.2;
                    control.publishControlMsg();
                }
            }

            // [STOP PARKING] Pose Sorting & Stop 3.5 sec?
            else if(parking_phase == 2){
                ROS_INFO("[STOP PARKING] Pose Sorting & Stop 3.5 sec?");
                
                // @@@@@@@@@@@@@@@@@@@@@@@@@@@
                // @@@@ Pose Sorting..... @@@@ 
                // @@@@@@@@@@@@@@@@@@@@@@@@@@@

                control.steering = 0.0;
                control.velocity = 0.0;
                control.brake = 1.0;
                control.publishControlMsg();
                
                control.timer.start();
                if(!control.timer.isTimeout()){
                    // ROS_INFO("[STOP PARKING] park");
                    continue;
                }

                control.brake = 0.0;
                parking_phase += 1;
                control.timer.resetTimer(3.0);
            }

            // [STOP PARKING] Front Moving (1)
            else if(parking_phase == 3){
                ROS_INFO("[STOP PARKING] Front Moving (1)");
                if(diff >= parking_mid_angle){
                    control.steering = 0.0;
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();

                    control.timer.start();
                    if(!control.timer.isTimeout()){
                        // ROS_INFO("[STOP PARKING] Vehicle Stabilization");
                        continue;
                    }

                    control.brake = 0.0;

                    parking_phase += 1;
                    control.timer.resetTimer(3.0);
                }
                
                else {
                    control.steering = 30.0;
                    control.velocity = 8.0;
                    control.publishControlMsg();
                }
            }

            // [STOP PARKING] Front Moving (2)
            else if(parking_phase == 4){
                ROS_INFO("[STOP PARKING] Front Moving (2)");
                if(diff <= 3){
                    control.steering = 0.0;
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();

                    parking_phase += 1;
                    control.brake = 0.0;    
                    control.optimal_throttle.shouldStop = false;
                    control.optimal_throttle.shouldDecel = false;
                    control.optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();
                } 
                
                else {
                    control.steering = -30.0;
                    control.velocity = 8.0;
                    control.publishControlMsg();
                }
            }
        }

        else if(control.is_local_path_set){
            control.lookahead_distance = control.local_lookahead_distance;
            control.velocity = control.local_velocity;
            control.calcSteering(control.curr_pose);
            control.publishControlMsg();
        }

        else if(control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_DELIVERY_B && control.optimal_throttle.shouldStop){
            // 이 부분은 path 생성시 delivery line 기반으로 좀 쭉 늘려놔서 제어 안정성을 늘려 놓고
            // 몇 정도에서 정지시킬지 정해야 함.

            if(delivery_phase == -1){
                if(control.delivery_curr_idx < control.delivery_path_size - 1){
                    control.lookahead_distance = control.local_lookahead_distance;
                    control.velocity = control.local_velocity;
                    control.calcSteering(control.curr_pose);
                    control.publishControlMsg();

                    control.timer.resetTimer(5.3);
                }

                // 암튼 윗 주석에서 말한 임계점에 도달을 했다면 멈췄다가 출발
                else {
                    control.brake = 1.0;
                    control.velocity = 0.;
                    control.steering = 0.;
                    control.publishControlMsg();

                    control.timer.start();
                    if(!control.timer.isTimeout()){
                        ROS_INFO("[STOP DELIVERY] Vehicle Stabilization");
                        continue;
                    }

                    control.start_yaw = -400;
                    control.timer.resetTimer(0.5);
                    delivery_phase += 1;
                }
            }

            else if(delivery_phase == 0){
                if(control.start_yaw == -400) continue;
                int diff = control.curr_yaw - control.start_yaw;
            
                if(diff >= 180)
                    diff = abs(diff - 360);
                
                else if(diff <= -180)
                    diff = abs(diff + 360);

                if(diff >= 10){
                    control.steering = 0.0;
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();

                    control.timer.start();
                    if(!control.timer.isTimeout()){
                        ROS_INFO("[STOP DELIVERY] Vehicle Stabilization");
                        continue;
                    }

                    control.brake = 0.0;
                    control.start_yaw = -400;
                    control.timer.resetTimer(0.5);
                    delivery_phase += 1;
                }
                
                else {
                    control.steering = 30.0 / 180 * M_PI;
                    control.velocity = -5.0;
                    control.brake = 0.2;
                    control.publishControlMsg();
                }
            }
        
            else if(delivery_phase == 1){
                if(control.start_yaw == -400) continue;
                int diff = control.curr_yaw - control.start_yaw;

                if(diff >= 20){
                    delivery_phase += 1;

                    control.brake = 0.0;
                    control.optimal_throttle.shouldStop = false;
                    control.optimal_throttle.shouldDecel = false;
                    control.optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();

                    continue;
                }
                
                else {
                    control.steering = 30.0;
                    control.velocity = 8.0;
                    control.publishControlMsg();
                }
            }
        }

        else if(!control.is_finish && control.is_path_set && control.is_pose_set){
            control.lookahead_distance = control.global_lookahead_distance;
            control.velocity = control.optimal_throttle.calcTanhVelocity(control.curr_pose);
            control.brake = (control.velocity < control.speed_now) ? 0.15 : 0.0;
            ROS_INFO_STREAM("brake : " << control.brake);
            ROS_INFO_STREAM("curr_idx : " << control.g_curr_idx);
            ROS_INFO_STREAM("velocity : " << control.velocity << ", speed_now : " << control.speed_now);

            const auto& event_type = control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type;
            if(event_type == EventType::STOP_LINE || event_type == EventType::STOP_DELIVERY_A)
                control.brake = (control.optimal_throttle.shouldStop) ? 1.0 : 0.0;

            else if(event_type == EventType::STOP_TRAFFIC){
                if(control.isLeftTurn() && !control.isLeftSign()) {
                    control.brake = 1.0;
                }

                else if(!control.isLeftTurn() && !control.isGreenSign()) {
                    control.brake = 1.0;
                }

                else {
                    control.brake = 0.0;
                    control.optimal_throttle.shouldStop = false;
                    control.optimal_throttle.shouldDecel = false;
                    control.optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();
                }
            }

            control.calcSteering(control.curr_pose);
            control.publishControlMsg();
        }

        else if(control.is_finish){
            ROS_INFO("FINISH");

            control.velocity = 0.;
            control.brake = 1.;

            control.publishControlMsg();
        }
        
        loop_rate.sleep();
    }

    return 0;
}

void clearConsole() {
    // ANSI escape code를 사용하여 화면을 지움
    std::cout << "\033[2J\033[1;1H"; // 화면 지우기 및 커서를 맨 위로 이동
}

void logging(const PurePursuit& control){
    // clearConsole();

    // // ROS_DEBUG_STREAM("absolute of difference between curr_yaw and target_yaw : " << (std::abs(curr_yaw - target_yaw) >= 30));

    // fmt::print("{:<10} {:<10} {:<10}\n", "g_curr_idx", "g_next_idx", "is_local_path_set");
    // fmt::print("{:<10} {:<10}\n", control.g_curr_idx, control.g_next_idx, control.is_local_path_set);

    // fmt::print("{:<10} {:<10} {:<10}\n", "velocity", "steering", "brake");
    // fmt::print("{:<10} {:<10.4f} {:<10}\n", control.velocity, control.steering, control.brake);

    // fmt::print("{:<15} {:<20} {:<10} {:<10}\n", "event_place_idx", "event_place_type", "elapsed time", "total time");
    // fmt::print("{:<15} ", control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().index);
    
    // if(control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_DELIVERY_A){
    //     fmt::print("{:<20} ", "STOP_DELIVERY_A");
    // } else if(control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_DELIVERY_B){
    //     fmt::print("{:<20} ", "STOP_DELIVERY_B");
    // } else if(control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_PARKING){
    //     fmt::print("{:<20} ", "STOP_PARKING");
    // } else if(control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type == EventType::STOP_LINE){
    //     fmt::print("{:<20} ", "STOP_LINE");
    // } else {
    //     fmt::print("{:<20} ", "END");    
    // }

    // fmt::print("{:<8.2f} {:<8.2f}\n", control.timer.getTimeSpan(), control.timer.getTargetTime());

    // fmt::print("{:<10} {:<10}\n", "shouldStop", "shouldDecel");
    // fmt::print("{:<10} {:<10}\n", control.optimal_throttle.shouldStop, control.optimal_throttle.shouldDecel);

    // if(!control.is_finish && control.is_path_set && control.is_pose_set){
    //     double roll, pitch, path_yaw, curr_yaw;
    //     control.quaternionToEuler(control.g_path.path[control.g_curr_idx].pose.orientation, roll, pitch, path_yaw);
    //     control.quaternionToEuler(control.curr_pose.orientation, roll, pitch, curr_yaw);
    //     fmt::print("{:<10} {:<10} {:<10}\n", "curr_yaw", "path_yaw", "diff");
    //     fmt::print("{:<10.2f} {:<10.2f} {:<10.2f}\n", curr_yaw, path_yaw, abs(curr_yaw - path_yaw));
    // }
}
