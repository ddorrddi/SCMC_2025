#include "pure_pursuit.h"
#include <ros/ros.h>

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

    bool isLogging = true; //argv[2][0] == 't';

    bool delivery_stop_once_flag = false; // 배달 목적지에서 멈출때 딱 한번쓰려고 만든 플래그
    ros::Time delivery_B_Stop_Time; // 배달 목적지에서 정지할때 쓸 타이머
    double delivery_B_Original_Velocity = 0.0; // 배달 목적지에서 천천히 감속하려고 쓰는 변수

    ros::Rate loop_rate(control.LOOP_RATE);

    ros::Publisher target_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/target_waypoint", 1);
    ros::Publisher curr_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/curr_waypoint", 1);

    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig> server;
    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig>::CallbackType f;
    
    f = boost::bind(&paramCallback, _1, &control);
    server.setCallback(f);
    
    ros::Time message_received_time;

    // 주차 공간 각도 349

    double prev_x = -1.0, prev_y = -1.0;


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

        // if (control.is_avoidance_success && !control.has_turned_after_avoidance) {
        //     ROS_INFO("AVOID!!!!!!!!!!!!!!!!!!!!!!!!!!!");

        //     const double steering_angle_rad = 7.0 * (M_PI / 180.0);
            
        //     // last_avoidance_direction == 1: 오른쪽 장애물을 왼쪽으로 회피 -> 오른쪽으로 꺾어 복귀
        //     if (control.last_avoidance_direction == 1) {
        //         ROS_INFO("RIGHT!!!!!!!!!!!!!!!!!");
        //         control.steering = -steering_angle_rad;  // 
        //     } 
        //     // last_avoidance_direction == -1: 왼쪽 장애물을 오른쪽으로 회피 -> 왼쪽으로 꺾어 복귀
        //     else if (control.last_avoidance_direction == -1) {
        //         ROS_INFO("LEFT!!!!!!!!!!!!!!!!!");
        //         control.steering = steering_angle_rad ; // 
        //     }

        //     control.velocity = 8.0; // 
        //     control.brake = 0.0;
        //     control.publishControlMsg();
            
        //     ros::Duration(1.0).sleep(); // 

        //     // 복귀 조향이 끝났으므로 플래그들을 리셋
        //     control.has_turned_after_avoidance = true;
        //     control.is_avoidance_success = false; 
        //     control.last_avoidance_direction = 0; 

        //     // 바로 다음 루프로 넘어가서 일반 주행 로직을 수행하도록 함
        //     continue; 
        // }

        if(isLogging && control.is_path_set && control.is_pose_set){
            logging(control);
        }

        if(control.is_local_path_set){
            control.lookahead_distance = control.local_lookahead_distance;
            control.velocity = control.local_velocity;
            control.calcSteering(control.curr_pose);
            control.publishControlMsg();
        }

        else if(control.is_delivery_done){
            if(!delivery_stop_once_flag){
                ROS_INFO("delivery_stop : init");
                delivery_B_Stop_Time = ros::Time::now();    // 정지 명령 시간 기록
                delivery_B_Original_Velocity = control.velocity;
                delivery_stop_once_flag = true;
            }
            
            ros::Duration elapsed = ros::Time::now() - delivery_B_Stop_Time;    // 경과 시간 계산

            if(elapsed < ros::Duration(0.5)){
                ROS_INFO("delivery_stop : decel");
                double ratio = 1.0 - (elapsed.toSec() / 0.5);
                control.velocity = delivery_B_Original_Velocity * ratio;
                control.brake = 0.0;
                control.publishControlMsg();
            }else if(elapsed < ros::Duration(3.5)){
                ROS_INFO("delivery_stop : stop");
                control.velocity = 0.0;
                control.brake = 1.0;
                control.publishControlMsg();
            }else{
                ROS_INFO("delivery_stop : go");
                control.brake = 0.0;
                control.publishControlMsg();
                control.is_delivery_done = false;
            }
        }

        else if(control.is_parking_stop_done){
            //여기 뭐 하드코딩해서 들어가도록 할까 어때??
            //아니면 parking_planner에서 로컬패스 만들어서 들어가는거는?
        }

        else if(!control.is_finish && control.is_path_set && control.is_pose_set){
            control.lookahead_distance = control.global_lookahead_distance;

            // if (control.g_curr_idx < 300) {
            //     control.velocity = 5.0; // 저속 설정
            // } else {
            //     control.velocity = control.optimal_throttle.calcTanhVelocity(control.curr_pose);
            // } delivery 구간에 맞게 수정

            control.velocity = control.optimal_throttle.calcTanhVelocity(control.curr_pose);
            
            // BIG 장애물 회피 구간 제한 속도 10
            if(710 < control.g_curr_idx && control.g_curr_idx < 930) {
                control.velocity = std::min(control.velocity, 10.0);
            }

            // PARKING 모드: 인덱스 3200부터 속도 5로 제한
            if(control.g_curr_idx >= 3200){
                ROS_INFO("[PARKING MODE] Speed limited to 5.0 (idx: %d)", control.g_curr_idx);
                control.velocity = std::min(control.velocity, 5.0);
            }
            control.brake = (control.velocity < control.speed_now) ? 0.15 : 0.0;
            // ROS_INFO_STREAM("brake : " << control.brake);
            // ROS_INFO_STREAM("curr_idx : " << control.g_curr_idx);
            // ROS_INFO_STREAM("velocity : " << control.velocity << ", speed_now : " << control.speed_now);

            const auto& event_type = control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type;
            if(event_type == EventType::STOP_LINE || event_type == EventType::STOP_DELIVERY_A)
                control.brake = (control.optimal_throttle.shouldStop) ? 1.0 : 0.0;

            else if(event_type == EventType::STOP_TRAFFIC){
                // 신호등 이벤트 감지 로그
                int traffic_event_idx = control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().index;
                
                // flowControl이 신호등에 가까워졌다고 판단했을 때만 (shouldStop == true) 아래 로직을 실행
                if (control.optimal_throttle.shouldStop) {
                    bool is_red_light = !control.isGreenSign();
                    
                    // 현재 신호등 상태 로그
                    ROS_INFO("[TRAFFIC LIGHT] Light state: %s", is_red_light ? "RED (STOP)" : "GREEN (GO)");
            
                    if (is_red_light) {
                        // 빨간불이면 계속 정지
                        ROS_INFO("[TRAFFIC LIGHT] RED LIGHT - Vehicle stopping");
                        control.brake = 1.0;
                    } else {
                        // 초록불이면 출발!
                        ROS_INFO("[TRAFFIC LIGHT] GREEN LIGHT - Vehicle proceeding");
                        control.brake = 0.0;
                        control.optimal_throttle.shouldStop = false;
                        control.optimal_throttle.shouldDecel = false;
                        control.optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();
                    }
                }
            }

            if (control.is_deceleration_requested) {
                // 원하는 감속 속도를 여기에 설정합니다.
                // 예: local_velocity보다 느리게 설정
                control.velocity = control.local_velocity; 
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
