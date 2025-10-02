#include "pure_pursuit.h"
#include <ros/ros.h>

void paramCallback(erp_drive::CtrlParamConfig &config, PurePursuit* control) {
    control->optimal_throttle.setMaxVelocity(config.global_velocity);
    control->global_lookahead_distance = config.global_lookahead_distance;

    control->local_velocity = config.local_velocity;
    control->local_lookahead_distance = config.local_lookahead_distance;

    control->reverse_local_lookahead_distance = config.reverse_local_lookahead_distance;
    control->parking_local_lookahead_distance = config.parking_local_lookahead_distance;

}

void clearConsole();

int main(int argc, char** argv) {

    bool parking_done_hold_active = false;
    ros::Time parking_done_start_time;
    ros::Duration stop_elapsed;
    ros::Duration hold_elapsed;

    ros::init(argc, argv, "pure_pursuit");

    ROS_INFO("controller_node init");


    PurePursuit control(argv[1][0]);

    bool delivery_stop_once_flag = false; // 배달 목적지에서 멈출때 딱 한번쓰려고 만든 플래그
    ros::Time delivery_B_Stop_Time; // 배달 목적지에서 정지할때 쓸 타이머
    double delivery_B_Original_Velocity = 0.0; // 배달 목적지에서 천천히 감속하려고 쓰는 변수

    bool parking_stop_initiated = false;        // 주차를 위한 점진적 정차를 한 번만 실행하기 위한 플래그
    ros::Time parking_stop_start_time;          // 점진적 정차 시작 시간을 기록할 타이머
    double original_velocity_before_parking = 0.0;      // 정차 직전 속도를 저장하기 위한 변수

    ros::Rate loop_rate(control.LOOP_RATE);

    ros::Publisher target_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/target_waypoint", 1);
    ros::Publisher curr_waypoint_pub = control.nh.advertise<geometry_msgs::PointStamped>("/curr_waypoint", 1);

    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig> server;
    dynamic_reconfigure::Server<erp_drive::CtrlParamConfig>::CallbackType f;
    
    f = boost::bind(&paramCallback, _1, &control);
    server.setCallback(f);
    
    ros::Time message_received_time;

    // 주차 공간 각도 349

    while(ros::ok()){
        ros::spinOnce();

        if(!control.is_pose_set || !control.is_path_set) continue;

        ROS_INFO("Pose Initialized : Found Index");
        break;
    }

    control.calcGlobalCurrWaypoint(control.curr_pose);
    control.optimal_throttle.eventPlaceInfoManager.init(control.g_curr_idx);

    while(ros::ok()){
        ros::spinOnce();
        control.flowControl(control.curr_pose);

        switch(control.current_state){
            case VehicleState::GLOBAL_PATH_DRIVING:
                if(!control.is_finish && control.is_path_set && control.is_pose_set){
                    control.lookahead_distance = control.global_lookahead_distance;
                    control.velocity = control.optimal_throttle.calcTanhVelocity(control.curr_pose);

                    // PARKING 모드: 인덱스 3450부터 속도 5로 제한
                    if(control.g_curr_idx >= 3450 && !control.go_to_finish){
                        control.velocity = std::min(control.velocity, 5.0);
                    }
                    control.brake = (control.velocity < control.speed_now) ? 0.15 : 0.0;

                    const auto& event_type = control.optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type;
                    if(event_type == EventType::STOP_LINE || event_type == EventType::STOP_DELIVERY_A)
                        control.brake = (control.optimal_throttle.shouldStop) ? 1.0 : 0.0;

                    else if(event_type == EventType::STOP_TRAFFIC){
                        if (control.optimal_throttle.shouldStop) {
                            bool is_red_light = !control.isGreenSign();
                            ROS_INFO("[TRAFFIC LIGHT] Light state: %s", is_red_light ? "RED (STOP)" : "GREEN (GO)");

                            if (is_red_light) {
                                ROS_INFO("[TRAFFIC LIGHT] RED LIGHT - Vehicle stopping");
                                control.brake = 1.0;
                            } else {
                                ROS_INFO("[TRAFFIC LIGHT] GREEN LIGHT - Vehicle proceeding");
                                control.brake = 0.0;
                                control.optimal_throttle.shouldStop = false;
                                control.optimal_throttle.shouldDecel = false;
                                control.optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();
                            }
                        }
                    }

                    // 배달 완료 후 정지 후 출발
                    if(control.is_delivery_done){
                        if(!delivery_stop_once_flag){
                            ROS_INFO("delivery_stop : init");
                            delivery_B_Stop_Time = ros::Time::now();
                            delivery_B_Original_Velocity = control.velocity;
                            delivery_stop_once_flag = true;
                        }

                        ros::Duration elapsed = ros::Time::now() - delivery_B_Stop_Time;

                        if(elapsed < ros::Duration(0.5)){
                            ROS_INFO("delivery_stop : decel");
                            double ratio = 1.0 - (elapsed.toSec() / 0.5);
                            control.velocity = delivery_B_Original_Velocity * ratio;
                            control.brake = 0.0;
                        }else if(elapsed < ros::Duration(3.5)){
                            ROS_INFO("delivery_stop : stop");
                            control.velocity = 0.0;
                            control.brake = 1.0;
                        }else{
                            ROS_INFO("delivery_stop : go");
                            control.brake = 0.0;
                            control.is_delivery_done = false;
                        }
                    }

                    if (control.is_deceleration_requested) {
                        control.velocity = control.local_velocity;
                    }

                    control.calcSteering(control.curr_pose);
                    control.publishControlMsg();
                }
                break;

            case VehicleState::LOCAL_PATH_DRIVING:
                if(control.is_local_path_set){
                    ROS_INFO_STREAM("State: LOCAL_PATH_DRIVING");
                    if(control.is_parking_done){
                        control.lookahead_distance = control.parking_local_lookahead_distance;
                    } else {
                        control.lookahead_distance = control.local_lookahead_distance;
                    }
                    control.velocity = control.local_velocity;
                    control.brake = 0.0;
                    control.calcSteering(control.curr_pose);
                    control.is_moving_back = false;
                    control.publishControlMsg();
                }
                break;

            case VehicleState::PARKING_STOPPING:
                if(!parking_stop_initiated){
                    ROS_INFO("[Parking Logic] Initiating gradual stop for parking.");
                    parking_stop_start_time = ros::Time::now();
                    original_velocity_before_parking = control.velocity;
                    parking_stop_initiated = true;
                }

                stop_elapsed = ros::Time::now() - parking_stop_start_time;

                if(stop_elapsed < ros::Duration(2.5)){
                    double ratio = 1.0 - (stop_elapsed.toSec() / 2.5);
                    control.velocity = original_velocity_before_parking * ratio;
                    control.brake = 0.0;
                } else {
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.is_parking_stop_done = false;
                }
                control.publishControlMsg();
                break;

            case VehicleState::PARKING_REVERSING:
                if(control.is_local_path_set){
                    ROS_INFO_STREAM("State: PARKING_REVERSING (Following reverse path)");
                    control.is_moving_back = true;
                    control.lookahead_distance = control.reverse_local_lookahead_distance;
                    control.brake = 0.0;
                    control.velocity = 3.0;
                    control.calcSteering(control.curr_pose);
                    control.publishControlMsg();
                }
                break;

            case VehicleState::PARKING_DONE:
                if(!parking_done_hold_active){
                    ROS_INFO("[Parking Logic] Parking done. Holding position for 3 seconds.");
                    parking_done_start_time = ros::Time::now();
                    parking_done_hold_active = true;
                }

                hold_elapsed = ros::Time::now() - parking_done_start_time;

                if(hold_elapsed < ros::Duration(3.0)){
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();
                } else {
                    control.velocity = 0.0;
                    control.brake = 1.0;
                    control.publishControlMsg();
                }
                break;
        }

        if(control.is_parking_end){
            ROS_INFO("[Parking Logic] Parking end. Vehicle returning to global path.");
            control.is_parking_done = false;
            control.is_parking_end = false;
            control.go_to_finish = true;
            control.calcGlobalCurrWaypoint(control.curr_pose);
        }

        if(control.is_finish){
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