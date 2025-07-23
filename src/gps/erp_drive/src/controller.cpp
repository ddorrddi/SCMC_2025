#include "controller.h"

Controller::Controller(char control_target):
    optimal_throttle(), timer(0), is_finish(false), is_delivery_path_set(false), 
    is_path_set(false), is_pose_set(false), is_local_path_set(false), g_curr_idx(0), l_curr_idx(0), delivery_curr_idx(0),
    velocity(0.), global_velocity(0.), local_velocity(0.), is_moving_front(false){
        this->delivery_stop_once = false;

        this->is_erp = control_target == 'e'; // e => erp (true), m => morai (false)
       
        this->erp_drive_pub = nh.advertise<race::drive_values>("control_value", 1);
        this->morai_drive_pub = nh.advertise<morai_msgs::CtrlCmd>("/ctrl_cmd", 1);

        this->curr_waypoint_pub = nh.advertise<std_msgs::Int16>("/curr_idx", 1);
        this->curr_point_pub = nh.advertise<geometry_msgs::PointStamped>("/curr_waypoint", 1);

        this->speed_sub = nh.subscribe("/gps_velocity", 1, &Controller::speedCallback, this);

        this->g_path_sub = nh.subscribe("/global_path", 1, &Controller::globalPathCallback, this);
        
        this->l_path_sub = nh.subscribe("/local_path", 1, &Controller::localPathCallback, this);
        this->l_path_done_sub = nh.subscribe("/local_path_done", 1, &Controller::localPathDoneCallback, this); 

        this->delivery_path_sub = nh.subscribe("/delivery_path", 1, &Controller::deliveryPathCallback, this);
        this->delivery_done_sub = nh.subscribe("/delivery_done", 1, &Controller::deliveryDoneCallback, this);
        this->delivery_end_sub = nh.subscribe("/delivery_end", 1, &Controller::deliveryEndCallback, this);
        
        this->curr_pose_sub = nh.subscribe("/current_pose", 1, &Controller::currentPoseCallback, this);
    
        this->imu_sub = nh.subscribe("/handsfree/imu", 1, &Controller::imuCallback, this);

        this->traffic_light_status_sub = nh.subscribe("/traffic_light_status", 1, &Controller::trafficLightStatusCallback, this);

        this->avoidance_direction_sub = nh.subscribe("/avoidance_direction", 1, &Controller::avoidanceDirectionCallback, this);
    }

void Controller::speedCallback(const std_msgs::Float64& speed_now){
    this->speed_now = speed_now.data;
}

void Controller::globalPathCallback(const erp_drive::PathReference& g_path){
    this->g_path = g_path;
    this->g_path_size = g_path.path.size();
    this->is_path_set = true;
}

void Controller::localPathCallback(const erp_drive::PathReference& l_path){
    this->l_path = l_path;
    this->l_path_size = l_path.path.size();
    this->is_local_path_set = true;
}

void Controller::localPathDoneCallback(const std_msgs::Int16& signal){
    ROS_INFO_STREAM("Done");
    this->is_local_path_set = false;
    this->l_curr_idx = 0;
    this->is_avoidance_success = true;
    this->has_turned_after_avoidance = false;
}

void Controller::deliveryPathCallback(const erp_drive::PathReference& delivery_path){
    this->delivery_path = delivery_path;
    this->delivery_path_size = delivery_path.path.size();
    this->is_delivery_path_set = true;
}

void Controller::deliveryDoneCallback(const std_msgs::Int16& signal){
    ROS_INFO_STREAM("Done");
    if(!this->delivery_stop_once){ // 콜백 한번만 실행용
        if(signal.data == 1){
            this->is_delivery_done = true;
            this->delivery_stop_once = true;
        }
        else{this->is_delivery_done = false;}
    }
}

void Controller::deliveryEndCallback(const std_msgs::Int16& signal){
    ROS_INFO_STREAM("End");
    if(signal.data == 1){
        this->is_delivery_end = true;
        this->is_delivery_path_set = false;
    }
    else{this->is_delivery_end = false;}
}

void Controller::imuCallback(const sensor_msgs::Imu& imu_data){
    this->imu = imu_data;

    double roll, pitch;

    tf::Quaternion q(
        this->imu.orientation.x,
        this->imu.orientation.y,
        this->imu.orientation.z,
        this->imu.orientation.w
    );

    tf::Matrix3x3 m(q);
    m.getRPY(roll, pitch, this->curr_yaw);
    this->curr_yaw *= (180 / M_PI);
    this->curr_yaw += 180;
    
    if(this->start_yaw == -400) this->start_yaw = static_cast<int>(this->curr_yaw);
}

// 직진 혹은 우회전에 대한 비교 함수
bool Controller::isGreenSign(){
    bool ret = this->traffic_light_status.compare(TrafficLightStatus::GREEN3) == 0;
    ret = ret || this->traffic_light_status.compare(TrafficLightStatus::GREEN4) == 0;
    ret = ret || this->traffic_light_status.compare(TrafficLightStatus::GREEN_LEFT4) == 0;
    return ret;
}

bool Controller::isLeftSign(){
    bool ret = this->traffic_light_status.compare(TrafficLightStatus::RED_LEFT4) == 0;
    ret = ret || this->traffic_light_status.compare(TrafficLightStatus::GREEN_LEFT4) == 0;
    return ret;
}

bool Controller::isLeftTurn() {
    calcGlobalCurrWaypoint(curr_pose);
    geometry_msgs::Pose curr_waypoint, target_waypoint;
    
    curr_waypoint = this->g_path.path.at(this->g_curr_idx).pose;
    target_waypoint = this->g_path.path.at(std::min(this->g_curr_idx + 30, this->g_path_size - 1)).pose;
    
    double roll, pitch, curr_yaw, target_yaw;

    tf2::Quaternion _curr_yaw(
        curr_waypoint.orientation.x,
        curr_waypoint.orientation.y,
        curr_waypoint.orientation.z,
        curr_waypoint.orientation.w
    );
    tf2::Matrix3x3(_curr_yaw).getRPY(roll, pitch, curr_yaw);

    tf2::Quaternion _target_yaw(
        target_waypoint.orientation.x,
        target_waypoint.orientation.y,
        target_waypoint.orientation.z,
        target_waypoint.orientation.w
    );
    
    tf2::Matrix3x3(_target_yaw).getRPY(roll, pitch, target_yaw);

    curr_yaw = curr_yaw * 180 / M_PI;
    target_yaw = target_yaw * 180 / M_PI;

    // @@@ TODO @@@ 적절한 각도 찾기
    return std::abs(curr_yaw - target_yaw) >= 30;
}

void Controller::trafficLightStatusCallback(const std_msgs::String& msg){
    bool isNone = msg.data.compare(TrafficLightStatus::NONE) == 0;
    if(!isNone)
        this->traffic_light_status = msg.data;
}

void Controller::currentPoseCallback(const geometry_msgs::Pose& curr_pose){
    this->curr_pose = curr_pose;
    this->is_pose_set = true;
}

void Controller::flowControl(const geometry_msgs::Pose& curr_pose){
    if(!this->is_path_set || !this->is_pose_set)
        return;

    if(this->optimal_throttle.eventPlaceInfoManager.isEventPlaceInfoEnd())
        return;

    int stop_idx = this->optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().index;

    if(this->g_path_size < stop_idx) return;
    double dist = this->optimal_throttle.eventPlaceInfoManager.calcCurrPoseToEventPlaceDist(curr_pose, this->g_path.path.at(stop_idx).pose);

    if(this->optimal_throttle.shouldDecel){
        switch(this->optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type){
        case EventType::STOP_TRAFFIC:
            if(stop_idx - 10 <= this->g_curr_idx){
                this->optimal_throttle.shouldDecel = false;
                this->optimal_throttle.shouldStop = true;
            }

            break;
        
        case EventType::STOP_LINE:
        case EventType::STOP_DELIVERY_A:
            if(stop_idx <= this->g_curr_idx){
                this->optimal_throttle.shouldDecel = false;
                this->optimal_throttle.shouldStop = true;
                this->is_timer_on = false;
            }
            
            break;

        case EventType::STOP_DELIVERY_B:
            if(!this->is_delivery_path_set){
                if(this->g_curr_idx >= 1468){
                    this->brake = 0.0;
                    this->optimal_throttle.shouldDecel = false;
                    this->optimal_throttle.shouldStop = false;
                    this->optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();
                }
                
                return;
            }

            ROS_INFO_STREAM("I get path");

            // 여기서는 path 가 확정된 상태이니 그냥 바로 Stop 으로 넘어감
            this->optimal_throttle.shouldDecel = false;
            this->optimal_throttle.shouldStop = true;
            break;

        default: break;
        }
    }

    // 이때 감속 제어는 optimal_throttle 에서
    else if(this->optimal_throttle.shouldStop){
        switch(this->optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type){
        case EventType::STOP_TRAFFIC:
            ROS_INFO("Traffic!!!!!!!!!!!");
            break;

        case EventType::STOP_LINE:
        case EventType::STOP_DELIVERY_A:
            if (!this->is_timer_on) { // 타이머가 시작되지 않았다면
                ROS_INFO("[STOP_LINE] Timer starting for 3 seconds.");
                this->timer.start();    // 타이머를 시작하고
                this->is_timer_on = true; // 플래그를 true로 변경
            }
        
            if (!this->timer.isTimeout()) {// 타이머 만료를 기다림
                ROS_INFO_STREAM("[STOP_LINE] Waiting... Elapsed time: " );
                return;
            }

            ROS_INFO("[STOP_LINE] Timeout! Resuming driving.");

            this->brake = 0.0;
            this->optimal_throttle.shouldStop = false;
            this->optimal_throttle.shouldDecel = false;
            this->optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();

            break;

        case EventType::STOP_DELIVERY_B:
            break;

        default: break;
        }
    }

    // optimal_throttle.DISTANCE (15m) 밖 부터 체크 시작 
    else if(dist < this->optimal_throttle.DISTANCE){
        switch(this->optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type){
        case EventType::STOP_TRAFFIC:
            this->optimal_throttle.shouldDecel = true;
            break;

        case EventType::STOP_LINE:
            this->timer.resetTimer(3);
            this->optimal_throttle.shouldDecel = true;
            break;

        case EventType::STOP_DELIVERY_A:
            this->timer.resetTimer(5);
            this->optimal_throttle.shouldDecel = true;
            break;

        case EventType::STOP_DELIVERY_B:
            this->timer.resetTimer(5);
            this->optimal_throttle.shouldDecel = true;
            break;

        default: break;
        }
    }
}

int Controller::calcGlobalCurrWaypoint(const geometry_msgs::Pose& curr_pose){
    // 1. Initial Position Search Logic
    // This runs only once when the is_initial_idx_found flag is false.
    if (!is_initial_idx_found) {
        ROS_INFO("[Controller] Starting global search for initial position...");
        double min_dist_init = 1e9;
        int initial_idx = -1;

        // Searches the entire path to find the closest point by distance only.
        for (int i = 0; i < this->g_path_size; i++) {
            double curr_dist = this->getDistance(this->g_path.path[i].pose, curr_pose);
            if (min_dist_init > curr_dist) {
                min_dist_init = curr_dist;
                initial_idx = i;
            }
        }

        if (initial_idx != -1) {
            this->g_curr_idx = initial_idx;
            this->is_initial_idx_found = true; // Set flag to true to prevent this block from running again.
            ROS_INFO("[Controller] Initial position found: Index %d", this->g_curr_idx);
        }
        // Return immediately after finding the initial position to start normal operation from the next cycle.
        return this->g_curr_idx;
    }


    // 2. Limited Range Search Logic (runs only after the initial position is found)
    int min_idx = -1;
    double min_dist = 1e9;
    double pitch, roll, path_yaw, curr_yaw;

    int search_start_idx = std::max(0, this->g_curr_idx - 20);
    int search_end_idx = std::min(this->g_path_size, this->g_curr_idx + 200);

    // 2-1. Primary Search: Considers both distance and Yaw angle within the limited range.
    for(int i = search_start_idx; i < search_end_idx; i++){
        double curr_dist = this->getDistance(this->g_path.path[i].pose, curr_pose);
        quaternionToEuler(this->g_path.path[i].pose.orientation, roll, pitch, path_yaw);
        quaternionToEuler(curr_pose.orientation, roll, pitch, curr_yaw);
        
        if(min_dist > curr_dist && abs(curr_yaw - path_yaw) <= M_PI / 6) {
            min_dist = curr_dist;
            min_idx = i;
        }
    }

    // 2-2. Secondary Search: If primary search fails, considers distance only within the limited range.
    if(min_idx == -1){
        ROS_WARN("[Controller] Primary search failed! Starting secondary search within the limited range.");
        min_dist = 1e9;
        for(int i = search_start_idx; i < search_end_idx; i++){
            double curr_dist = this->getDistance(this->g_path.path[i].pose, curr_pose);
            if(min_dist > curr_dist) {
                min_dist = curr_dist;
                min_idx = i;
            }
        }
    }
    
    // 2-3. Index Update and Jump Prevention
    if (min_idx != -1) {
        if (std::abs(min_idx - this->g_curr_idx) > 50) {
             ROS_WARN("[Controller] Index jump detected! (curr: %d, new: %d) Ignoring update.", this->g_curr_idx, min_idx);
        } else {
            this->g_curr_idx = min_idx;
        }
    }
    
    if(this->g_curr_idx >= this->g_path_size - 1)
        this->is_finish = true;

    std_msgs::Int16 msg;
    msg.data = this->g_curr_idx;
    this->curr_waypoint_pub.publish(msg);

    return this->g_curr_idx;
}

int Controller::calcLocalCurrWaypoint(const geometry_msgs::Pose& curr_pose){
    int min_idx = -1;
    double min_dist = 1e9;

    for(int i = l_curr_idx; i < this->l_path_size; i++){
        double curr_dist = this->getDistance(this->l_path.path[i].pose, curr_pose);
        if(min_dist > curr_dist) {
            min_dist = curr_dist;
            min_idx = i;
        }
    }

    this->l_curr_idx = min_idx;

    if(this->l_curr_idx == this->l_path_size - 1){
        this->is_local_path_set = false;
        this->l_curr_idx = 0;
    }

    return this->l_curr_idx;
}

int Controller::calcDeliveryCurrWaypoint(const geometry_msgs::Pose& curr_pose){
    int min_idx = -1;
    double min_dist = 1e9;

    for(int i = delivery_curr_idx; i < this->delivery_path_size; i++){
        double curr_dist = this->getDistance(this->delivery_path.path[i].pose, curr_pose);
        if(min_dist > curr_dist) {
            min_dist = curr_dist;
            min_idx = i;
        }
    }

    this->delivery_curr_idx = min_idx;

    if(this->delivery_curr_idx == this->delivery_path_size - 1){
        this->is_delivery_path_set = false;
        this->delivery_curr_idx = 0;
    }

    return this->delivery_curr_idx;
}

void Controller::publishControlMsg(){
    // ERP로 주행 시 실행되는 블럭
    if(this->is_erp){
        race::drive_values msg;
        msg.throttle = this->velocity; // 6.; // 장애물 회피
        // 주의! deg 단위임
        msg.steering = -this->steering * (180 / M_PI);
        if(msg.throttle < 0)
            msg.steering *= -1;

        msg.brake = this->brake;

        this->erp_drive_pub.publish(msg);
    }

    // 모라이로 주행 시 실행되는 블럭
    else {
        morai_msgs::CtrlCmd msg;
        msg.longlCmdType = 2;
        // msg.accel = 10.0;
        msg.velocity = fabs(this->velocity);
        // 주의! rad 단위임
        if (this->velocity < 0) {
            msg.steering = this->steering * -1;
        } else {
            msg.steering = this->steering;
        }

        msg.brake = this->brake;

        this->morai_drive_pub.publish(msg);    
    }
}


void Controller::avoidanceDirectionCallback(const std_msgs::Int16& msg){
    // planner가 보낸 ccw 값을 저장합니다 (1: 좌회피, -1: 우회피)
    this->last_avoidance_direction = msg.data;
    ROS_INFO_STREAM("[Controller] Avoid direction: " << this->last_avoidance_direction);
}