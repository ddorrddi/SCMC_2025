#include "controller.h"

Controller::Controller(char control_target):

    optimal_throttle(), timer(0), is_finish(false), is_delivery_path_set(false),
    is_path_set(false), is_pose_set(false), is_local_path_set(false), is_initial_idx_found(false), g_curr_idx(0), l_curr_idx(0), delivery_curr_idx(0),
    velocity(0.), global_velocity(0.), local_velocity(0.), is_moving_front(false), is_moving_back(false), go_to_finish(false),
    is_parking_done(false), is_parking_end(false), current_state(VehicleState::GLOBAL_PATH_DRIVING){
        this->delivery_stop_once = true;
        this->parking_stop_once = true;
        this->parking_done_once = true;
        this->parking_end_once = true;

        this->is_erp = control_target == 'e'; // e => erp (true), m => morai (false)
       
        this->erp_drive_pub = nh.advertise<race::drive_values>("control_value", 1);
        this->morai_drive_pub = nh.advertise<morai_msgs::CtrlCmd>("/ctrl_cmd", 1);

        this->curr_waypoint_pub = nh.advertise<std_msgs::Int16>("/curr_idx", 1);
        this->curr_point_pub = nh.advertise<geometry_msgs::PointStamped>("/curr_waypoint", 1);

        // 방향 전환 트리거 Publisher
        this->reverse_direction_trigger_pub = nh.advertise<std_msgs::Int16>("/reverse_direction_trigger", 1);

        this->speed_sub = nh.subscribe("/gps_velocity", 1, &Controller::speedCallback, this);

        this->g_path_sub = nh.subscribe("/global_path", 1, &Controller::globalPathCallback, this);
        
        this->l_path_sub = nh.subscribe("/local_path", 1, &Controller::localPathCallback, this);
        this->l_path_done_sub = nh.subscribe("/local_path_done", 1, &Controller::localPathDoneCallback, this); 

        this->delivery_done_sub = nh.subscribe("/delivery_done", 1, &Controller::deliveryDoneCallback, this);

        this->parking_stop_done_sub = nh.subscribe("/parking_stop_done", 1, &Controller::parkingStopDoneCallback, this);
        this->parking_path_sub = nh.subscribe("/parking_path", 1, &Controller::parkingPathCallback, this);
        this->parking_done_sub = nh.subscribe("/parking_done", 1, &Controller::parkingDoneCallback, this);
        this->return_path_sub = nh.subscribe("/return_path", 1, &Controller::returnPathCallback, this);
        this->parking_end_sub = nh.subscribe("/parking_end", 1, &Controller::parkingEndCallback, this);
        
        this->curr_pose_sub = nh.subscribe("/current_pose", 1, &Controller::currentPoseCallback, this);
    
        this->imu_sub = nh.subscribe("/handsfree/imu", 1, &Controller::imuCallback, this);

        this->traffic_light_status_sub = nh.subscribe("/traffic_light_status", 1, &Controller::trafficLightStatusCallback, this);

        this->deceleration_request_sub = nh.subscribe("/deceleration_request", 1, &Controller::decelerationRequestCallback, this);
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
    this->current_state = VehicleState::LOCAL_PATH_DRIVING;
}

void Controller::localPathDoneCallback(const std_msgs::Int16& signal){
    ROS_INFO_STREAM("Done");
    this->is_local_path_set = false;
    this->l_curr_idx = 0;
    this->is_avoidance_success = true;

    this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
}

void Controller::deliveryDoneCallback(const std_msgs::Int16& signal){
    if(this->delivery_stop_once){ // 콜백 한번만 실행용
        if(signal.data == 1){
            ROS_INFO_STREAM("Delivery Done");
            this->is_delivery_done = true;
            this->delivery_stop_once = false;
        }
        else{this->is_delivery_done = false;}
    }
}

void Controller::parkingStopDoneCallback(const std_msgs::Int16& signal){
    if(this->parking_stop_once){
        if(signal.data == 1){
            ROS_INFO_STREAM("[Controller] Parking Stop Done signal received.");
            this->is_parking_stop_done = true;
            this->parking_stop_once = false;

            this->current_state = VehicleState::PARKING_STOPPING;
        }
        else{this->is_parking_stop_done = false;}
    }
}

void Controller::parkingPathCallback(const erp_drive::PathReference& parking_path){
    // 나중에 인덱스 기준으로 콜백 종료 조건 추가
    ROS_INFO_STREAM("[Controller] Received new parking path.");

    this->l_path = parking_path;
    this->l_path_size = parking_path.path.size();
    this->is_local_path_set = true;
    this->l_curr_idx = 0;

    // 방향 전환 트리거 발행 (전진 → 후진)
    std_msgs::Int16 trigger_msg;
    trigger_msg.data = 1;
    this->reverse_direction_trigger_pub.publish(trigger_msg);

    ROS_INFO("[Controller] Published reverse direction trigger (forward → reverse)");

    this->current_state = VehicleState::PARKING_REVERSING;
}

void Controller::parkingDoneCallback(const std_msgs::Int16& signal){
    if(this->parking_done_once){
        if(signal.data == 1){
            ROS_INFO_STREAM("[Controller] Parking Done signal received.");
            this->is_parking_done = true;
            this->parking_done_once = false;
            this->is_local_path_set = false;
            this->l_curr_idx = 0;

            this->current_state = VehicleState::PARKING_DONE;
        }
        else{
            this->is_parking_done = false;
        }
    }
}

void Controller::returnPathCallback(const erp_drive::PathReference& return_path){
    // 나중에 인덱스 기준으로 콜백 종료 조건 추가
    ROS_INFO_STREAM("[Controller] Received new return path.");

    this->l_path = return_path;
    this->l_path_size = return_path.path.size();
    this->is_local_path_set = true;
    this->l_curr_idx = 0;

    // 방향 전환 트리거 발행 (후진 → 전진)
    std_msgs::Int16 trigger_msg;
    trigger_msg.data = 1;
    this->reverse_direction_trigger_pub.publish(trigger_msg);
    ROS_INFO("[Controller] Published reverse direction trigger (reverse → forward)");
}

void Controller::parkingEndCallback(const std_msgs::Int16& signal){
    if(this->parking_end_once){
        if(signal.data == 1){
            ROS_INFO("[Controller] Parking End signal received. Returning to global path driving.");
            this->is_parking_end = true;
            this->parking_end_once = false;
            this->is_local_path_set = false;
            this->l_curr_idx = 0;

            this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
        }
    }
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

void Controller::trafficLightStatusCallback(const std_msgs::String& msg){
    bool isNone = msg.data.compare(TrafficLightStatus::NONE) == 0;
    if(!isNone)
        this->traffic_light_status = msg.data;
}

void Controller::currentPoseCallback(const geometry_msgs::Pose& curr_pose){
    // 항상 원본 pose 사용
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
            // [개선 1] 감속 중 초록불이면 바로 통과
            if(this->isGreenSign()){
                ROS_INFO("[TRAFFIC LIGHT] GREEN detected during deceleration - skipping stop");
                this->optimal_throttle.shouldDecel = false;
                this->optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();
                break;
            }

            // [개선 3] -10에서 -5로 변경하여 신호등 인식 시간 확보
            if(stop_idx - 5 <= this->g_curr_idx){
                // [개선 2] 초록불이면 shouldStop으로 전환하지 않고 통과
                if(this->isGreenSign()){
                    ROS_INFO("[TRAFFIC LIGHT] GREEN at stop point - proceeding without stop");
                    this->optimal_throttle.shouldDecel = false;
                    this->optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();
                } else {
                    this->optimal_throttle.shouldDecel = false;
                    this->optimal_throttle.shouldStop = true;
                }
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


        default: break;
        }
    }

    // 이때 감속 제어는 optimal_throttle 에서
    else if(this->optimal_throttle.shouldStop){
        switch(this->optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type){
        case EventType::STOP_TRAFFIC:
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
            this->is_timer_on = false;
            this->optimal_throttle.shouldStop = false;
            this->optimal_throttle.shouldDecel = false;
            this->optimal_throttle.eventPlaceInfoManager.nextEventPlaceInfo();

            break;


        default: break;
        }
    }

    // Start deceleration 50 waypoints before stop point (index-based)
    else if(stop_idx - 50 <= this->g_curr_idx && this->g_curr_idx < stop_idx){
        switch(this->optimal_throttle.eventPlaceInfoManager.getCurrEventPlaceInfo().type){
        case EventType::STOP_TRAFFIC:
            ROS_INFO("[TRAFFIC LIGHT] Approaching traffic light (idx: %d/%d) - starting deceleration", this->g_curr_idx, stop_idx);
            this->optimal_throttle.shouldDecel = true;
            break;

        case EventType::STOP_LINE:
            ROS_INFO("[STOP_LINE] Approaching stop line (idx: %d/%d) - starting deceleration", this->g_curr_idx, stop_idx);
            this->timer.resetTimer(3);
            this->optimal_throttle.shouldDecel = true;
            break;

        case EventType::STOP_DELIVERY_A:
            ROS_INFO("[DELIVERY] Approaching delivery zone (idx: %d/%d) - starting deceleration", this->g_curr_idx, stop_idx);
            this->timer.resetTimer(5);
            this->optimal_throttle.shouldDecel = true;
            break;


        default: break;
        }
    }
}

int Controller::calcGlobalCurrWaypoint(const geometry_msgs::Pose& curr_pose){
    // 1. Initial Position Search Logic
    if (!is_initial_idx_found) {

        if (this->g_path_size == 0 || !this->is_path_set) {
            return this->g_curr_idx;
        }

        double min_dist_init = 1e9;
        int initial_idx = -1;

        // Get current vehicle's yaw
        double roll, pitch, vehicle_yaw;
        quaternionToEuler(curr_pose.orientation, roll, pitch, vehicle_yaw);

        // Vehicle's heading vector in 2D
        double vehicle_heading_x = cos(vehicle_yaw);
        double vehicle_heading_y = sin(vehicle_yaw);

        for (int i = 0; i < this->g_path_size; i++) {
            const auto& path_pose = this->g_path.path[i].pose;
            
            // Vector from vehicle to path point
            double vec_to_path_x = path_pose.position.x - curr_pose.position.x;
            double vec_to_path_y = path_pose.position.y - curr_pose.position.y;

            // Dot product to check if the point is in front of the vehicle
            double dot_product = vehicle_heading_x * vec_to_path_x + vehicle_heading_y * vec_to_path_y;

            // Consider only points that are generally in front of the vehicle
            if (dot_product > 0) {
                double curr_dist = this->getDistance(path_pose, curr_pose);
                if (min_dist_init > curr_dist) {
                    min_dist_init = curr_dist;
                    initial_idx = i;
                }
            }
        }
        
        // If no point is found in front, fall back to the original distance-only search
        if (initial_idx == -1) {
            for (int i = 0; i < this->g_path_size; i++) {
                double curr_dist = this->getDistance(this->g_path.path[i].pose, curr_pose);
                if (min_dist_init > curr_dist) {
                    min_dist_init = curr_dist;
                    initial_idx = i;
                }
            }
        }

        if (initial_idx != -1) {
            this->g_curr_idx = initial_idx;
            this->is_initial_idx_found = true;
        } 
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
        
        if(min_dist > curr_dist && abs(curr_yaw - path_yaw) <= M_PI / 4) {
            min_dist = curr_dist;
            min_idx = i;
        }
    }

    // 2-2. Secondary Search: If primary search fails, considers distance only within the limited range.
    if(min_idx == -1){
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
        // msg.throttle = this->velocity; // 6.; // 장애물 회피

        if (this->is_moving_back) {
            msg.throttle = -this->velocity; // 후진
        } else {
            msg.throttle = this->velocity; // 전진
        }

        // 지역 변수로 rad->deg 변환 (this->steering은 rad 단위 유지)
        double steering_deg = this->steering * (180.0 / M_PI);

        // CAN 통신 각도 제한 (-30 ~ 30도)
        steering_deg = std::max(-30.0, std::min(30.0, steering_deg));

        if (this->is_moving_back) {
            msg.steering = steering_deg * -1;
            // ROS_INFO("!!!!!!!!!!!!!!!!악");
        } else {
            msg.steering = steering_deg;
            // ROS_INFO("악!!!!!!!!!!!!!!!!");
        }

        msg.brake = (this->brake < 0.01) ? 0.0 : this->brake;

        this->erp_drive_pub.publish(msg);
    }

    // 모라이로 주행 시 실행되는 블럭
    else {
        morai_msgs::CtrlCmd msg;
        msg.longlCmdType = 2;
        // msg.accel = 10.0;
        msg.velocity = fabs(this->velocity);
        // 주의! rad 단위임
        if (this->is_moving_back) {
            msg.steering = this->steering * -1;
        } else {
            msg.steering = this->steering;
        }

        msg.brake = this->brake;

        this->morai_drive_pub.publish(msg);
    }
}

void Controller::decelerationRequestCallback(const std_msgs::Int16& msg){
    this->is_deceleration_requested = (msg.data == 1);
    if (this->is_deceleration_requested) {
        ROS_INFO("[Controller] Deceleration request received for large obstacle.");
    } else {
        ROS_INFO("[Controller] Resuming normal speed.");
    }
}