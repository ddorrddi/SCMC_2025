#include "controller.h"

Controller::Controller(char control_target):
    current_state(VehicleState::GLOBAL_PATH_DRIVING),
    optimal_throttle(), is_finish(false), flag(true),
    is_path_set(false), is_pose_set(false), is_local_path_set(false), g_curr_idx(0), 
    velocity(0.), global_velocity(0.), local_velocity(0.), is_moving_back(false), is_parking_complete(false)
    {
        this->is_erp = control_target == 'e'; // e => erp (true), m => morai (false)
       
        this->global_path_finish_pub = nh.advertise<std_msgs::Bool>("/global_path_finish", 1);

        this->erp_drive_semi_pub = nh.advertise<race::drive_values>("control_value", 1);
        this->morai_drive_pub = nh.advertise<morai_msgs::CtrlCmd>("/ctrl_cmd", 1);

        this->curr_waypoint_pub = nh.advertise<std_msgs::Int16>("/curr_idx", 1);

        this->g_path_sub = nh.subscribe("/global_path", 1, &Controller::globalPathCallback, this);
        
        this->l_path_sub = nh.subscribe("/local_path", 1, &Controller::localPathCallback, this);

        this->parking_path_sub = nh.subscribe("/parking_path", 1, &Controller::parkingPathCallback, this);
        this->parking_path_done_sub = nh.subscribe("/parking_path_done", 1, &Controller::parkingPathDoneCallback, this);

        this->curr_pose_sub = nh.subscribe("/current_pose", 1, &Controller::currentPoseCallback, this);

        this->imu_sub = nh.subscribe("/handsfree/imu", 1, &Controller::imuCallback, this);

        this->client = nh.serviceClient<erp_drive_semi::InitializePose>("initialize_pose");

        this->morai_event_cmd_client = nh.serviceClient<morai_msgs::MoraiEventCmdSrv>("/Service_MoraiEventCmd");
        
    }

void Controller::globalPathCallback(const erp_drive_semi::PathReference& g_path){
    this->g_path = g_path;
    this->g_path_size = g_path.path.size();
    this->is_path_set = true;
    // this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
}

void Controller::localPathCallback(const erp_drive_semi::PathReference& l_path){
    if(this->g_curr_idx < 1080) return;

    if(this->start_yaw == -400)
        this->start_yaw = static_cast<int>(this->curr_yaw);

    this->l_path = l_path;
    this->l_path_size = l_path.path.size();
    this->is_local_path_set = true;
    if(this->g_curr_idx < 1380){
        this->current_state = VehicleState::LOCAL_PATH_DRIVING;
    } else {
        this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
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
    target_waypoint = this->g_path.path.at(std::max(this->g_curr_idx + 30, this->g_path_size - 1)).pose;
    
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


void Controller::currentPoseCallback(const geometry_msgs::Pose& curr_pose){
    this->curr_pose = curr_pose;
    this->is_pose_set = true;
}

int Controller::calcGlobalCurrWaypoint(const geometry_msgs::Pose& curr_pose){
    int min_idx = -1;
    double min_dist = 1e9;
  
    int len = std::min(this->g_curr_idx + 100, this->g_path_size);
    for(int i = this->g_curr_idx; i < this->g_path_size; i++){
        double curr_dist = this->getDistance(this->g_path.path[i].pose, curr_pose);
        
        if(min_dist > curr_dist) {
            min_dist = curr_dist;
            min_idx = i;
        }
    }

    this->g_curr_idx = min_idx;
    if(this->g_curr_idx >= this->g_path_size - 20){
        this->is_finish = true;
    }
    
    std_msgs::Int16 msg;
    msg.data = this->g_curr_idx;
    this->curr_waypoint_pub.publish(msg); //planner에서 활용

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

        this->erp_drive_semi_pub.publish(msg);
    }

    // 모라이로 주행 시 실행되는 블럭
    else {
        morai_msgs::CtrlCmd msg;
        msg.longlCmdType = 2;
        // msg.accel = 10.0;
        msg.velocity = this->velocity;
        // 주의! rad 단위임
        if (this->is_moving_back) {
            msg.steering = this->steering * -1;
        } else {
            msg.steering = this->steering;
        }

        msg.brake = this->brake;
        // ROS_INFO_STREAM("brake : " << msg.brake);

        this->morai_drive_pub.publish(msg);    
    }
}

void Controller::parkingPathCallback(const erp_drive_semi::PathReference& parking_path){

    if(this->g_curr_idx < 180 or this->g_curr_idx > 300) return;

    ROS_INFO("Received New Parking Path, treating as a local path.");
    
    // 받은 주차 경로를 '로컬 경로(l_path)' 변수에 그대로 복사합니다.
    this->l_path = parking_path;
    this->l_path_size = parking_path.path.size();
    
    // 로컬 경로 추종 모드를 활성화하여 Pure Pursuit가 이 경로를 따라가도록 합니다.
    this->is_local_path_set = true;
    
    // 새로운 주차를 시작하므로, 주차 완료 상태는 false로 리셋합니다.
    this->is_parking_complete = false; 
    this->l_curr_idx = 0; // 로컬 경로 인덱스 초기화

    this->current_state = VehicleState::LOCAL_PATH_DRIVING;
}

// '/parking_path_done' 토픽을 받았을 때 호출되는 함수
void Controller::parkingPathDoneCallback(const std_msgs::Int16& signal){
    if(signal.data == 1){
        ROS_INFO("Parking Path Done signal received. Stopping vehicle.");
        
        // 모든 경로 추종 모드를 비활성화합니다.
        this->is_local_path_set = false;
        
        this->current_state = VehicleState::PARKING_DONE_STOPPING;

        // '주차 완료 및 정지' 상태로 전환합니다.
        this->is_parking_complete = true; 
        this->l_curr_idx = 0;

        this->stop_timer = nh.createTimer(ros::Duration(3.0), &Controller::stopTimerCallback, this, true);
    }
}

void Controller::stopTimerCallback(const ros::TimerEvent&) {
    ROS_INFO("3-second stop finished. Switching to REVERSING state and changing gear to R.");
    
    // 1. 기어 변경 서비스 메시지 생성
    morai_msgs::MoraiEventCmdSrv srv;
    srv.request.request.option = 2; // Gear 제어
    srv.request.request.ctrl_mode = 3; // ExternalCtrl 모드
    srv.request.request.gear = 2; // R (후진)
    
    // 2. 서비스 호출
    if (this->morai_event_cmd_client.call(srv)) {
        ROS_INFO("Successfully changed gear to REVERSE.");
    } else {
        ROS_ERROR("Failed to call gear change service.");
    }

    // 3. 상태를 '후진'으로 변경
    this->current_state = VehicleState::PARKING_REVERSING;
    
    // 4. 3초 후에 reverseTimerCallback을 호출하는 타이머 시작
    this->reverse_timer = nh.createTimer(ros::Duration(3.0), &Controller::reverseTimerCallback, this, true);
}

void Controller::reverseTimerCallback(const ros::TimerEvent&) {
    ROS_INFO("3-second reverse finished. Switching to DRIVE and resuming global path.");
    
    // 1. 기어를 다시 D (전진)로 변경
    morai_msgs::MoraiEventCmdSrv srv;
    srv.request.request.option = 2;
    srv.request.request.ctrl_mode = 3;
    srv.request.request.gear = 4; // D (전진)
    
    if (this->morai_event_cmd_client.call(srv)) {
        ROS_INFO("Successfully changed gear to DRIVE.");
    } else {
        ROS_ERROR("Failed to call gear change service.");
    }
    
    // 2. 상태를 '전역 경로 주행'으로 복귀
    this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
}