#include "controller.h"

Controller::Controller(char control_target):
    current_state(VehicleState::GLOBAL_PATH_DRIVING),
    optimal_throttle(), is_finish(false), flag(true),
    is_path_set(false), is_pose_set(false), is_local_path_set(false), g_curr_idx(0),
    velocity(0.), global_velocity(0.), local_velocity(0.), is_moving_back(false), is_parking_complete(false), uturn_done(false)
    {
        this->is_erp = control_target == 'e'; // e => erp (true), m => morai (false)
       
        this->global_path_finish_pub = nh.advertise<std_msgs::Bool>("/global_path_finish", 1);

        this->erp_drive_semi_pub = nh.advertise<race::drive_values>("control_value", 1);
        this->morai_drive_pub = nh.advertise<morai_msgs::CtrlCmd>("/ctrl_cmd", 1);

        this->curr_waypoint_pub = nh.advertise<std_msgs::Int16>("/curr_idx", 1);

        this->g_path_sub = nh.subscribe("/global_path", 1, &Controller::globalPathCallback, this);
        
        this->l_path_sub = nh.subscribe("/local_path", 1, &Controller::localPathCallback, this);
        this->waypoint_done_sub = nh.subscribe("/waypoint_done", 1, &Controller::waypointDoneCallback, this);

        this->parking_path_sub = nh.subscribe("/parking_path", 1, &Controller::parkingPathCallback, this);
        this->parking_path_done_sub = nh.subscribe("/parking_path_done", 1, &Controller::parkingPathDoneCallback, this);
        this->reverse_path_sub = nh.subscribe("/reverse_parking_path", 1, &Controller::reversePathCallback, this);
        this->reverse_path_done_sub = nh.subscribe("/reverse_parking_done", 1, &Controller::reversePathDoneCallback, this);

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

void Controller::waypointDoneCallback(const std_msgs::Int16& signal){
    if(signal.data == 1){
        ROS_INFO_STREAM("[Controller] Waypoint Done signal received. Switching to GLOBAL_PATH_DRIVING.");
        this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
        this->is_local_path_set = false;
        this->uturn_done = true;
        this->l_curr_idx = 0;
    }
}

void Controller::localPathCallback(const erp_drive_semi::PathReference& l_path){
    if(this->g_curr_idx < 1080) return;

    if(this->start_yaw == -400)
        this->start_yaw = static_cast<int>(this->curr_yaw);

    this->l_path = l_path;
    this->l_path_size = l_path.path.size();
    this->is_local_path_set = true;
    this->current_state = VehicleState::LOCAL_PATH_DRIVING;
    // if(this->g_curr_idx < 1500){
    //     this->current_state = VehicleState::LOCAL_PATH_DRIVING;
    // } else {
    //     this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
    // }
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
        // msg.throttle = this->velocity; // 6.; // 장애물 회피
        
        if (this->is_moving_back) {
            msg.throttle = -this->velocity; // 후진
        } else {
            msg.throttle = this->velocity * -1; // 전진
        }
        // 주의! deg 단위임
        msg.steering = -this->steering * (180 / M_PI);
        // if(msg.throttle < 0)
        //     msg.steering *= -1;

        if (this->is_moving_back) {
            msg.steering = this->steering * -1;
        } else {
            msg.steering = this->steering;
        }

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
    if(this->g_curr_idx < 110 or this->g_curr_idx > 230) return;

    this->l_path = parking_path;
    this->l_path_size = parking_path.path.size();

    this->is_local_path_set = true;
    this->is_parking_complete = false; 
    this->l_curr_idx = 0; 

    this->current_state = VehicleState::PARKING_PATH_DRIVING;
}

void Controller::parkingPathDoneCallback(const std_msgs::Int16& signal){
    if(signal.data == 1){
        this->is_local_path_set = false;

        this->current_state = VehicleState::PARKING_DONE_STOPPING;

        this->is_parking_complete = true; 
        this->l_curr_idx = 0;

        this->stop_timer = nh.createTimer(ros::Duration(3.0), &Controller::stopTimerCallback, this, true);
    }
}

void Controller::stopTimerCallback(const ros::TimerEvent&) {
    
    if(this->reverse_path_size > 0) {
        this->l_path = this->reverse_path;
        this->l_path_size = this->reverse_path_size;

        this->l_curr_idx = 0;
        this->is_local_path_set = true;

        this->current_state = VehicleState::PARKING_REVERSING;
    } 
}

void Controller::reversePathCallback(const erp_drive_semi::PathReference& reverse_path){
    this->reverse_path = reverse_path;
    this->reverse_path_size = reverse_path.path.size();
}

void Controller::reversePathDoneCallback(const std_msgs::Int16& signal){
    if(signal.data == 1){        
        this->is_local_path_set = false;

        this->current_state = VehicleState::REVERSE_DONE_STOPPING;
        
        this->reverse_done_timer = nh.createTimer(ros::Duration(1.0), &Controller::reverseDoneTimerCallback, this, true);
    }
}

void Controller::reverseDoneTimerCallback(const ros::TimerEvent&) {
    this->current_state = VehicleState::GLOBAL_PATH_DRIVING;
}
