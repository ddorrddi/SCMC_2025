#ifndef DECISION_H
#define DECISION_H

// ros 관련 헤더
#include <ros/ros.h>

// 좌표계 및 기하 연산 관련 헤더
#include <cmath>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <std_msgs/Int16.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>

// Path 관련 커스텀 메세지
#include <erp_drive/PathReference.h>
#include <erp_drive/PathReferenceElement.h>

// 제어 관련 메세지
#include <race/drive_values.h>
#include <morai_msgs/CtrlCmd.h>

// velocity 등 parameter 서버를 위한 헤더 
#include <dynamic_reconfigure/server.h>
#include <erp_drive/CtrlParamConfig.h>

#include <string>

#include "optimal_throttle.h"
#include "erp_drive/InitializePose.h"
#include "erp_drive/PermitParking.h"

#include "timer.h"

class Controller {
public:
    Controller(char control_target);

    // 상수 값
    const int LOOP_RATE = 30;
    const double WHEEL_BASE = 1.04;

    // 속도 | rosparam 으로 관리될 값
    double velocity;
    double global_velocity;
    double local_velocity;
    double parking_velocity;

    OptimalThrottle optimal_throttle;
    void flowControl(const geometry_msgs::Pose& curr_pose);

    Timer timer;

    double speed_now;
    ros::Subscriber speed_sub;
    void speedCallback(const std_msgs::Float64& speed_now);

    // 조향각
    double steering, brake;

    // ERP에 대한 제어를 실행하는지 아닌지
    // global path가 세팅되었는지 아닌지
    // pose가 세팅되었는지 아닌지
    // local path가 세팅되었는지 아닌지
    bool is_erp, is_path_set, is_pose_set, is_local_path_set;
    bool is_delivery_path_set, is_finish;

    // Publisher 및 Subscriber 등록용
    ros::NodeHandle nh;

    // ERP 주행용, 모라이 주행용 Publisher
    ros::Publisher erp_drive_pub;
    ros::Publisher morai_drive_pub;
    void publishControlMsg();

    // planner 에서 활용 하기 위한 현재 차량 인덱스 Publisher
    ros::Publisher curr_idx_pub;

    // global path 및 Subscriber
    erp_drive::PathReference g_path;
    int g_path_size, g_curr_idx;
    ros::Subscriber g_path_sub;
    void globalPathCallback(const erp_drive::PathReference& g_path);

    // local path 및 Subscriber
    erp_drive::PathReference l_path;
    int l_path_size, l_curr_idx;
    ros::Subscriber l_path_sub;
    ros::Subscriber l_path_done_sub;
    void localPathCallback(const erp_drive::PathReference& l_path);
    void localPathDoneCallback(const std_msgs::Int16& signal);

    erp_drive::PathReference delivery_path;
    int delivery_path_size, delivery_curr_idx;
    bool delivery_path_done = false;
    ros::Subscriber delivery_path_sub;
    void deliveryPathCallback(const erp_drive::PathReference& delivery_path);

    // @FMTC
    int parking_start_idx[3] = {
        // 120, 133, 147
        120, // 3466,
        133, // 3479,
        147 // 3496
    };

    int parking_space_num;
    ros::Subscriber parking_space_num_sub;
    void parkingSpaceNumCallback(const std_msgs::Int16& parking_space_num);

    bool is_moving_front;

    int start_yaw = -400;
    double curr_yaw;
    sensor_msgs::Imu imu;
    ros::Subscriber imu_sub;
    void imuCallback(const sensor_msgs::Imu& imu_data);

    bool stop_flag = true;
    // 현재 차량의 pose 및 Subscriber
    geometry_msgs::Pose curr_pose;
    ros::Subscriber curr_pose_sub;
    void currentPoseCallback(const geometry_msgs::Pose& curr_pose);

    int calcGlobalCurrWaypoint(const geometry_msgs::Pose& curr_pose);
    int calcLocalCurrWaypoint(const geometry_msgs::Pose& curr_pose);
    int calcDeliveryCurrWaypoint(const geometry_msgs::Pose& curr_pose);

    ros::Publisher curr_waypoint_pub;
    ros::Publisher curr_point_pub;

    std::string traffic_light_status;
    ros::Subscriber traffic_light_status_sub;
    void trafficLightStatusCallback(const std_msgs::String& traffic_light_status);
    bool isLeftSign();
    bool isGreenSign();
    bool isLeftTurn();

    ros::ServiceClient client;
    ros::ServiceClient client_for_parking_path;
    erp_drive::InitializePose srv;
    erp_drive::PermitParking srv_parking_path;
     
    // geometry_msg::Pose -> tf::Vector3 로 변환
    tf::Vector3 pose2vec(const geometry_msgs::Pose& pose) {
        return tf::Vector3(pose.position.z, pose.position.y, pose.position.z);
    }

    // geometry_msg::Point -> tf::Vector3 로 변환
    tf::Vector3 pt2vec(const geometry_msgs::Point& pt) {
        return tf::Vector3(pt.x, pt.y, pt.z);
    }

    // index 기반 거리 구하기
    double getDistance(const int& idx1, const int& idx2) {
        return getDistance(this->g_path.path[idx1].pose, this->g_path.path[idx2].pose);
    }

    // geometry_msgs::Pose 버전 거리 구하기
    double getDistance(const geometry_msgs::Pose& a, const geometry_msgs::Pose& b) {
        double dist = (a.position.x - b.position.x) * (a.position.x - b.position.x);
        dist += (a.position.y - b.position.y) * (a.position.y - b.position.y);
        dist = sqrt(dist);
        
        return dist;
    }

    // geometry_msgs::Point 버전 거리 구하기 
    double getDistance(const geometry_msgs::Point& a, const geometry_msgs::Point& b) {
        return tf::tfDistance(pt2vec(a), pt2vec(b));
    }

    void quaternionToEuler(const geometry_msgs::Quaternion& quat, double& roll, double& pitch, double& yaw) const {
        // geometry_msgs::Quaternion을 tf::Quaternion으로 변환
        tf::Quaternion tf_quat(quat.x, quat.y, quat.z, quat.w);
        
        // tf::Quaternion을 tf::Matrix3x3으로 변환
        tf::Matrix3x3 m(tf_quat);
        
        // 행렬에서 오일러 각을 추출
        m.getRPY(roll, pitch, yaw);
    }
};

#endif
