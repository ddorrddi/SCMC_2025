// 필요한 헤더 파일 포함
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/MarkerArray.h>
#include <vector>

// Message
#include <lidar_trafficcone_detection/Trafficcone.h>
#include "header.h"
#include "dbscan.h"

#include <visualization_msgs/Marker.h>

using namespace std;

// 전역 변수 선언 및 초기화
int cluster_id = 0;
float center_x = 0.0, center_y = 0.0, center_z = 0.0;
pcl::PointXYZI minPoint, maxPoint;
pcl::PointCloud<pcl::PointXYZI> totalcloud_clustered;

// msg
lidar_trafficcone_detection::Trafficcone Trafficcone;


int minPoints;
double epsilon; 

int minClusterSize; 
int maxClusterSize; 

double xMinROI, xMaxROI, yMinROI, yMaxROI, zMinROI, zMaxROI; // ROI(PassThrough) 범위 지정 변수
double xMinBoundingBox, xMaxBoundingBox, yMinBoundingBox, yMaxBoundingBox, zMinBoundingBox, zMaxBoundingBox; // init bounding box

typedef pcl::PointXYZ PointT;

vector<float> obstacle;
vector< vector<float> > obstacle_vec;

ros::Publisher pubTrafficcone;
ros::Publisher markerPub;
ros::Publisher roiPub;       //ROI Publisher
ros::Publisher clusterPub;   //Cluster Publisher


void dynamicParamCallback(lidar_object_detector::traffic_hyper_parameterConfig &config, int32_t level) {
    xMinROI = config.traffic_xMinROI;
    xMaxROI = config.traffic_xMaxROI;
    yMinROI = config.traffic_yMinROI;
    yMaxROI = config.traffic_yMaxROI;
    zMinROI = config.traffic_zMinROI;
    zMaxROI = config.traffic_zMaxROI;

    minPoints = config.traffic_minPoints;
    epsilon = config.traffic_epsilon;
    minClusterSize = config.traffic_minClusterSize;
    maxClusterSize = config.traffic_maxClusterSize;

  //ROI를 이용하여 걸러내진 못한 물체들을 BoundingBOX변수를 이용하여 한번 더 filtering 할 수 있음.
    xMinBoundingBox = config.traffic_xMinBoundingBox;
    xMaxBoundingBox = config.traffic_xMaxBoundingBox;
    yMinBoundingBox = config.traffic_yMinBoundingBox;
    yMaxBoundingBox = config.traffic_yMaxBoundingBox;
    zMinBoundingBox = config.traffic_zMinBoundingBox;
    zMaxBoundingBox = config.traffic_zMaxBoundingBox; // BoundingBox 크기 범위 지정 변수
}

// Point cloud callback 함수
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& inputcloud) {
    // ROS message를 PCL 포인트 클라우드로 변환
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*inputcloud, *cloud);

    lidar_trafficcone_detection::Trafficcone ObjectInfoMsg;
    visualization_msgs::MarkerArray markerArray; 
    visualization_msgs::Marker marker;

    // Region of Interest (ROI) 필터링
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyzf(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> xfilter;
    xfilter.setInputCloud(cloud);
    xfilter.setFilterFieldName("x");
    xfilter.setFilterLimits(xMinROI, xMaxROI);
    xfilter.setFilterLimitsNegative(false);
    xfilter.filter(*cloud_xyzf);

    pcl::PassThrough<pcl::PointXYZ> yfilter;
    yfilter.setInputCloud(cloud_xyzf);
    yfilter.setFilterFieldName("y");
    yfilter.setFilterLimits(yMinROI, yMaxROI);
    yfilter.setFilterLimitsNegative(false);
    yfilter.filter(*cloud_xyzf);

    pcl::PassThrough<pcl::PointXYZ> zfilter;
    zfilter.setInputCloud(cloud_xyzf);
    zfilter.setFilterFieldName("z");
    zfilter.setFilterLimits(zMinROI, zMaxROI);
    zfilter.setFilterLimitsNegative(false);
    zfilter.filter(*cloud_xyzf);

    // KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    if (cloud_xyzf->size() > 0) {
        tree->setInputCloud(cloud_xyzf);
    }

    // Segmentation
    vector<pcl::PointIndices> cluster_indices;
    DBSCANKdtreeCluster<pcl::PointXYZ> dc;
    dc.setCorePointMinPts(minPoints);
    dc.setClusterTolerance(epsilon);
    dc.setMinClusterSize(minClusterSize);
    dc.setMaxClusterSize(maxClusterSize);
    dc.setSearchMethod(tree);
    dc.setInputCloud(cloud_xyzf);
    dc.extract(cluster_indices);

    vector<float> left_x;
    vector<float> left_y;

    int cluster_id = 0; // 초기화!
    
    for (const auto& cluster_index : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZI> eachcloud_clustered;
        float cluster_counts = static_cast<float>(cluster_indices.size());
        
        // 각 클러스터 내 각 포인트 접근
        for (const auto& point_index : cluster_index.indices) {
            pcl::PointXYZI point;
            point.x = cloud_xyzf->points[point_index].x;
            point.y = cloud_xyzf->points[point_index].y;
            point.z = cloud_xyzf->points[point_index].z;
            point.intensity = cluster_id % 100;
            eachcloud_clustered.push_back(point);
        }

        pcl::PointXYZI minPoint, maxPoint;
        pcl::getMinMax3D(eachcloud_clustered, minPoint, maxPoint);

        float x_len = abs(maxPoint.x - minPoint.x);   //직육면체 x 모서리 크기
        float y_len = abs(maxPoint.y - minPoint.y);   //직육면체 y 모서리 크기
        float z_len = abs(maxPoint.z - minPoint.z);   //직육면체 z 모서리 크기 
        float volume = x_len * y_len * z_len;         //직육면체 부피

        float center_x = (minPoint.x + maxPoint.x)/2; //직육면체 중심 x 좌표
        float center_y = (minPoint.y + maxPoint.y)/2; //직육면체 중심 y 좌표
        float center_z = (minPoint.z + maxPoint.z)/2; //직육면체 중심 z 좌표 

        cout << center_x <<  ' '  << center_y << ' ' <<center_z << "!!!!!" << endl;

        ObjectInfoMsg.lengthX[cluster_id] = x_len;
        ObjectInfoMsg.lengthY[cluster_id] = y_len;
        ObjectInfoMsg.lengthZ[cluster_id] = z_len;
        ObjectInfoMsg.centerX[cluster_id] = center_x;
        ObjectInfoMsg.centerY[cluster_id] = center_y;
        ObjectInfoMsg.centerZ[cluster_id] = center_z;

        // 여기에 Marker 생성 코드 추가        
        marker.header.frame_id = "velodyne";
        marker.header.stamp = ros::Time::now();
        marker.ns = "traffic_cone";
        marker.id = cluster_id;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        center_x = ObjectInfoMsg.centerX[cluster_id];
        center_y = ObjectInfoMsg.centerY[cluster_id];
        center_z = ObjectInfoMsg.centerZ[cluster_id];

        // 위치 설정
        marker.pose.position.x = ObjectInfoMsg.centerX[cluster_id];
        marker.pose.position.y = ObjectInfoMsg.centerY[cluster_id];
        marker.pose.position.z = ObjectInfoMsg.centerZ[cluster_id];

        // 방향 설정 (회전 없음)
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        // 크기 설정
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;

        // 색상 설정 (예: 빨간색)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;  // 반투명

        // 라이다 센서의 위치는 0,0,0이기 때문에 이를 바탕으로 거리정보를 구할 수 있음
        float distance = sqrt(center_x * center_x + center_y * center_y); //장애물 <-> 차량 거리
        
        if ( (xMinBoundingBox < x_len && x_len < xMaxBoundingBox) && (yMinBoundingBox < y_len && y_len < yMaxBoundingBox) && (zMinBoundingBox < z_len && z_len < zMaxBoundingBox) ) {
            marker.header.frame_id = "velodyne";
            marker.header.stamp = ros::Time();
            marker.ns = std::to_string(cluster_counts); // Convert float to string
            marker.id = cluster_id; 
            marker.type = visualization_msgs::Marker::CUBE; // Already set earlier, but no harm in setting again
            marker.action = visualization_msgs::Marker::ADD;
            
            // Each coordinate's center
            marker.pose.position.x = center_x; 
            marker.pose.position.y = center_y;
            marker.pose.position.z = center_z;

            left_x.push_back(center_x);
            left_y.push_back(center_y);

            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            
            // Length 
            marker.scale.x = x_len;
            marker.scale.y = y_len;
            marker.scale.z = z_len;
            
            // Set bounding box color
            marker.color.a = 0.5; // Transparency
            marker.color.r = 0.0; // RGB values
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            
            // Set marker lifetime and add to markerArray
            marker.lifetime = ros::Duration(0.1); 
            markerArray.markers.emplace_back(marker);
        }

        cluster_id++; //intensity 증가

        // Convert to ROS data type
        pcl::PCLPointCloud2 cloud_p;
        pcl::toPCLPointCloud2(eachcloud_clustered, cloud_p);

        sensor_msgs::PointCloud2 cluster;
        pcl_conversions::fromPCL(cloud_p, cluster);
        cluster.header.frame_id = "velodyne";

        clusterPub.publish(cluster);
    }

    ObjectInfoMsg.objectCounts = cluster_id;
    pubTrafficcone.publish(ObjectInfoMsg);

    markerPub.publish(markerArray);

    pcl::PCLPointCloud2 cloud_cropbox;
    pcl::toPCLPointCloud2(*cloud_xyzf, cloud_cropbox);

    sensor_msgs::PointCloud2 ROI;
    pcl_conversions::fromPCL(cloud_cropbox, ROI);
    ROI.header.frame_id = "velodyne";

    roiPub.publish(ROI);

}

int main(int argc, char **argv) {
    ros::init(argc, argv, "traffic_cone_detector");
    ros::NodeHandle nh;
    
    pubTrafficcone = nh.advertise<lidar_trafficcone_detection::Trafficcone>("/traffic_cone", 1);
    roiPub = nh.advertise<sensor_msgs::PointCloud2>("/roi", 1);
    clusterPub = nh.advertise<sensor_msgs::PointCloud2>("/cluster", 1);
    markerPub = nh.advertise<visualization_msgs::MarkerArray>("/traffic_cone_markers", 1);

    dynamic_reconfigure::Server<lidar_object_detector::traffic_hyper_parameterConfig> server;
    dynamic_reconfigure::Server<lidar_object_detector::traffic_hyper_parameterConfig>::CallbackType f;

    f = boost::bind(&dynamicParamCallback, _1, _2);
    server.setCallback(f);

    ros::Subscriber rawDataSub = nh.subscribe("/velodyne_points", 1, cloud_cb);

    ros::spin();
 
}