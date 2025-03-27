#include "header.h"  //header.h 참조
#include "dbscan.h"  // dbscan.h 참조

using namespace std;

/*dbscan clustering 사용을 위한 변수*/
int minPoints = 10; //10          //Core Point 기준 필요 인접점 최소 개수
double epsilon = 0.3; //0.3        //Core Point 기준 주변 탐색 반경 
// double epsilon_big = 0.3; //0.3        //Core Point 기준 주변 탐색 반경 

//clustering 사이즈 설정해주어서 사이즈에 해당하는 clustring만 보여줌 - clustering 사이즈에 따라서 인식할 수 있는 물체가 달라짐 
// min clustering과 maxclustering size를 이용하여 미션시 라바콘과 실제 차량의 차이를 구별 할 수 있지 않을까 생각됨.
int minClusterSize = 100; //10     //Cluster 최소 사이즈
int maxClusterSize = 10000; //10000  //Cluster 최대 사이즈

//roi 값을 바꿔주면 관심 범위가 달라짐
// ex) xmin,ymin,zmin을 모두 0보다 크게 주면, 범위가 양수의 범위에만 bounding box가 쳐짐
//ROI를 이용하여 걸러내진 못한 물체들을 BoundingBOX변수를 이용하여 한번 더 filtering 할 수 있음.

double xMinROI, xMaxROI, yMinROI, yMaxROI, zMinROI, zMaxROI; // ROI(PassThrough) 범위 지정 변수
double xMinBoundingBox, xMaxBoundingBox, yMinBoundingBox, yMaxBoundingBox, zMinBoundingBox, zMaxBoundingBox; // init bounding box

typedef pcl::PointXYZ PointT;

ros::Publisher roiPub; //ROI Publishser
ros::Publisher clusterPub; //Cluster Publishser
ros::Publisher boundingBoxPub; //Bounding Box Visualization Publisher
ros::Publisher obstaclePub; // Obstacle publisher

void dynamicParamCallback(static_obstacle_detection::static_hyper_parameterConfig &config, int32_t level) {
    xMinROI = config.static_xMinROI;
    xMaxROI = config.static_xMaxROI;
    yMinROI = config.static_yMinROI;
    yMaxROI = config.static_yMaxROI;
    zMinROI = config.static_zMinROI;
    zMaxROI = config.static_zMaxROI;

    minPoints = config.static_minPoints;
    minClusterSize = config.static_minClusterSize;
    maxClusterSize = config.static_maxClusterSize;

  //ROI를 이용하여 걸러내진 못한 물체들을 BoundingBOX변수를 이용하여 한번 더 filtering 할 수 있음.
    epsilon = config.static_epsilon;
    xMinBoundingBox = config.static_xMinBoundingBox;
    xMaxBoundingBox = config.static_xMaxBoundingBox;
    yMinBoundingBox = config.static_yMinBoundingBox;
    yMaxBoundingBox = config.static_yMaxBoundingBox;
    zMinBoundingBox = config.static_zMinBoundingBox;
    zMaxBoundingBox = config.static_zMaxBoundingBox; // BoundingBox 크기 범위 지정 변수
}

//point cloud callback 함수 선언
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& inputcloud) {
    //ROS message 변환
    //PointXYZI가 아닌 PointXYZ로 선언하는 이유 -> 각각의 Cluster를 다른 색으로 표현해주기 위해서. Clustering 이후 각각 구별되는 intensity value를 넣어줄 예정.
    //PointXYZI는 intensity정보를 가지고 있고, 이것은 반사강도를 나타내는데 크게 중요하지 않기 때문에 PointXYZ를 사용함.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);//스마트 포인터를 선언하고, 이 포인터를 통해 새로운 Point Cloud를 생성
    pcl::fromROSMsg(*inputcloud, *cloud); // ROS에서 받아온 Point Cloud 메시지를 PCL의 Point Cloud 형식으로 변환
    
    static_obstacle_detection::ObstacleInfo ObstacleInfoMsg;
    
    pcl::PointCloud<pcl::PointXYZI> totalcloud_clustered; // init
    int cluster_id = 0;
    int obstacle_id = 0;

    //Visualizing에 필요한 Marker 선언
    visualization_msgs::Marker boundingBox;  // xyz를 boundingbox 처리
    visualization_msgs::MarkerArray boundingBoxArray; // boundingbox들을 array로 만들어서 한번에 여러개의 boundingbox를 보여줌 

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xf(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> xfilter;
    xfilter.setInputCloud(cloud);
    xfilter.setFilterFieldName("x");
    xfilter.setFilterLimits(xMinROI, xMaxROI); 
    xfilter.setFilterLimitsNegative(false);
    xfilter.filter(*cloud_xf);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyf(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> yfilter;
    yfilter.setInputCloud(cloud_xf);
    yfilter.setFilterFieldName("y");
    yfilter.setFilterLimits(yMinROI, yMaxROI);
    yfilter.setFilterLimitsNegative(false);
    yfilter.filter(*cloud_xyf);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyzf(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> zfilter;
    zfilter.setInputCloud(cloud_xyf);
    zfilter.setFilterFieldName("z");
    zfilter.setFilterLimits(zMinROI, zMaxROI); // -0.62, 0.0
    zfilter.setFilterLimitsNegative(false);
    zfilter.filter(*cloud_xyzf);

    // //Voxel Grid를 이용한 DownSampling
    // pcl::VoxelGrid<pcl::PointXYZ> vg;    // VoxelGrid 선언
    // vg.setInputCloud(cloud_xyzf);             // Raw Data 입력
    // vg.setLeafSize(1.0f, 1.0f, 1.0f); // 사이즈를 너무 작게 하면 샘플링 에러 발생
    // vg.filter(*cloud_xyzf);          // Filtering 된 Data를 cloud PointCloud에 삽입

    //KD-Tree
    //이 포인터를 통해 KD-Tree 검색 방법을 사용할 수 있도록 하는 객체를 생성합니다
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    //Point Cloud(cloud_xyzf)가 비어있지 않은 경우에 한해 KD-Tree 검색 방법에서 사용할 입력 Point Cloud를 설정
    if (cloud_xyzf->size() > 0) {
        tree->setInputCloud(cloud_xyzf);
    }
    
    //Segmentation
    //클러스터의 index 정보를 저장할 vector를 선언합니다. 
    vector<pcl::PointIndices> cluster_indices;
    DBSCANKdtreeCluster<pcl::PointXYZ> dc;
    dc.setCorePointMinPts(minPoints);   //Set minimum number of neighbor points
    dc.setClusterTolerance(epsilon); //Set Epsilon 
    dc.setMinClusterSize(minClusterSize);
    dc.setMaxClusterSize(maxClusterSize);
    dc.setSearchMethod(tree); // kdtree
    dc.setInputCloud(cloud_xyzf); // point_cloud xyz
    dc.extract(cluster_indices);

    for(int i = 0; i < 1; i++){
      for(vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
        pcl::PointCloud<pcl::PointXYZI> eachcloud_clustered; // 클러스터 하나의 point cloud
        float cluster_counts = cluster_indices.size();

        //각 Cluster내 각 Point 접근
        for(vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            pcl::PointXYZI tmp;
            tmp.x = cloud_xyzf->points[*pit].x;
            tmp.y = cloud_xyzf->points[*pit].y;
            tmp.z = cloud_xyzf->points[*pit].z;
            tmp.intensity = cluster_id % 100; // 클러스터별로 색을 다르게 하기 위해 tmp.intensity 값을 cluster_id % 8로 설정
            eachcloud_clustered.push_back(tmp); // 클러스터 하나의 point cloud
            totalcloud_clustered.push_back(tmp); // 모든 클러스터의 point cloud
        }

        //minPoint와 maxPoint 받아오기
        // 각 클러스터의 bounding box의 크기를 계산하기 위해, pcl::getMinMax3D 함수를 통해 eachcloud_clustered의 최소값과 최대값을 구합니다.
        pcl::PointXYZI minPoint, maxPoint;
        pcl::getMinMax3D(eachcloud_clustered, minPoint, maxPoint);

        float x_len = abs(maxPoint.x - minPoint.x);   //직육면체 x 모서리 크기
        float y_len = abs(maxPoint.y - minPoint.y);   //직육면체 y 모서리 크기
        float z_len = abs(maxPoint.z - minPoint.z);   //직육면체 z 모서리 크기 
        float volume = x_len * y_len * z_len;         //직육면체 부피

        float center_x = (minPoint.x + maxPoint.x)/2; //직육면체 중심 x 좌표
        float center_y = (minPoint.y + maxPoint.y)/2; //직육면체 중심 y 좌표
        float center_z = (minPoint.z + maxPoint.z)/2; //직육면체 중심 z 좌표 


        // 메시지 값 할당

        // 라이다 센서의 위치는 0,0,0이기 때문에 이를 바탕으로 거리정보를 구할 수 있음
        float distance = sqrt(center_x * center_x + center_y * center_y); //장애물 <-> 차량 거리
        
        if ( (xMinBoundingBox < x_len && x_len < xMaxBoundingBox) && (yMinBoundingBox < y_len && y_len < yMaxBoundingBox) && (zMinBoundingBox < z_len && z_len < zMaxBoundingBox) ) {
            ObstacleInfoMsg.lengthX[obstacle_id] = x_len;
            ObstacleInfoMsg.lengthY[obstacle_id] = y_len;
            ObstacleInfoMsg.lengthZ[obstacle_id] = z_len;
            ObstacleInfoMsg.centerX[obstacle_id] = center_x;
            ObstacleInfoMsg.centerY[obstacle_id] = center_y;
            ObstacleInfoMsg.centerZ[obstacle_id] = center_z;
            
            boundingBox.header.frame_id = "velodyne";
            boundingBox.header.stamp = ros::Time();
            boundingBox.ns = cluster_counts; //ns = namespace
            boundingBox.id = obstacle_id; 
            boundingBox.type = visualization_msgs::Marker::CUBE; //직육면체로 표시
            boundingBox.action = visualization_msgs::Marker::ADD;
            
            // 각 좌표의 중심
            boundingBox.pose.position.x = center_x; 
            boundingBox.pose.position.y = center_y;
            boundingBox.pose.position.z = center_z;
            
            boundingBox.pose.orientation.x = 0.0;
            boundingBox.pose.orientation.y = 0.0;
            boundingBox.pose.orientation.z = 0.0;
            boundingBox.pose.orientation.w = 1.0; // w 값이 1.0이면, 바운딩 박스의 회전이 없음을 의미. 
            //즉, bounding box가 원점에 대해 정렬되어 있어서 회전하지 않은 상태입니다. 이는 bounding box의 모든 면이 축에 평행하게 정렬되어 있을 때 발생할 수 있습니다.
            
            // 길이 
            boundingBox.scale.x = x_len;
            boundingBox.scale.y = y_len;
            boundingBox.scale.z = z_len;
            
            // bounding box 색깔 설정
            boundingBox.color.a = 0.5; //직육면체 투명도, a = alpha
            boundingBox.color.r = 1.0; //직육면체 색상 RGB값
            boundingBox.color.g = 1.0;
            boundingBox.color.b = 1.0;
            
            // boundingbox 지속 시간 및 boundingBoxArray에 boundingbox추가
            // boundingbox지속시간을 0.1로 준 이유는 cluster한개만 보이는 것은 필요없기 때문에 빠르게 사라지게 하기 위해서 시간을 짧게 설정
            // emplace_back은 push_back과 같속
            boundingBox.lifetime = ros::Duration(0.1); //box 지속시간
            boundingBoxArray.markers.emplace_back(boundingBox);
            obstacle_id++;
        }
        cluster_id++;
      }
    }

    // if(cluster_id > 0) cluster_id--;

//Convert To ROS data type
  pcl::PCLPointCloud2 cloud_p;
  pcl::toPCLPointCloud2(totalcloud_clustered, cloud_p);

  sensor_msgs::PointCloud2 cluster;
  pcl_conversions::fromPCL(cloud_p, cluster);
  cluster.header.frame_id = "velodyne";


  pcl::PCLPointCloud2 cloud_cropbox;
  pcl::toPCLPointCloud2(*cloud_xyzf, cloud_cropbox);

  sensor_msgs::PointCloud2 ROI;
  pcl_conversions::fromPCL(cloud_cropbox, ROI);
  ROI.header.frame_id = "velodyne";

  roiPub.publish(ROI);
  clusterPub.publish(cluster);
  boundingBoxPub.publish(boundingBoxArray);

  ObstacleInfoMsg.obstacleCounts = obstacle_id;
  obstaclePub.publish(ObstacleInfoMsg);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "bounding_box");
  ros::NodeHandle nh;

  ros::Subscriber rawDataSub = nh.subscribe("/velodyne_points", 1, cloud_cb);  // velodyne_points 토픽 구독. velodyne_points = 라이다 raw data

  roiPub = nh.advertise<sensor_msgs::PointCloud2>("/roi", 0.001); 
  clusterPub = nh.advertise<sensor_msgs::PointCloud2>("/cluster", 0.001);                  
  boundingBoxPub = nh.advertise<visualization_msgs::MarkerArray>("/boundingBox", 0.001); 
  obstaclePub = nh.advertise<static_obstacle_detection::ObstacleInfo>("/obstacle_info", 0.001);

  dynamic_reconfigure::Server<static_obstacle_detection::static_hyper_parameterConfig> server;
  dynamic_reconfigure::Server<static_obstacle_detection::static_hyper_parameterConfig>::CallbackType f;

  f = boost::bind(&dynamicParamCallback, _1, _2);
  server.setCallback(f);

  ros::spin();

  return 0;
}