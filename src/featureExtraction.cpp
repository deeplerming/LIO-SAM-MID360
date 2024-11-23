#include "ros/console.h"
#include "ros/publisher.h"
#include "sensor_msgs/PointCloud2.h"
#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "pcl/kdtree/kdtree.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;	// 订阅当前激光帧运动畸变校正后的点云信息

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;
	ros::Publisher pubCylinderPoints;
	ros::Publisher pubCylinderNearestPoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
	pcl::PointCloud<PointType>::Ptr cylinderCloud;
	pcl::PointCloud<PointType>::Ptr cylinderNearestCloud;

	pcl::PointCloud<PointType>::Ptr cylinderFilterCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

	// std::ve
    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;			// 1： corner

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
		pubCylinderPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_cylinder", 1);
		pubCylinderNearestPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_cylinder_nearest", 1);

        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
		cylinderCloud.reset(new pcl::PointCloud<PointType>());
		cylinderNearestCloud.reset(new pcl::PointCloud<PointType>());

		// 
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }
	// 计算点的曲率 曲率越大越不平滑 // 这种计算方式合理吗？
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = 
                            cloudInfo.pointRange[i-2]  + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 4
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2];            
			double distance = pointDistance(extractedCloud->points[i]);
            cloudCurvature[i] = diffRange*diffRange/distance;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;		// denote unprocessed point
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i; // 在这一帧所有点中的索引 可是自己本身不就是索引吗？ 会剔除某些点？ pop 操作
        }
    }
	// outlier process  Occluded points  LiLi loam 
    void markOccludedPoints()
    {
		cornerCloud->clear();
        surfaceCloud->clear();
		cylinderCloud->clear();
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
		// int position = 5;
		// int length = 1;
		// int last_continuios_length = 0;
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));
			// int surfthres = 20/depth1;
			// int cylithres = 10/depth1;

            // 两个激光点之间的一维索引差值，如果在一条扫描线上，那么值为1；
            // 如果两个点之间有一些无效点被剔除了，可能会比1大，但不会特别大
            // 如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会很大
			// 提取树干部分和地面部分，树叶部分滤除
            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.08){
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
					// cloudLabel[i] = -1;
					// cloudLabel[i-1] = -1;
					// last_continuios_length = length;
					// length = 0;
                }else if (depth2 - depth1 > 0.08){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
					// cloudLabel[i+1] = -1;
					// cloudLabel[i+2] = -1;
					// last_continuios_length = length;
					// length = 0;
                }
				// else {
				// 	length++;
				// }
            }
			// if (last_continuios_length > surfthres)
			// {
			// 	for (int j = 0; j < surfthres/3; j++)
			// 		surfaceCloud->push_back(extractedCloud->points[i-3*j]);
			// 	last_continuios_length = 0;
			// }
			// if (last_continuios_length > cylithres)
			// {
			// 	for (int j = 0; j < cylithres/3; j++)
			// 		cylinderCloud->push_back(extractedCloud->points[i - 3*j]);
			// 	last_continuios_length = 0;
			// 	// cylinderCloud->push_back(extractedCloud->points[i - 8]);
			// }
	        // parallel beam // 没什么用 ，这个场景不太需要考虑
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.1 * cloudInfo.pointRange[i] && diff2 > 0.1 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();
		cylinderCloud->clear();
		cylinderNearestCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cylinderCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cylinderCloudScanDS(new pcl::PointCloud<PointType>());
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
			// 将一条扫描线扫描一周的点云数据，划分为6段，每段分开提取有限数量的特征，保证特征均匀分布
            for (int j = 0; j < 6; j++)
            {
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;
				// extrat edge features  after sort, start from large smoothness to get edge features can be faster
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 40){
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
					if(cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > cyliThreshold)
					{
						int m;
						double range = pointDistance(extractedCloud->points[ind]);
                        cloudLabel[ind] = 2;
						cloudNeighborPicked[ind] = 1;

						for (m = 1; m <= 1; m++)
						{
							if (std::abs(int(cloudInfo.pointColInd[ind + m] - cloudInfo.pointColInd[ind + m - 1])) > 10 
								|| range > pointDistance(extractedCloud->points[ind + m]))
							{
								break;
							}
							cloudNeighborPicked[ind + m] = 1;
						}
						if (m == 1)
						{
							for (m = -1; m >= -1; m--)
							{
								if (std::abs(int(cloudInfo.pointColInd[ind + m] - cloudInfo.pointColInd[ind + m - 1])) > 10
									|| range > pointDistance(extractedCloud->points[ind + m]))
								{
									break;
								}
								cloudNeighborPicked[ind + m] = 1;
							}	
						}
						if (m == -1)
						{
							cloudLabel[ind] = 3;		// label 3 used to cal cylinder param
							cylinderNearestCloud->push_back(extractedCloud->points[ind]);

							// find nearest in next line ? but only 4 line here...
						}
						// cylinderCloud->push_back(extractedCloud->points[ind]);
					}
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] > 0){
                        cylinderCloudScan->push_back(extractedCloud->points[k]);
                    }
					if (cloudLabel[k] < 0){
						surfaceCloudScan->push_back(extractedCloud->points[k]);
					}
                }
            }

            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            cylinderCloudScanDS->clear();
            downSizeFilter.setInputCloud(cylinderCloudScan);
            downSizeFilter.filter(*cylinderCloudScanDS);			

            *cylinderCloud += *cylinderCloudScanDS;
			*surfaceCloud += *surfaceCloudScanDS;
        }
    }

	void fitCylinderFunc()
	{
		
	}

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
		// publish cylinder features
		cloudInfo.cloud_cylinder = publishCloud(pubCylinderPoints, cylinderCloud, cloudHeader.stamp, lidarFrame);
		sensor_msgs::PointCloud2 cylinder_nearest = publishCloud(pubCylinderNearestPoints, cylinderNearestCloud, cloudHeader.stamp, lidarFrame);
		std::cout << cylinderCloud->size() << "\t" << surfaceCloud->size() << "\t" << cornerCloud->size() << "\t" << cylinderNearestCloud->size() << std::endl;
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}