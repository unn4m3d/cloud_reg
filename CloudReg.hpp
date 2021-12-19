#pragma once
#include <filesystem>
#include <pcl/point_types.h>
#include <pcl/console/time.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include "InputManager.hpp"
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/concave_hull.h>

namespace clouds
{
    class CloudReg
    {
    public:
        using PCPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
        using PCColPtr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;
        using PCVisPtr = pcl::visualization::PCLVisualizer::Ptr;
        using PCColHandler = pcl::visualization::PointCloudColorHandler<pcl::PointXYZ>;
        using PCNormalPtr = pcl::PointCloud<pcl::Normal>::Ptr;

        CloudReg();

        void run(int argc, char** argv);
        PCPtr loadFile(const std::filesystem::path& file);
        void openViewer();
        void addCloud(const PCPtr&, const std::string&);
        void addCloud(const PCColPtr&, const std::string&);
        void addCloud(const PCPtr&, const PCColHandler&, const std::string&);

        void downsample(const PCPtr& input, PCPtr& output, float leaf_size);
        void estimateNormals(const PCPtr& input, PCNormalPtr& output);
        PCColPtr regionGrowing(const PCPtr& input, const PCNormalPtr& normals, std::vector<pcl::PointIndices>&);
        std::vector<Eigen::Vector3f> centersOfMass(const PCPtr&, const std::vector<pcl::PointIndices>&);
        std::vector<pcl::ConvexHull<pcl::PointXYZ>> calculateHulls(const PCPtr&, std::vector<pcl::PointIndices>&);

        void yield();
    private:
        InputManager input;
        PCVisPtr viewer;
        PCPtr cloud, downsampled, transformed;
        pcl::console::TicToc timer;
        pcl::search::Search<pcl::PointXYZ>::Ptr search_tree;
    };
}