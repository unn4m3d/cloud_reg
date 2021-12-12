#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <thread>
#include <filesystem>
#include "ply_loader.hpp"

using namespace std::chrono_literals;

int main (int argc, char** argv)
{
    std::string filename = "test_pcd.pcd";
    if(argc >= 1)
    {
        filename = argv[1];
    }

    using path = std::filesystem::path;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    auto extstr = path(filename).extension().string();
    if(extstr == ".pcd")
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZ> ("test_pcd.pcd", *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            return (-1);
        }
    }
    else if(extstr == ".ply")
    {
        try
        {
            clouds::ply_loader ldr(cloud);
            ldr.load(filename);
        } 
        catch(...)
        {
            std::cerr << "oops" << std::endl;
        }
    }
    else
    {
        std::cerr << "Unknown extension " << extstr << std::endl;
        return -1;
    }
    std::cout << "Loaded "
                << cloud->width * cloud->height
                << " data points from test_pcd.pcd with the following fields: "
                << std::endl;

    std::cout << "Downsampling..." << std::endl;

    pcl::VoxelGrid<pcl::PointXYZ> filter;
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);

    filter.setInputCloud(cloud);
    filter.setLeafSize(5,5,5);
    filter.filter(*downsampled);

    std::cout << "Downsample to " << downsampled->width*downsampled->height << "points" << std::endl;



    std::cout << "Opening viewer..." << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (downsampled, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

     while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);
            std::this_thread::sleep_for(100ms);
        }

    return (0);
}