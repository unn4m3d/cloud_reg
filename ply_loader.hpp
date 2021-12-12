#pragma once
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace clouds
{
    typedef pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud_ptr;
    /* PLY loader for PhoXi*/
    class ply_loader
    {
    public:
        ply_loader(xyz_cloud_ptr ptr) : cloud(ptr){}

        void load(const std::string& filename);
    private:
        xyz_cloud_ptr cloud;
    };
}