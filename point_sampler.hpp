#pragma once
#include <pcl/point_types.h>

#include <pcl/point_cloud.h>
#include <Eigen/Core>

namespace stl_reader
{
    template<typename, typename>
    class StlMesh;
}

namespace clouds
{
    extern double deviate(double value);
    extern Eigen::Vector3f sampleTrianglePoint(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c, float r1, float r2);

    // TODO : Use samples number per unit of area
    extern pcl::PointCloud<pcl::PointXYZLNormal>::Ptr sampleMesh(const stl_reader::StlMesh<float, unsigned>& mesh, float samples);
}