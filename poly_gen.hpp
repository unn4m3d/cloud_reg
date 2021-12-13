#pragma once
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vtkSmartPointer.h>
#include <vtkPlaneSource.h>

namespace clouds
{
    extern Eigen::AlignedBox<float, 3> computeAlignedBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr&);

    extern vtkSmartPointer<vtkPolyData> generatePlane(const Eigen::Vector4f& normal, const Eigen::AlignedBox<float, 3>& bounding_box);

    extern Eigen::Vector4f computeCoeffs(const std::vector<pcl::PointXYZ>& three_pts);
}