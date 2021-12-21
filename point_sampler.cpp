#include "point_sampler.hpp"
#include "stl_reader.hpp"
#include <random>


namespace clouds
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);

    double deviate(double value)
    {
        return dis(gen) * value;
    }

    Eigen::Vector3f sampleTrianglePoint(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c, float r1, float r2)
    {
        float r1sqr = std::sqrt(r1);
        float one_min_r1sqr = 1 - r1sqr;

        auto a_m = a * one_min_r1sqr;
        auto b_m = b * (1 - r2);

        return r1sqr * (r2 * c + b_m) + a_m;
    }


    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr sampleMesh(const stl_reader::StlMesh<float, unsigned>& mesh, size_t samples)
    {
        pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZLNormal>());

        auto ntris = mesh.num_tris();

        cloud->reserve(ntris*samples);

        for(size_t i_tri = 0; i_tri < ntris; ++i_tri)
        {
            const Eigen::Vector3f
                pt1(mesh.tri_corner_coords(i_tri, 0)),
                pt2(mesh.tri_corner_coords(i_tri, 1)),
                pt3(mesh.tri_corner_coords(i_tri, 2)),
                normal(mesh.tri_normal(i_tri));

            for(size_t i_sample = 0; i_sample < samples; ++i_sample)
            {
                auto pos = sampleTrianglePoint(pt1, pt2, pt3, deviate(1.0), deviate(1.0));
                pcl::PointXYZLNormal pt;
                pt.x = pos.x(); pt.y = pos.y(); pt.z = pos.z();
                pt.normal_x = normal.x(); pt.normal_y = normal.y(); pt.normal_z = normal.z();
                pt.label = 0;

                cloud->push_back(pt);
            }
        }

        return cloud;
    }
}