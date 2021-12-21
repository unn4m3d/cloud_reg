#include "poly_gen.hpp"
#include <pcl/common/transforms.h>

namespace clouds
{
    Eigen::AlignedBox<float, 3> computeAlignedBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        float minx = 0, miny = 0, minz = 0, maxx = 0, maxy = 0, maxz = 0;
        for(auto& pt : *cloud)
        {
            if(pt.x < minx) minx = pt.x;
            if(pt.x > maxx) maxx = pt.x;
            if(pt.y < miny) miny = pt.y;
            if(pt.y > maxy) maxy = pt.y;
            if(pt.z < minz) minz = pt.z;
            if(pt.z > maxz) maxz = pt.z;
        }

        Eigen::AlignedBox<float, 3> box(Eigen::Vector3f{minx,miny,minz}, Eigen::Vector3f{maxx, maxy, maxz});
        return box;
    }

    Eigen::Vector4f computeCoeffs(const std::vector<pcl::PointXYZ>& three_pts)
    {
        pcl::PointXYZ 
            p0 = three_pts[0],
            p1 = three_pts[1],
            p2 = three_pts[2];

        float 
            x0 = p0.x, x1 = p1.x, x2 = p2.x,
            y0 = p0.y, y1 = p1.y, y2 = p2.y,
            z0 = p0.z, z1 = p1.z, z2 = p2.z;

        float
            xc = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0),
            yc = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0),
            zc = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);

        float a = xc, b = yc, c = zc, d = -x0*xc - y0*yc - z0*zc;

        return Eigen::Vector4f(a,b,c,d);
    }

    vtkSmartPointer<vtkPolyData> generatePlane(const Eigen::Vector4f& normal, const Eigen::AlignedBox<float, 3>& bounding_box)
    {
        vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();

        float a = normal.x(), b = normal.y(), c = normal.z();
        plane->SetNormal(normal.x(),normal.y(),normal.z());
        // Ax + By + Cz + D = 0
        // z = -(Ax + By + D) / C;
        auto mn = bounding_box.min(), mx = bounding_box.max();
        plane->SetOrigin(mn.x(), mn.y(), (-a*mn.x() - b*mn.y() - normal.w())/c);
        plane->SetPoint1(mn.x(), mx.y(), (-a*mn.x() - b*mx.y() - normal.w())/c);
        plane->SetPoint2(mx.x(), mn.y(), (-a*mx.x() - b*mn.y() - normal.w())/c);

        plane->Update();

        return plane->GetOutput();
    }

    inline pcl::PointCloud<pcl::PointXYZLNormal>::Ptr generatePts(size_t num)
    {
        pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);

        auto dxi = 1 / std::ceil(std::sqrt(num * 2));
        cloud->reserve(num);

        // y = 1 - x
        for(float x = 0.0; x <= 1.0; x += dxi)
            for(float y = 0.0; y <= 1.0 - x; y += dxi)
            {
                pcl::PointXYZLNormal pt;
                pt.x = x; pt.y = y; pt.z = 1;
                pt.normal_x = 0; pt.normal_y = 0; pt.normal_z = 1;
                cloud->push_back(pt);
            }
        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr generatePts(const std::vector<pcl::PointXYZ>& three_pts, const Eigen::Vector3f& normal, size_t num)
    {
        pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);

        auto dxi = 1 / std::ceil(std::sqrt(num * 2));
        cloud->reserve(num);

        float inv_src_data[] = {
            -1.0, -1.0, 1.0,
             0.0, 1.0, 0.0,
             1.0, 0.0, 0.0
        };

        float dest_data[] = {
            three_pts[0].x, three_pts[1].x, three_pts[2].x,
            three_pts[0].y, three_pts[1].y, three_pts[2].y,
            three_pts[0].z, three_pts[1].z, three_pts[2].z
        };

        Eigen::Matrix3f inv_source(inv_src_data), destination(dest_data);

        auto transform = inv_source * destination;

        Eigen::Vector3f orig(0,0,1);

        for(; orig.x() <= 1.0; orig.x() += dxi)
            for(; orig.y() <= 1.0 - orig.x(); orig.y() += dxi)
            {
                auto tr = transform * orig;
                pcl::PointXYZLNormal pt;
                pt.x = tr.x(); pt.y = tr.y(); pt.z = tr.z();
                pt.normal_x = normal.x(); pt.normal_y = normal.y(); pt.normal_z = normal.z();
                cloud->push_back(pt);
            }

        return cloud;

    }
}