#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <thread>
#include <filesystem>
#include "ply_loader.hpp"
#include "poly_gen.hpp"
#include <Eigen/Core>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

using namespace std::chrono_literals;

std::vector<size_t> selection;
template<typename T = float>
struct vec2
{
    T x, y;

    vec2(T _x, T _y) : x(_x), y(_y){}

    template<typename U>
    vec2<T> operator +(const vec2<U>& other)
    {
        return vec2(x + other.x, y + other.y);
    }

    template<typename U>
    vec2<T> operator -(const vec2<U>& other)
    {
        return vec2(x - other.x, y - other.y);
    }

    T length()
    {
        return std::sqrt(x*x + y*y);
    }

    template<typename U>
    T distance(const vec2<U>& other)
    {
        return ((*this) - other).length();
    }
};

std::vector<vec2<>> projected;

inline size_t selectNearestPoint(pcl::visualization::PCLVisualizer::Ptr viewer, unsigned x, unsigned y)
{
    size_t nearest_pt = 0;
    float distance = projected[0].distance(vec2<unsigned>{x,y});
    for(size_t i = 1; i < projected.size(); ++i)
    {
        auto dist = projected[i].distance(vec2<unsigned>{x,y});
        if(dist < distance)
        {
            distance = dist;
            nearest_pt = i;
        }
    }

    return nearest_pt;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);

void mouseEventOccurred (const pcl::visualization::MouseEvent &event, void* viewer_void) 
{
    auto viewer = *static_cast<pcl::visualization::PCLVisualizer::Ptr*>(viewer_void);

    if(viewer)
    {
        if(event.getButton() == pcl::visualization::MouseEvent::LeftButton && event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease)
        {
            std::cout << "Clicked at (" << event.getX() << "; " << event.getY() << ")" << std::endl;
            if(selection.size() < 3)
            {
                selection.push_back(selectNearestPoint(viewer, event.getX(), event.getY()));
                viewer->updateText("Select next point", 0, 0, "prompt");
                std::cout << "Selected point with index " << selection.back() << std::endl;

            }

            if(selection.size() == 3)
            {
                auto bb = clouds::computeAlignedBox(downsampled);
                auto normal = clouds::computeCoeffs({downsampled->at(selection[0]), downsampled->at(selection[1]), downsampled->at(selection[2])});
                auto plane = clouds::generatePlane(normal, bb);

                viewer->addModelFromPolyData(plane, "plane");

                auto normal_vec = Eigen::Vector3f(normal.x(), normal.y(), normal.z());
                auto up_vec = Eigen::Vector3f(0,0,1);
                
                auto quat = Eigen::Quaternionf::FromTwoVectors(normal_vec, up_vec);
                auto pt0 = downsampled->at(selection[0]);
                
                Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
                translation (0,3) = -pt0.x;
                translation (1,3) = -pt0.y;
                translation (2,3) = -pt0.z;

                Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
                auto rmat = quat.matrix();
                rotation (0,0) = rmat (0,0); rotation (0,1) = rmat (0,1); rotation (0,2) = rmat (0,2);
                rotation (1,0) = rmat (1,0); rotation (1,1) = rmat (1,1); rotation (1,2) = rmat (1,2);
                rotation (2,0) = rmat (2,0); rotation (2,1) = rmat (2,1); rotation (2,2) = rmat (2,2);
                rotation (3,3) = 1;
                
                pcl::transformPointCloud(*downsampled, *transformed_cloud, rotation*translation);
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
                viewer->addPointCloud(transformed_cloud, single_color, "transformed");



            }
        }
    }
}

int main (int argc, char** argv)
{
    std::string filename = "test_pcd.pcd";
    if(argc >= 1)
    {
        filename = argv[1];
    }

    using path = std::filesystem::path;

    

    auto extstr = path(filename).extension().string();
    if(extstr == ".pcd")
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (filename, *cloud) == -1) //* load the file
        {
            ///PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            std::cerr << "Couldn't read file " << filename << std::endl;
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
            return -1;
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
    

    filter.setInputCloud(cloud);
    filter.setLeafSize(5,5,5);
    filter.filter(*downsampled);

    auto cnt = downsampled->width*downsampled->height;
    std::cout << "Downsampled to " << cnt << "points" << std::endl;
    
    std::cout << "Opening viewer..." << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    //viewer->addPointCloud<pcl::PointXYZ> (downsampled, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    std::cout << "Calculating projections" << std::endl;

    std::vector<pcl::visualization::Camera> cameras;
    viewer->getCameras(cameras);
    auto cam = cameras[0];

    projected.reserve(cnt);

    Eigen::Matrix4d proj, view;
    cam.computeProjectionMatrix(proj);
    cam.computeViewMatrix(view);
    auto cmat = proj*view;

    for(size_t i = 0; i < cnt; ++i)
    {
        auto& pt = downsampled->at(i);
        Eigen::Vector4d evt;
        if(pt.x != 0 || pt.y != 0 || pt.z != 0)
        {
            cam.cvtWindowCoordinates(pt, evt, cmat);
            projected.emplace_back(evt.x(), evt.y());
        }
    }


    viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);

    viewer->addText("Select first point", 0, 0, "prompt");

    std::cout << "Using RegionGrowing..." << std::endl;

    pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (downsampled);
    normal_estimator.setKSearch (50);
    normal_estimator.compute (*normals);
    std::cout << "Calculated normals" << std::endl;
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (1000000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (downsampled);
    //reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();

    std::cout << "Segmentation done" << std::endl;
    viewer->addPointCloud(colored_cloud, "colored");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "colored");


     while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(100ms);
    }

    return (0);
}