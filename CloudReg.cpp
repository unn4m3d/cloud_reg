#include "CloudReg.hpp"
#include "happly.hpp"
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <regex>
#include <charconv>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <thread>

using namespace clouds;

static void loadPLY(const std::filesystem::path& path, CloudReg::PCPtr& cloud)
{
    happly::PLYData input(path.string());

    auto vertices = input.getVertexPositions();

    auto obj_info = input.objInfoComments[0];
    size_t w = 1, h = vertices.size();

    std::regex rx("Width=(\\d+); Height=(\\d+)");
    std::smatch m;
    if(std::regex_search(obj_info, m, rx))
    {
        size_t _w, _h;
        auto m1b = m[1].str().data(), m2b = m[2].str().data();
        auto m1e = m1b + m[1].str().size(), m2e = m2b + m[2].str().size();
        auto res_w = std::from_chars(m1b, m1e, _w);
        auto res_h = std::from_chars(m2b, m2e, _h);

        if(res_w.ec == std::errc() && res_h.ec == std::errc())
        {
            w = _w, h = _h;
        }
        else
        {
            throw std::runtime_error("Cannot parse width or height");
        }
    }

    cloud->resize(vertices.size());
    cloud->width = w;
    cloud->height = h;

    for(size_t i = 0; i < vertices.size(); ++i)
    {
        decltype(auto) pt = cloud->at(i);
        pt.x = vertices[i][0];
        pt.y = vertices[i][1];
        pt.z = vertices[i][2];
    }
}

CloudReg::PCPtr CloudReg::loadFile(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    CloudReg::PCPtr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    std::cout << "Loading file " << path.string() << " ... ";
    std::cout.flush();
    timer.tic();
    if(ext == ".pcd")
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (path.string(), *cloud) == -1) //* load the file
        {
            throw std::runtime_error(std::string("Cannot load .pcd file") + path.string()); 
        }
    }
    else if(ext == ".ply")
    {
        loadPLY(path, cloud);
    }   
    else
    {
        throw std::runtime_error(std::string("Unknown file extension: ") + ext);
    }
    
    std::cout << cloud->size() << " pts, " << timer.toc() << " ms" << std::endl;

    return cloud;
}

void CloudReg::openViewer()
{
    viewer.reset(new pcl::visualization::PCLVisualizer("CloudReg 3D Viewer"));
    viewer->setBackgroundColor(0,0,0);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
}

void CloudReg::addCloud(const CloudReg::PCPtr& c, const std::string& id)
{
    viewer->addPointCloud(c, id);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
}

void CloudReg::addCloud(const CloudReg::PCColPtr& c, const std::string& id)
{
    viewer->addPointCloud(c, id);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
}

void CloudReg::addCloud(const CloudReg::PCPtr& c, const CloudReg::PCColHandler& col, const std::string& id)
{
    viewer->addPointCloud(c, col, id);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, id);
}

void CloudReg::downsample(const CloudReg::PCPtr& input, CloudReg::PCPtr& output, float leaf_size)
{
    std::cout << "Downsampling with leaf_size = " << leaf_size << "... ";
    std::cout.flush();
    timer.tic();

    pcl::VoxelGrid<pcl::PointXYZ> filter;
    
    filter.setInputCloud(input);
    filter.setLeafSize(leaf_size,leaf_size,leaf_size);
    filter.filter(*output);

    auto cnt = output->width*output->height;
    std::cout << cnt << " pts, " << timer.toc() << " ms" << std::endl;
}

void CloudReg::estimateNormals(const PCPtr& input, PCNormalPtr& output)
{
    std::cout << "Estimating normals... ";
    std::cout.flush();
    timer.tic();
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod (search_tree);
    normal_estimator.setInputCloud (input);
    normal_estimator.setKSearch (50);
    normal_estimator.compute (*output);
    std::cout << timer.toc() << " ms" << std::endl;
}

CloudReg::PCColPtr CloudReg::regionGrowing(const PCPtr& input, const PCNormalPtr& normals, std::vector<pcl::PointIndices>& clusters)
{
    std::cout << "Segmenting cloud w/ RegionGrowing... ";
    std::cout.flush();
    timer.tic();

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (1000000);
    reg.setSearchMethod (search_tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (input);
    //reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);

    reg.extract(clusters);

    std::cout << timer.toc() << " ms" << std::endl;
    return reg.getColoredCloud();
}

std::vector<Eigen::Vector3f> CloudReg::centersOfMass(const PCPtr& cloud, const std::vector<pcl::PointIndices>& indices)
{
    std::cout << "Computing centers of mass for " << indices.size() << "segments... ";
    std::cout.flush();
    timer.tic();
    std::vector<Eigen::Vector3f> centers;
    centers.reserve(indices.size());

    for(auto& pt_indices : indices)
    {
        auto& ind = pt_indices.indices;
        float weight = 1.0f/ind.size();
        centers.push_back(Eigen::Vector3f(0,0,0));
        auto& vec = centers.back();
        for(auto& idx : ind)
        {
            auto& pt = cloud->at(idx);
            vec += Eigen::Vector3f(pt.x, pt.y, pt.z) * weight;
        }
    }

    std::cout << timer.toc() << " ms" << std::endl;
    return centers;
}

CloudReg::CloudReg() :
    cloud(new pcl::PointCloud<pcl::PointXYZ>),
    downsampled(new pcl::PointCloud<pcl::PointXYZ>),
    search_tree(new pcl::search::KdTree<pcl::PointXYZ>)
    {}

void CloudReg::run(int argc, char** argv)
{
    std::filesystem::path input("test_pcd.pcd");
    if(argc >= 1)
    {
        input = argv[1];
    }

    this->cloud = loadFile(input);

    downsample(this->cloud, this->downsampled, 5);

    CloudReg::PCNormalPtr normals(new pcl::PointCloud<pcl::Normal>);
    estimateNormals(downsampled, normals);

    std::vector<pcl::PointIndices> clusters;

    auto colored_cloud = regionGrowing(downsampled, normals, clusters);

    openViewer();
    addCloud(colored_cloud, "segmented");

    auto centers = centersOfMass(downsampled, clusters);

    auto center_idx = 0;
    for(auto& c : centers)
    {
        
        viewer->addSphere(pcl::PointXYZ(c.x(), c.y(), c.z()),10, 0, 0, 255, std::string("sphere") + std::to_string(center_idx++));
    }
    
    while(!viewer->wasStopped())
    {
        yield();
    }
}

void CloudReg::yield()
{
    using namespace std::chrono_literals;
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
}
