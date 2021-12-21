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
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include "stl_reader.hpp"
#include "poly_gen.hpp"
#include "point_sampler.hpp"

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

std::vector<pcl::ConvexHull<pcl::PointXYZ>> CloudReg::calculateHulls(const PCPtr& cloud, std::vector<pcl::PointIndices>& clusters)
{
    std::vector<pcl::ConvexHull<pcl::PointXYZ>> vec;
    vec.reserve(clusters.size());

    std::cout << "Calculating convex hulls... ";
    std::cout.flush(); 
    timer.tic(); 


    for(size_t i = 0; i < clusters.size(); ++i)
    {
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr 
            inliers (new pcl::PointIndices), 
            indices(&clusters[i], [](pcl::PointIndices*){});
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (0.01);
        seg.setIndices(indices);
        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);

        PCPtr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::ProjectInliers<pcl::PointXYZ> proj;
        proj.setModelType (pcl::SACMODEL_PLANE);
        //proj.setIndices (*inliers);
        proj.setInputCloud (cloud);
        proj.setModelCoefficients (coefficients);
        proj.filter (*cloud_projected);


        pcl::ConvexHull<pcl::PointXYZ> chull;
        chull.setInputCloud (cloud_projected);
        //chull.setAlpha (0.1);
        
        vec.push_back(chull);
    }

    std::cout << timer.toc() << " ms" << std::endl;

    return vec;
}


CloudReg::CloudReg() :
    cloud(new pcl::PointCloud<pcl::PointXYZ>),
    downsampled(new pcl::PointCloud<pcl::PointXYZ>),
    transformed(new pcl::PointCloud<pcl::PointXYZ>),
    search_tree(new pcl::search::KdTree<pcl::PointXYZ>)
    {}

static pcl::ModelCoefficients approxPlane(const CloudReg::PCPtr& cloud, pcl::PointIndices& indices)
{
    pcl::ModelCoefficients coeffs;
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices), ind(&indices, [](auto*){});
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    seg.setIndices(ind);
    seg.setInputCloud (cloud);
    seg.segment (*inliers, coeffs);
    return coeffs;
}

static inline auto getCloudFromSTL(const std::filesystem::path& path, float number)
{

    stl_reader::StlMesh <float, unsigned int> mesh(path.string());

    return clouds::sampleMesh(mesh, number);
}

static inline Eigen::Matrix4f createTransform(const Eigen::Vector3f& coord, const Eigen::Quaternionf& rot)
{
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation (0,3) = -coord.x();
    translation (1,3) = -coord.y();
    translation (2,3) = -coord.z();

    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    auto rmat = rot.matrix();
    rotation (0,0) = rmat (0,0); rotation (0,1) = rmat (0,1); rotation (0,2) = rmat (0,2);
    rotation (1,0) = rmat (1,0); rotation (1,1) = rmat (1,1); rotation (1,2) = rmat (1,2);
    rotation (2,0) = rmat (2,0); rotation (2,1) = rmat (2,1); rotation (2,2) = rmat (2,2);
    rotation (3,3) = 1;

    return rotation*translation;
}

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

    auto hulls = calculateHulls(downsampled, clusters);

    std::cout << "Finding largest hull by area... ";
    std::cout.flush();
    timer.tic();
    auto largest = std::max_element(hulls.begin(), hulls.end(), [](auto& a, auto& b){ return a.getTotalArea() < b.getTotalArea();});
    auto index = std::distance(largest, hulls.begin());

    std::cout << index << "th one, " << timer.toc() << " ms" << std::endl;

    auto& lc = centers[index];
    auto& cluster = clusters[index];

    viewer->addSphere(pcl::PointXYZ(lc.x(), lc.y(), lc.z()), 25, 255, 0, 0, "sphere_select");

    auto plane = approxPlane(downsampled, cluster);

    float size = 250;
    std::vector<pcl::PointIndices> selection_indices;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(downsampled);
    extract.setNegative(false);
    PCPtr extracted_segment(new pcl::PointCloud<pcl::PointXYZ>), trans_ext_segment(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Vector3f plane_normal(plane.values[0], plane.values[1], plane.values[2]), up(0, 0, 1);
    Eigen::Quaternionf rotation = Eigen::Quaternionf::FromTwoVectors(plane_normal, up);
    auto transform = createTransform(lc, rotation);

    for(size_t i = 0; i < centers.size(); ++i)
    {
        if(i != index)
        {
            if((centers[i]-centers[index]).norm() <= size)
            {
                selection_indices.push_back(clusters[i]);
                viewer->addSphere(pcl::PointXYZ(centers[i].x(), centers[i].y(), centers[i].z()), 25, 0, 255, 0, std::string("sphere_select") + std::to_string(i));
                pcl::PointIndices::Ptr iptr(&clusters[i], [](auto*){});
                extract.setIndices(iptr);
                extract.filter(*extracted_segment);
                pcl::transformPointCloud(*extracted_segment, *trans_ext_segment, transform);
                *transformed += *trans_ext_segment;
            }
        }
    }

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(transformed, 0, 255, 0);
    addCloud(transformed, green, "transformed_cloud");

    std::cout << "Loading pts from STL...";
    std::cout.flush();
    timer.tic();

    auto stl_cloud = getCloudFromSTL("test/test.stl", 0.2);
    //auto stl_cloud = loadFile("test/model.pcd");

    std::cout << " " << stl_cloud->width*stl_cloud->height << " pts, " << timer.toc() << " ms" << std::endl;
    std::cout << "Drawing original STL cloud..." << std::endl; 
    pcl::visualization::PointCloudColorHandlerCustom<decltype(stl_cloud)::element_type::PointType> yellow(stl_cloud, 255, 255, 0);
    viewer->addPointCloud(stl_cloud, yellow, "stl");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "stl");

    pcl::PointCloud<pcl::Normal>::Ptr tr_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr tr_normal_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    estimateNormals(transformed, tr_normals);
    pcl::concatenateFields(*transformed, *tr_normals, *tr_normal_cloud);

    pcl::IterativeClosestPointNonLinear<pcl::PointXYZLNormal, pcl::PointXYZLNormal> icp;

    icp.setInputSource(stl_cloud);
    icp.setInputTarget(tr_normal_cloud);
    std::cout << "Trying ICP " << std::endl;

    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZLNormal>);
    icp.align(*aligned);

    pcl::visualization::PointCloudColorHandlerCustom<decltype(aligned)::element_type::PointType> red(aligned, 255, 0, 0);
    viewer->addPointCloud(aligned, red, "aligned");

    std::cout << "ICP has converged: " << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

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
