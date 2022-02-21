#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/region_growing.h>

typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

auto centersOfMass(const typename PointCloudT::Ptr& cloud, const std::vector<pcl::PointIndices>& indices)
{

    PointCloudT::Ptr output(new PointCloudT);

    for(auto& pt_indices : indices)
    {
        auto& ind = pt_indices.indices;
        float weight = 1.0f/ind.size();
        output->points.push_back(PointNT());
        auto& point = output->points.back();
        for(auto& idx : ind)
        {
            auto& pt = cloud->at(idx);
            point.x += pt.x * weight;
            point.y += pt.y * weight;
            point.z += pt.z * weight;
            point.normal_x = pt.normal_x * weight;
            point.normal_y = pt.normal_y * weight;
            point.normal_z = pt.normal_z * weight;
        }
    }
    return output;
}

int main()
{
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr object_aligned (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);
  FeatureCloudT::Ptr object_features (new FeatureCloudT);
  FeatureCloudT::Ptr scene_features (new FeatureCloudT);
  
  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<PointNT> ("../test/extracted.pcd", *object) < 0 ||
      pcl::io::loadPCDFile<PointNT> ("../test/scene.pcd", *scene) < 0)
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
  }
  
  // Downsample
  pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointNT> grid;
  const float leaf = 4;
  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (object);
  grid.filter (*object);
  grid.setInputCloud (scene);
  grid.filter (*scene);
  
  // Estimate normals for scene
  pcl::console::print_highlight ("Estimating scene normals...\n");
  pcl::NormalEstimationOMP<PointNT,PointNT> nest;
  nest.setRadiusSearch (10);
  nest.setInputCloud (scene);
  nest.compute (*scene);
  
  // Estimate features
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;
  fest.setRadiusSearch (16.0f);
  fest.setInputCloud (object);
  fest.setInputNormals (object);
  fest.compute (*object_features);
  fest.setInputCloud (scene);
  fest.setInputNormals (scene);
  fest.compute (*scene_features);
  
  // Perform alignment
  pcl::console::print_highlight ("Starting alignment...\n");
  pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
  align.setInputSource (object);
  align.setSourceFeatures (object_features);
  align.setInputTarget (scene);
  align.setTargetFeatures (scene_features);
 align.setMaximumIterations (2000); // Number of RANSAC iterations
  align.setNumberOfSamples (5); // Number of points to sample for generating/prerejecting a pose
  align.setCorrespondenceRandomness (3); // Number of nearest features to use
  align.setSimilarityThreshold (0.4f); // Polygonal edge length similarity threshold
  align.setMaxCorrespondenceDistance (5.0f * leaf); // Inlier threshold
  align.setInlierFraction (0.2f); // Required inlier fraction for accepting a pose hypothesis
  {
    pcl::ScopeTime t("RANSAC Alignment");
    align.align (*object_aligned);
  }
  
  if (align.hasConverged ())
  {
    //Print results
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());

    // Segmentation
    pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr nms(new pcl::PointCloud<pcl::Normal>);

    pcl::copyPointCloud(*object_aligned, *pts);
    pcl::copyPointCloud(*object_aligned, *nms);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (1000000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (pts);
    //reg.setIndices (indices);
    reg.setInputNormals (nms);
    reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    auto com = centersOfMass(object_aligned, clusters);
    
    pcl::copyPointCloud(*scene, *pts);
    pcl::copyPointCloud(*scene, *nms);
    reg.setInputCloud(pts);
    reg.setInputNormals(nms);
    clusters.erase(clusters.begin(), clusters.end());
    reg.extract(clusters);
    
    auto com_scene = centersOfMass(scene, clusters);


    PointCloudT::Ptr com_aligned(new PointCloudT);

    pcl::IterativeClosestPoint<PointNT, PointNT> icp;
    icp.setMaximumIterations (100000);
    icp.setInputSource (com);
    icp.setInputTarget (com_scene);
    {
      pcl::ScopeTime t("ICP Alignment");
      icp.align (*com_aligned);
    }
    
    if(icp.hasConverged())
    {
      // Show alignment
      pcl::console::print_info("ICP rules\n");

      auto align = icp.getFinalTransformation();

      pcl::transformPointCloud(*object_aligned, *object_aligned, align);

      pcl::visualization::PCLVisualizer visu("Alignment");
      visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
      visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
      visu.spin ();
    }
    else
    {
      pcl::console::print_error("ICP fucked up\n");
    }
  }
  else
  {
    pcl::console::print_error ("Alignment failed!\n");
   return (1);
  }
  
  return (0);
}