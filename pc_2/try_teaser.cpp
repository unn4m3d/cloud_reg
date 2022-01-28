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
#include <teaser/matcher.h>
#include <teaser/registration.h>
#include <teaser/fpfh.h>


using PointT = pcl::PointXYZ;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<pcl::Normal> NormalCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointT,pcl::Normal,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

int main()
{
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);
  PointCloudT::Ptr object_d (new PointCloudT);
  PointCloudT::Ptr scene_d (new PointCloudT);
  PointCloudT::Ptr object_aligned(new PointCloudT);
  NormalCloudT::Ptr o_n(new NormalCloudT), s_n(new NormalCloudT);
  FeatureCloudT::Ptr o_f(new FeatureCloudT), s_f(new FeatureCloudT);
  
  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../test/model.pcd", *object) < 0 ||
      pcl::io::loadPCDFile<pcl::PointXYZ> ("../test/scene.pcd", *scene) < 0)
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
  }
  
  // Downsample
  pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointT> grid;
  const float leaf = 10;
  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (object);
  grid.filter (*object_d);
  grid.setInputCloud (scene);
  grid.filter (*scene_d);

  // Estimate normals for scene
  pcl::console::print_highlight ("Estimating normals...\n");
  pcl::NormalEstimationOMP<PointT,pcl::Normal> nest;
  nest.setRadiusSearch (10);
  nest.setInputCloud (scene_d);
  nest.compute (*s_n);
  nest.setInputCloud (object_d);
  nest.compute (*o_n);
  
  // Estimate features
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;
  fest.setRadiusSearch (10.0f);
  fest.setInputCloud (object_d);
  fest.setInputNormals (o_n);
  fest.compute (*o_f);
  fest.setInputCloud (scene_d);
  fest.setInputNormals (s_n);
  fest.compute (*s_f);


  teaser::PointCloud teaser_object, teaser_scene;
  pcl::console::print_warn("Using safe but slow conversion from pcl::PointCloud to teaser::PointCloud\n");
  {
      pcl::ScopeTime t("Conversion");

      for(auto& p : object_d->points)
      {
          teaser_object.push_back({p.x, p.y, p.z});
      }

      for(auto& p : scene_d->points)
      {
          teaser_scene.push_back({p.x, p.y, p.z});
      }
  }

  pcl::console::print_highlight("Matching correspondences\n");

  teaser::Matcher matcher;
  auto correspondences = matcher.calculateCorrespondences(
      teaser_object, teaser_scene, *o_f, *s_f, false, true, false, 0.95);

  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = 0.05;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 100;
  params.rotation_gnc_factor = 1.4;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  params.rotation_cost_threshold = 0.005;

  teaser::RobustRegistrationSolver solver(params);


  pcl::console::print_highlight("Solving \n");
  {
      pcl::ScopeTime t("Registration");

      solver.solve(teaser_object, teaser_scene, correspondences);
  }
  
  //if (align.hasConverged ())
  //{
    //Print results
    printf ("\n");
    //Eigen::Matrix4f transformation = align.getFinalTransformation ();
    auto transformation = solver.getSolution();
    std::cout << "Valid : " << std::boolalpha << transformation.valid << std::endl;
    std::cout << "Translation : " << transformation.translation << std::endl;
    std::cout << "Scale :" << transformation.scale << std::endl;
    std::cout << "Rotation " << std::endl << transformation.rotation << std::endl;
    

    Eigen::Transform<double, 3, Eigen::TransformTraits::Affine> affine = 
        Eigen::Translation3d(transformation.translation) * 
        transformation.rotation *
        Eigen::Scaling(transformation.scale);

    pcl::transformPointCloud(*object_d, *object_aligned, affine.matrix());

    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (scene_d, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
    visu.spin ();
  
  return (0);
}