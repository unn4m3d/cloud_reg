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
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>



using PointT = pcl::PointXYZ;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::PointCloud<pcl::Normal> NormalCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT, PointNT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

using loadfunc = std::function<int(const std::string &, pcl::PointCloud<PointNT> &)>;
const std::map<std::string, loadfunc> loaders =
{
    { "obj", pcl::io::loadOBJFile<PointNT> },
    { "ply", pcl::io::loadPLYFile<PointNT> },
    { "pcd", pcl::io::loadPCDFile<PointNT> }
};


// src: https://en.cppreference.com/w/cpp/string/byte/tolower
std::string str_tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); } // correct
                  );
    return s;
}

bool
load(const std::string& filename, PointCloudT::Ptr& pcloud){
    auto getFileExt = [] (const std::string& s) -> std::string {

       size_t i = s.rfind('.', s.length());
       return i != std::string::npos
                 ? str_tolower( s.substr(i+1, s.length() - i) )
                 : "";
    };

    std::string ext = getFileExt(filename);
    auto l = loaders.find(ext);
    if( l != loaders.end() ) {
        return (*l).second( filename, *pcloud ) >= 0;
    }
    pcl::console::print_error ("Unsupported file extension: %s\n", ext.c_str());
    return false;
}


int main(int argc, char** argv)
{
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);
  PointCloudT::Ptr object_aligned(new PointCloudT);
  FeatureCloudT::Ptr o_f(new FeatureCloudT), s_f(new FeatureCloudT);
  
  // Load object and scene
  if (argc < 3)
  {
    pcl::console::print_error ("Syntax is: %s scene.obj object.obj [PARAMS]\n", argv[0]);
    return (-1);
  }

  std::string objPath {argv[2]};
  std::string scenePath {argv[1]};

  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if ( ! ( load(objPath, object) && load(scenePath, scene) ) )
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (-1);
  }
  // Estimate features
  pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;
  fest.setRadiusSearch (90.0f);
  fest.setInputCloud (object);
  fest.setInputNormals (object);
  fest.compute (*o_f);
  fest.setInputCloud (scene);
  fest.setInputNormals (scene);
  fest.compute (*s_f);


  teaser::PointCloud teaser_object, teaser_scene;
  pcl::console::print_warn("Using safe but slow conversion from pcl::PointCloud to teaser::PointCloud\n");
  {
      pcl::ScopeTime t("Conversion");

      for(auto& p : object->points)
      {
          teaser_object.push_back({p.x, p.y, p.z});
      }

      for(auto& p : scene->points)
      {
          teaser_scene.push_back({p.x, p.y, p.z});
      }
  }

  pcl::console::print_highlight("Matching correspondences\n");

  teaser::Matcher matcher;
  auto correspondences = matcher.calculateCorrespondences(
      teaser_object, teaser_scene, *o_f, *s_f, false, true, false, 0.95);

  teaser::RobustRegistrationSolver::Params params;
  params.noise_bound = 0.01;
  params.cbar2 = 1;
  params.estimate_scaling = false;
  params.rotation_max_iterations = 1000000;
  params.rotation_gnc_factor = 1.1;
  params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
  params.rotation_cost_threshold = 0.001;

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

    pcl::transformPointCloud(*object, *object_aligned, affine.matrix());

    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
    visu.spin ();
  
  return (0);
}