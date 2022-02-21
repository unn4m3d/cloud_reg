#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>

#include <string>
#include <functional> // std::function
#include <map>

#include <pcl/registration/super4pcs.h>

#include <gr/utils/shared.h>
#include <thread>
#include <ranges>
#include <mutex>


static std::string input1 = "input1.obj";

// Second input.
static std::string input2 = "input2.obj";

// Output. The transformed second input.
static std::string output = "";
// Default name for the '.obj' output file
static std::string defaultObjOutput = "output.obj";
// Default name for the '.ply' output file
static std::string defaultPlyOutput = "output.ply";

// Transformation matrice.
static std::string outputMat = "";

// Sampled cloud 1
static std::string outputSampled1 = "";

// Sampled cloud 2
static std::string outputSampled2 = "";

// Delta (see the paper).
static double delta = 5.0;

// Estimated overlap (see the paper).
static double overlap = 0.3;

// Threshold of the computed overlap for termination. 1.0 means don't terminate
// before the end.
static double thr = 1.0;

// Maximum norm of RGB values between corresponded points. 1e9 means don't use.
static double max_color = -1;

// Number of sampled points in both files. The 4PCS allows a very aggressive
// sampling.
static int n_points = 400;

// Maximum angle (degrees) between corresponded normals.
static double norm_diff = 5;

// Maximum allowed computation time.
static int max_time_seconds = 10;

// Point type to use - ExtPointBinding demo
static int point_type = 0;
static int max_point_type = 2; // 0: gr::Point3D, 1: extlib1::PointType1, 2: extlib2::PointType2

static bool use_super4pcs = true;
// Types
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

using namespace gr;

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



typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;
using Eigen::Affine3d;

static std::random_device rd;
static std::mt19937_64 gen(rd());
static std::uniform_real_distribution<double> dis(-M_PI, M_PI);

inline Eigen::Vector3d randomUnitVector()
{
    return Eigen::Vector3d(dis(gen), dis(gen), dis(gen)).normalized();
}

inline Affine3d randomRotationMatrix()
{
    return Affine3d(Eigen::AngleAxisd(dis(gen), randomUnitVector()));
}

struct AlignResult
{
  Eigen::Affine3d initial;
  Eigen::Matrix4f transform;
  double score;
  bool ok;
};


AlignResult align(const PointCloudT::Ptr& object, const PointCloudT::Ptr& scene, const PointCloudT::Ptr& aligned)
{
  auto initial = randomRotationMatrix();
  PointCloudT::Ptr object_tr(new PointCloudT), object_backup(new PointCloudT);
  pcl::transformPointCloud(*object, *object_tr, randomRotationMatrix().matrix());

  pcl::Super4PCS<PointNT,PointNT> align;
  auto &options = align.getOptions();
  bool overlapOk = options.configureOverlap(overlap);
    if(! overlapOk )  {
        //logger.Log<Utils::ErrorReport>("Invalid overlap configuration. ABORT");
        return {initial, Eigen::Matrix4f::Identity(), 0.0, false};;
    }
    options.sample_size = n_points ;
    options.max_normal_difference = norm_diff;
    options.max_color_distance = max_color;
    options.max_time_seconds = 500;
    options.delta = delta;
  pcl::transformPointCloud(*object, *object, initial.matrix());
  align.setInputSource(object);
  align.setInputTarget(scene);

  {
      pcl::ScopeTime t("Alignment");
      align.align(*aligned);
  }

  //pcl::copyPointCloud(*object_backup, *object);

  if(align.hasConverged())
  {
    return {initial, align.getFinalTransformation(), align.getFitnessScore(), true };
  }
  else
  {
    return {initial, Eigen::Matrix4f::Identity(), 0.0, false};
  }
}

// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
  // Point clouds
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr object_aligned (new PointCloudT);
  PointCloudT::Ptr object_backup(new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);

  // Get input object and scene
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

  std::vector<AlignResult> results;
  std::vector<PointCloudT::Ptr> aligned;
  std::vector<std::thread> threads;

  constexpr int N = 12;

  results.resize(N);

  std::mutex copy_mutex;

  for(int i = 0; i < N; ++i)
  {
    aligned.emplace_back(new PointCloudT);
    threads.emplace_back([&, i](){
      PointCloudT::Ptr object_copy(new PointCloudT), scene_copy(new PointCloudT);
      {
        std::lock_guard<std::mutex> lk(copy_mutex);
        pcl::copyPointCloud(*object, *object_copy);
        pcl::copyPointCloud(*scene, *scene_copy);
        
      }
      results[i] = align(object_copy, scene_copy, aligned[i]);
    });
  }

  for(auto& t: threads) t.join();


  int index = -1;
  double max_score = 0;

  for(int i = 0; i < results.size(); ++i)
  {
    if(results[i].ok)
    {
      if(results[i].score > max_score)
      {
        index = i;
        max_score = results[i].score;
      }
    }
  }

  if(index < 0)
  {
    pcl::console::print_error("That's a fail man\n");
  }
  else
  {
    pcl::console::print_highlight("OK\n");

    pcl::visualization::PCLVisualizer visu("Alignment - Super4PCS");
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (aligned[index], ColorHandlerT (aligned[index], 0.0, 0.0, 255.0), "object_aligned");
    visu.spin ();

  }

  return (0);
}