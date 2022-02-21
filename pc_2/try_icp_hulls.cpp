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
#include <pcl/registration/icp_nl.h>

#include <string>
#include <functional> // std::function
#include <map>

#include <pcl/registration/super4pcs.h>

#include <gr/utils/shared.h>
#include <unistd.h>


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
static double delta = 5.0; // 8/20 at 5.0

// Estimated overlap (see the paper).
static double overlap = 0.3; // 0.3

// Threshold of the computed overlap for termination. 1.0 means don't terminate
// before the end.
static double thr = 1.0; // 1.0

// Maximum norm of RGB values between corresponded points. 1e9 means don't use.
static double max_color = -1;

// Number of sampled points in both files. The 4PCS allows a very aggressive
// sampling.
static int n_points = 2000; //2000

// Maximum angle (degrees) between corresponded normals.
static double norm_diff = 5; // -1

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


using PtIndex = size_t;
using PtSet = std::vector<PtIndex>;
using Correspondence = std::pair<PtSet, PtSet>;
using CorrSet = std::vector<Correspondence>;

struct CorrData
{
    CorrSet set;
    Correspondence all_pts;
};
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

// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
  // Point clouds
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr object_aligned (new PointCloudT);
  PointCloudT::Ptr object_backup(new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);

  FeatureCloudT::Ptr object_features (new FeatureCloudT);
  FeatureCloudT::Ptr scene_features (new FeatureCloudT);

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

  //pcl::transformPointCloud(*object, *object, randomRotationMatrix().matrix());
  //pcl::copyPointCloud(*object, *object_backup);
  pcl::IterativeClosestPointNonLinear<PointNT, PointNT> align;
  align.setInputSource(object);
  align.setInputTarget(scene);
  align.setMaximumIterations(10000);
   
  {
    pcl::ScopeTime t("Alignment");
    align.align (*object_aligned);
  }

  if (align.hasConverged ())
  {
    // Print results
    printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");

    // Show alignment
    pcl::visualization::PCLVisualizer visu("Alignment - Super4PCS");
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
    visu.addPointCloud (object_backup, ColorHandlerT (object, 255.0, 0.0, 0.0), "object");
    visu.addText(std::to_string(align.getFitnessScore()), 0,30);
    visu.addText(std::to_string(getpid()), 0, 50, "pidtext");

    std::string output = "output." + std::to_string(getpid()) + ".pcd";

    pcl::console::print_highlight ("Saving registered cloud to %s ...\n", output.c_str());
    pcl::io::savePCDFile<PointNT>(output, *object_aligned);

    visu.spin ();
  }
  else
  {
    pcl::console::print_error ("Alignment failed!\n");
    return (-1);
  }

  return (0);
}