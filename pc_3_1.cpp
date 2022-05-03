#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/icp_nl.h>
#include <vtkRenderWindow.h>

#include <string>
#include <functional> // std::function
#include <map>

#include <pcl/registration/super4pcs.h>

#include <gr/utils/shared.h>
#include <unistd.h>
#include "matconv.hpp"


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


// Align a rigid object to a scene with clutter and occlusions
int
main (int argc, char **argv)
{
  // Point clouds
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr scene (new PointCloudT);
  PointCloudT::Ptr object_b (new PointCloudT);
  PointCloudT::Ptr scene_b (new PointCloudT);
  PointCloudT::Ptr object_a (new PointCloudT);

  // Get input object and scene
  if (argc < 4)
  {
    pcl::console::print_error ("Syntax is: %s scene.obj object.obj tr1 tr2\n", argv[0]);
    return (-1);
  }

  std::string objPath {argv[2]};
  std::string scenePath {argv[1]};
  std::string trPath { argv[3] };
  std::string cropPath {argv[4] };

  // Load object and scene
  pcl::console::print_highlight ("Loading point clouds...\n");
  if ( ! ( load(objPath, object) && load(scenePath, scene) ) )
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (-1);
  }

  std::ifstream tr(trPath), crop(cropPath);
  auto mat = clouds::parse_matrix<4,4,float>(tr);
  auto mat2 = clouds::parse_matrix<4,4,float>(crop);


pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointNT> grid;
  const float leaf = 4.0f;
  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (object);
  grid.filter (*object);
  grid.setInputCloud (scene);
  grid.filter (*scene);

  pcl::copyPointCloud(*object, *object_b);
  pcl::copyPointCloud(*scene, *scene_b);
  pcl::transformPointCloudWithNormals(*scene, *scene, mat);
  pcl::transformPointCloudWithNormals(*object, *object, mat2);

  /*pcl::CropBox<PointNT> cbox;
  cbox.setMin(Eigen::Vector4f(crop_mat(0, 0), crop_mat(1, 0), crop_mat(2, 0), 1));
  cbox.setMax(Eigen::Vector4f(crop_mat(0, 1), crop_mat(1, 1), crop_mat(2, 1), 1));
  cbox.setRotation(Eigen::Vector3f(crop_mat(0, 2), crop_mat(1, 2), crop_mat(2, 2)));
  cbox.setInputCloud(scene);
  cbox.filter(*scene);*/
  

  Eigen::Vector3f centroid(0,0,0);
  
  for(const auto& pt : scene->points)
  {
    centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
  }

  centroid /= scene->points.size();

   /* pcl::search::Search<PointNT>::Ptr tree;
    if (object->isOrganized ())
    {
        tree.reset (new pcl::search::OrganizedNeighbor<PointNT> ());
    }
    else
    {
        tree.reset (new pcl::search::KdTree<PointNT> (false));
    }

    // Set the input pointcloud for the search tree
    tree->setInputCloud (object);

    pcl::NormalEstimationOMP<PointNT, PointNT> ne;
    ne.setInputCloud (object);
    ne.setKSearch(6);
    ne.setSearchMethod (tree);
    ne.compute(*object);

    tree->setInputCloud(scene);
    ne.setInputCloud(scene);
    ne.compute(*scene); */
  
    pcl::visualization::PCLVisualizer visu("Alignment - Manual");
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (object, ColorHandlerT (object, 255.0, 0.0, 0.0), "object");
    //visu.addPointCloud (scene_b, ColorHandlerT(scene_b, 0.0, 0.0, 255.0), "ob");
    visu.addText(std::to_string(getpid()), 0, 50, "pidtext");
    visu.addOrientationMarkerWidgetAxes(visu.getRenderWindow()->GetInteractor());
    visu.addSphere(pcl::PointXYZ(centroid.x(), centroid.y(), centroid.z()), 20, "sph");
    visu.addSphere(pcl::PointXYZ(0, 0, 0), 20, "sph0");

    visu.spin ();


  return (0);
}