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
#include <random>

using PointT = pcl::PointXYZ;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<pcl::Normal> NormalCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointT,pcl::Normal,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointT> ColorHandlerT;

constexpr static double norm_pow = 0.2;

NormalCloudT::Ptr o_n(new NormalCloudT), s_n(new NormalCloudT);
FeatureCloudT::Ptr o_f(new FeatureCloudT), s_f(new FeatureCloudT);

struct SolutionAndCorrs
{
    teaser::RegistrationSolution solution;
    std::vector<std::pair<int, int>> corrs;
};

SolutionAndCorrs teaserStep(PointCloudT::Ptr object, PointCloudT::Ptr scene, teaser::RobustRegistrationSolver::Params& params)
{
    pcl::console::print_highlight("TEASER Step...\n");

    pcl::console::print_highlight ("Estimating normals...\n");
    pcl::NormalEstimationOMP<PointT,pcl::Normal> nest;
    nest.setRadiusSearch (10);
    nest.setInputCloud (scene);
    nest.compute (*s_n);
    nest.setInputCloud (object);
    nest.compute (*o_n);
    
    // Estimate features
    pcl::console::print_highlight ("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch (10.0f);
    fest.setInputCloud (object);
    fest.setInputNormals (o_n);
    fest.compute (*o_f);
    fest.setInputCloud (scene);
    fest.setInputNormals (s_n);
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
    teaser::RobustRegistrationSolver solver(params);

    {
        pcl::ScopeTime t("TEASER");
        solver.solve(teaser_object, teaser_scene, correspondences);
    }

    return {solver.getSolution(), correspondences};
}

using Affine3d = Eigen::Transform<double, 3, Eigen::Affine>;

inline Affine3d getTransform(const teaser::RegistrationSolution& solution)
{
    return Eigen::Translation3d(solution.translation) * 
        solution.rotation *
        Eigen::Scaling(solution.scale);
}

inline Affine3d getTransform(const SolutionAndCorrs& solution)
{
    return getTransform(solution.solution);
}

constexpr static double samplingStep = M_PI/2;


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

struct RandomSampleResult
{
    Affine3d transform, pretransform;
    double prescore, score;
};

template<typename Point>
double norm(const Point& a, const Point& b, double n = 2.0)
{
    return std::pow(
        std::pow(std::abs(a.x - b.x), n) +
        std::pow(std::abs(a.y - b.y), n) + 
        std::pow(std::abs(a.z - b.z), n), (n == 0.0 ? 1.0 : 1.0/n));
}

template<typename Cloud>
double scoreClouds(const Cloud& a, const Cloud& b)
{
    double score = 0.0;
    #pragma omp parallel for collapse(2) reduction(min: sc_p) reduction(+: score)
    for(auto& pt_a : a)
    {
        double sc_p;
        for(auto& pt_b : b)
        {
            sc_p = norm(pt_a, pt_b, norm_pow);
        }
        score += sc_p;
    }
    return score;
}

template<typename Point>
inline Point transform(const Point& pt, const Affine3d& transform)
{
    Eigen::Vector3d ptv(pt.x, pt.y, pt.z);
    auto ptnv = transform * ptv;
    return Point(ptnv.x(), ptnv.y(), ptnv.z());
}

template<typename Cloud>
double scoreClouds(const Cloud& a, const Cloud& b, const std::vector<std::pair<int, int>>& corrs, const Affine3d& tr)
{
    double score = 0.0;
    #pragma omp parallel for reduction(+: score)
    for(auto& corr : corrs)
    {
        score += norm(transform(a.at(corr.first), tr), b.at(corr.second), norm_pow);
    }
    return score;
}

int main()
{
    PointCloudT::Ptr object (new PointCloudT);
    PointCloudT::Ptr scene (new PointCloudT);
    PointCloudT::Ptr object_d (new PointCloudT);
    PointCloudT::Ptr scene_d (new PointCloudT);
    PointCloudT::Ptr object_aligned(new PointCloudT);
    
    // Load object and scene
    pcl::console::print_highlight ("Loading point clouds...\n");
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("../test/model.pcd", *object) < 0 ||
        pcl::io::loadPCDFile<pcl::PointXYZ> ("../test/extracted.pcd", *scene) < 0)
    {
        pcl::console::print_error ("Error loading object/scene file!\n");
        return (1);
    }
    
    // Downsample
    pcl::console::print_highlight ("Downsampling...\n");
    pcl::VoxelGrid<PointT> grid;
    const float leaf = 2;
    grid.setLeafSize (leaf, leaf, leaf);
    grid.setInputCloud (object);
    grid.filter (*object_d);
    grid.setInputCloud (scene);
    grid.filter (*scene_d);

    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = 0.01;
    params.cbar2 = 1;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 10000;
    params.rotation_gnc_factor = 1.1;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::FGR;
    params.rotation_cost_threshold = 0.001;

    pcl::console::print_highlight("Initial TEASER alignment...\n");

    auto solution = teaserStep(object_d, scene_d, params);

    pcl::transformPointCloud(*object_d, *object_d, getTransform(solution).matrix());


    /*
        Perform random rotation N times and compare results
    */

    std::vector<RandomSampleResult> results;

    

    for(int i = 0; i < 100; ++i)
    {
        std::cout << "STEP " << (i+1) << " /10\n";
        // Rotate 
        auto pt = randomRotationMatrix();
        auto matrix = pt.matrix();
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*object_d, *transformed, matrix);

        auto solution = teaserStep(transformed, scene_d, params);

        results.push_back({.transform = getTransform(solution), .pretransform = pt, .prescore = 0, .score = scoreClouds(*transformed, *scene_d, solution.corrs, getTransform(solution)) });
    }

    auto best_score = std::ranges::min_element(results, [](auto& a, auto& b){ return a.score < b.score; });
    std::cout << "Best by score : " << best_score->prescore << ", " << best_score->score << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::transformPointCloud(*object_d, *aligned, best_score->pretransform.matrix());
    pcl::transformPointCloud(*object_d, *aligned, best_score->transform.matrix());

    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (scene_d, ColorHandlerT (scene_d, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (aligned, ColorHandlerT (aligned, 255.0, 0.0, 0.0), "object_aligned");
    visu.spin ();

}