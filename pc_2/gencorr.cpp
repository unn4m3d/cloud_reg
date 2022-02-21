#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/io/pcd_io.h>
#include "json.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

std::vector<Eigen::Vector2f> projected;

void project(pcl::visualization::PCLVisualizer& v, const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    std::vector<pcl::visualization::Camera> cameras;
    v.getCameras(cameras);
    auto cam = cameras[0];
    projected.clear();

    Eigen::Matrix4d proj, view;
    cam.computeProjectionMatrix(proj);
    cam.computeViewMatrix(view);
    auto cmat = proj*view;

    for(size_t i = 0; i < cloud.size(); ++i)
    {
        auto& pt = cloud.at(i);
        Eigen::Vector4d evt;
        if(pt.x != 0 || pt.y != 0 || pt.z != 0)
        {
            cam.cvtWindowCoordinates(pt, evt, cmat);
            projected.emplace_back(evt.x(), evt.y());
        }
    }
}

using PtIndex = size_t;
using PtSet = std::vector<PtIndex>;
using Correspondence = std::pair<PtSet, PtSet>;
using CorrSet = std::vector<Correspondence>;

PtIndex nearest(float x, float y)
{
    Eigen::Vector2f v(x, y);
    return std::distance(projected.cbegin(), std::min_element(projected.cbegin(), projected.cend(),\
        [&](const auto& vec1, const auto& vec2)
        {
            return (v - vec1).norm() < (v - vec2).norm();
        }
    ));
}

CorrSet set;

template<typename T>
T& createOrGet(std::vector<T>& vec, size_t idx)
{
    while(vec.size() < idx + 1) vec.emplace_back();
    return vec[idx];
}

void upd(auto& visu, bool first, int idx)
{
    std::string txt{ first ? "first, " : "second, " };
    visu.updateText(txt + std::to_string(idx), 0, 20, "text");
}

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        std::cout << "Zalupa i pupa\n";
        return 1;
    }

    std::string scene_p{ argv[1] }, object_p{ argv[2] }, res_p{ argv[3] };

    pcl::PointCloud<pcl::PointXYZ>::Ptr 
        scene(new pcl::PointCloud<pcl::PointXYZ>),
        object(new pcl::PointCloud<pcl::PointXYZ>);

    if(pcl::io::loadPCDFile(scene_p, *scene) < 0 || pcl::io::loadPCDFile(object_p, *object) < 0)
    {
        std::cout << "Full blown zalupa\n";
        return 2;
    }

    bool spin = true;

    pcl::visualization::PCLVisualizer visu;
    visu.addPointCloud(scene, "cloud");
    visu.addText("first, 0", 0, 15, "text");
    project(visu, *scene);

    auto first = true;
    auto idx = 0;

    auto fire = false;

    auto&& mh = visu.registerMouseCallback(
        [&](const auto& evt)
        { 
            if(!fire || 
                evt.getButton() != pcl::visualization::MouseEvent::MouseButton::LeftButton ||
                evt.getType() != pcl::visualization::MouseEvent::Type::MouseButtonRelease) return;
            auto& s = createOrGet(set, idx);
            auto& sp = first ? s.first : s.second;
            sp.push_back(nearest(evt.getX(), evt.getY()));
            std::cout << "Got one" << std::endl;
        }
    );

    auto&& kh = visu.registerKeyboardCallback(
        [&](const auto& evt)
        {
            if(evt.keyDown())
            {
                std::cout << evt.getKeySym() << std::endl;
                if(evt.getKeyCode() == ' ')
                {
                    idx++;
                    upd(visu, first, idx);
                }
                else if(evt.getKeySym() == "Return")
                {
                    idx = 0;
                    first = !first;
                    visu.updatePointCloud(first ? scene : object, "cloud");
                    project(visu, (first ? *scene : *object));
                    upd(visu, first, idx);
                    fire = false;
                }
                else if(evt.getKeyCode() == 's')
                {
                    fire = !fire;
                    project(visu, (first ? *scene : *object));
                    std::cout << "Zalupa" << std::endl;
                }
            }
        }
    );

    while (!visu.wasStopped ())
    {
        visu.spinOnce (100);
        std::this_thread::sleep_for(100ms);
    }

    nlohmann::json j;

    for(const auto& corr : set)
    {
        nlohmann::json corr_j_first, corr_j_second;

        for(const auto& cf : corr.first)
        {
            corr_j_first.push_back(cf);
        }

        for(const auto& cs : corr.second)
        {
            corr_j_second.push_back(cs);
        }

        nlohmann::json ca;
        ca.push_back(corr_j_first);
        ca.push_back(corr_j_second);

        j.push_back(ca);
    }

    std::ofstream f(res_p);
    f << j;

}
