#include "ply_loader.hpp"
#include <vector>
#include "happly.hpp"
#include <regex>
#include <charconv>
#include <exception>

void clouds::ply_loader::load(const std::string& filename)
{
    happly::PLYData input(filename);

    auto vertices = input.getVertexPositions();

    auto obj_info = input.objInfoComments[0];
    size_t w = 1, h = vertices.size();
    const std::string prefix = "Photoneo PLY PointCloud";

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