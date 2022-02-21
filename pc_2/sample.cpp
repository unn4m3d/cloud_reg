#include "../stl_reader.hpp"
#include "../point_sampler.hpp"
#include <pcl/io/pcd_io.h>

int main()
{
    stl_reader::StlMesh<float, unsigned> mesh("../test/model_2.stl");
    auto cloud = clouds::sampleMesh(mesh, 0.2);
    pcl::io::savePCDFile("../test/model_2.pcd", *cloud);
}