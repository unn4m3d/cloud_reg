#pragma once
#include <Eigen/Core>

namespace clouds
{
    template<size_t H, size_t W, typename Scalar = double>
    auto parse_matrix(std::istream& str)
    {
        Eigen::Matrix<Scalar, H, W> mat;
        for(size_t i = 0; i < H; ++i)
        {
            for(size_t j = 0; j < W; ++j)
            {
                str >> mat(i, j);
            }
        }
        return mat;
    }

    template<size_t H, size_t W, typename Scalar>
    void serialize_matrix(const Eigen::Matrix<Scalar, H, W>& m, std::ostream& str)
    {
        for(size_t i = 0; i < H; ++i)
        {
            for(size_t j = 0; j < W; ++j)
            {
                str << m(i, j);
            }
            str << std::endl;
        }
    }
}