
#pragma once

#include "../src/pch.h"
#include <sleekslam/core/pose.hpp>
namespace vo {

    // One residual value plus its Jacobian with respect to the six
    // left-perturbation parameters.
    struct Eval {
        double r{};
        JacRow J = JacRow::Zero();
    };

    // Jacobian for q' = q + rho + phi x q.
    //
    // Parameter ordering: translation first, rotation second.
    // J = [ g^T, (q x g)^T ].
    [[nodiscard]] inline JacRow poseJacobian(const Vec3& q, const Vec3& g){
        JacRow J;
        J.leftCols<3>() = g.transpose();
        J.rightCols<3>() = q.cross(g).transpose();
        return J;
    }

    // Bilinear lookup in a CV_32F image at subpixel coordinates
    // (u = column, v = row).
    [[nodiscard]] inline std::optional<float> bilinear(const cv::Mat& img, double u,double v) noexcept{
        if (img.empty() || img.type() != CV_32FC1 || !std::isfinite(u) || !std::isfinite(v) || img.cols < 2 ||img.rows < 2 ||
            u < 0.0 || v < 0.0 || u >= static_cast<double>(img.cols - 1) || v >= static_cast<double>(img.rows - 1)) {
            return std::nullopt;
        }

        const int x = static_cast<int>(u);
        const int y = static_cast<int>(v);

        const float dx = static_cast<float>(u - x);
        const float dy = static_cast<float>(v - y);

        const float* r0 = img.ptr<float>(y);
        const float* r1 = img.ptr<float>(y + 1);

        const float value = (1.0f - dx) * (1.0f - dy) * r0[x] + dx * (1.0f - dy) * r0[x + 1] + (1.0f - dx) * dy * r1[x] + dx * dy * r1[x + 1];

        if (!std::isfinite(value)) return std::nullopt;

        return value;
    }

    // Keep at most maxCount elements by taking every k-th element.
    // Deterministic and approximately uniform over the input sequence.
    template <class T>
    [[nodiscard]] inline std::vector<T> strideSubsample(std::vector<T> values, std::size_t maxCount){
        if (maxCount == 0 || values.size() <= maxCount) return values;

        const std::size_t step =(values.size() + maxCount - 1) / maxCount;

        std::vector<T> out;
        out.reserve((values.size() + step - 1) / step);

        for (std::size_t i = 0; i < values.size(); i += step)
            out.push_back(std::move(values[i]));

        return out;
    }

} // namespace vo
