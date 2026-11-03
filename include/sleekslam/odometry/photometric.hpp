
#pragma once

#include "../src/pch.h"
#include <sleekslam/core/frame.hpp>
#include <sleekslam/core/residual_common.hpp>


namespace vo {

    struct PhotoSample {
        Vec3 pRef;                  // 3D point in reference camera frame
        double refIntensity{};      // Reference image intensity
        double gradientMagnitude{}; // Reference gradient magnitude
    };

    // Semi-dense sample selection.
    //
    // The image is divided into grid cells. Within each cell, inspect a
    // sparse, deterministic pixel grid and retain the strongest gradients.
    // This trades some edge coverage for lower sample-selection cost.
    [[nodiscard]] inline std::vector<PhotoSample> selectPhotometricSamples(const PyramidLevel& ref, double minGradient, std::size_t maxSamples){
        if (maxSamples == 0 || ref.w < 3 || ref.h < 3 || ref.points.empty() || ref.gray.empty() || ref.gradX.empty() || ref.gradY.empty()) {
            return {};
        }

        if (ref.points.cols != ref.w || ref.points.rows != ref.h || ref.gray.cols != ref.w || ref.gray.rows != ref.h || ref.gradX.cols != ref.w || ref.gradX.rows != ref.h ||
            ref.gradY.cols != ref.w || ref.gradY.rows != ref.h) {
            return {};
        }

        struct Candidate {
            PhotoSample sample;
            double gradientSquared{};
        };

        constexpr int gridCols = 8;
        constexpr int gridRows = 6;
        constexpr int cellCount = gridCols * gridRows;

        // Inspect approximately this many candidate pixels per retained sample.
        constexpr double candidatesPerSample = 7.0;

        std::array<std::vector<Candidate>, cellCount> cells;
        const double thresholdSquared =
            minGradient > 0.0 ? minGradient * minGradient : 0.0;

        const std::size_t baseQuota = maxSamples / cellCount;
        const std::size_t remainder = maxSamples % cellCount;

        for (int row = 0; row < gridRows; ++row) {
            const int y0 = row * ref.h / gridRows;
            const int y1 = (row + 1) * ref.h / gridRows;

            for (int col = 0; col < gridCols; ++col) {
                const int x0 = col * ref.w / gridCols;
                const int x1 = (col + 1) * ref.w / gridCols;
                const int cell = row * gridCols + col;

                const std::size_t quota = baseQuota + (static_cast<std::size_t>(cell) < remainder ? 1 : 0);

                if (quota == 0 || x1 - x0 < 3 || y1 - y0 < 3)
                    continue;

                // Estimate a stride that gives several candidates per desired sample, while avoiding a full-resolution scan.
                const double area = static_cast<double>(x1 - x0) * static_cast<double>(y1 - y0);

                const double targetCandidates = std::max(1.0, static_cast<double>(quota) * candidatesPerSample);

                const int step = std::max(1, static_cast<int>(std::sqrt(area / targetCandidates)));

                auto& candidates = cells[static_cast<std::size_t>(cell)];
                candidates.reserve(quota * 8 + 1);

                const int beginY = std::max(1, y0);
                const int endY = std::min(ref.h - 1, y1);
                const int beginX = std::max(1, x0);
                const int endX = std::min(ref.w - 1, x1);

                for (int v = beginY; v < endY; v += step) {
                    const cv::Vec3f* pointRow = ref.points.ptr<cv::Vec3f>(v);
                    const float* gxRow = ref.gradX.ptr<float>(v);
                    const float* gyRow = ref.gradY.ptr<float>(v);
                    const float* grayRow = ref.gray.ptr<float>(v);

                    for (int u = beginX; u < endX; u += step) {
                        const cv::Vec3f& p = pointRow[u];

                        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || p[2] <= 0.0f) {
                            continue;
                        }

                        const float gx = gxRow[u];
                        const float gy = gyRow[u];
                        const float intensity = grayRow[u];

                        if (!std::isfinite(gx) || !std::isfinite(gy) || !std::isfinite(intensity)) {
                            continue;
                        }

                        const double g2 = static_cast<double>(gx) * gx + static_cast<double>(gy) * gy;

                        if (g2 < thresholdSquared)
                            continue;

                        candidates.push_back({ {
                                Vec3(p[0], p[1], p[2]), static_cast<double>(intensity), std::sqrt(g2)}, g2 });
                    }
                }
            }
        }

        std::vector<PhotoSample> samples;
        samples.reserve(maxSamples);

        for (int i = 0; i < cellCount; ++i) {
            auto& candidates = cells[static_cast<std::size_t>(i)];

            const std::size_t quota =
                baseQuota +
                (static_cast<std::size_t>(i) < remainder ? 1 : 0);

            if (quota == 0 || candidates.empty())
                continue;

            if (candidates.size() > quota) {std::nth_element(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(quota), candidates.end(),
                    [](const Candidate& a, const Candidate& b) {
                        return a.gradientSquared > b.gradientSquared;
                    });

                candidates.resize(quota);
            }

            for (const Candidate& candidate : candidates)
                samples.push_back(candidate.sample);
        }

        return samples;
    }

    // Interpolated current-image data.
    struct PhotoInterpolated {
        float intensity{};
        float gx{};
        float gy{};
    };

    // Bilinearly interpolate intensity and, optionally, both gradients.
    // Interpolation weights are calculated once and reused.
    [[nodiscard]] inline std::optional<PhotoInterpolated> bilinearPhoto(const PyramidLevel& cur, double u, double v, bool withGradients) noexcept{
        if (!std::isfinite(u) || !std::isfinite(v) || cur.gray.empty() || cur.gray.cols < 2 || cur.gray.rows < 2 ||
            u < 0.0 || v < 0.0 || u >= static_cast<double>(cur.gray.cols - 1) || v >= static_cast<double>(cur.gray.rows - 1)) {
            return std::nullopt;
        }

        if (withGradients && (cur.gradX.empty() || cur.gradY.empty() || cur.gradX.size() != cur.gray.size() || cur.gradY.size() != cur.gray.size())) {
            return std::nullopt;
        }

        const int x = static_cast<int>(u);
        const int y = static_cast<int>(v);

        const float dx = static_cast<float>(u - x);
        const float dy = static_cast<float>(v - y);

        const float ax = 1.0f - dx;
        const float ay = 1.0f - dy;

        const float w00 = ax * ay;
        const float w10 = dx * ay;
        const float w01 = ax * dy;
        const float w11 = dx * dy;

        const auto interpolate =
            [&](const cv::Mat& img) noexcept -> float {
            const float* row0 = img.ptr<float>(y);
            const float* row1 = img.ptr<float>(y + 1);

            return w00 * row0[x] + w10 * row0[x + 1] + w01 * row1[x] + w11 * row1[x + 1]; };

        PhotoInterpolated out;
        out.intensity = interpolate(cur.gray);

        if (!std::isfinite(out.intensity))
            return std::nullopt;

        if (withGradients) {
            out.gx = interpolate(cur.gradX);
            out.gy = interpolate(cur.gradY);

            if (!std::isfinite(out.gx) || !std::isfinite(out.gy)) {
                return std::nullopt;
            }
        }

        return out;
    }

    // Photometric residual:
    // r = I_cur(project(T_cr * p_ref)) - I_ref.
    //
    // nullopt means this sample is invalid for the evaluation.
    [[nodiscard]] inline std::optional<Eval> photometricEval(const SE3& T_cr, const PhotoSample& s, const PyramidLevel& cur,
        bool withJacobian)
    {
        const Vec3 q = T_cr * s.pRef;

        if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) || q.z() <= 1e-3) {
            return std::nullopt;
        }

        const Pinhole& K = cur.K;
        const double iz = 1.0 / q.z();

        const double u = K.fx * q.x() * iz + K.cx;
        const double v = K.fy * q.y() * iz + K.cy;

        const auto pixel = bilinearPhoto(cur, u, v, withJacobian);

        if (!pixel)
            return std::nullopt;

        Eval out;
        out.r = static_cast<double>(pixel->intensity) - s.refIntensity;

        if (!std::isfinite(out.r))
            return std::nullopt;

        if (withJacobian) {const Vec3 gq(pixel->gx * K.fx * iz, pixel->gy * K.fy * iz, -(pixel->gx * K.fx * q.x() + pixel->gy * K.fy * q.y()) * iz * iz);

            out.J = poseJacobian(q, gq);
        }

        return out;
    }

} // namespace vo
