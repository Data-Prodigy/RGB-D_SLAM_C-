
#pragma once

#include "../src/pch.h"
#include <sleekslam/core/frame.hpp>
#include <sleekslam/core/residual_common.hpp>

namespace vo {

    // Correspondences failing these checks are rejected.
    struct GeoGate {
        double maxDist = 0.10;       // Meters between moved and matched points
        double minNormalCos = 0.7;   // Minimum normal agreement
    };

    struct GeoSample {
        Vec3 pRef;
        Vec3 nRef;
    };

    // Sparse, spatially distributed geometric sample selection.
    //
    // Instead of scanning every pixel twice, scan a deterministic sparse grid within each cell. If a cell yields too many valid candidates, select
    // approximately evenly spaced entries from that cell.
    [[nodiscard]] inline std::vector<GeoSample> selectGeometricSamples(const PyramidLevel& ref, std::size_t maxSamples)
    {
        if (maxSamples == 0 || ref.w <= 0 || ref.h <= 0 || ref.points.empty() || ref.normals.empty() || ref.points.cols != ref.w || ref.points.rows != ref.h ||
            ref.normals.cols != ref.w || ref.normals.rows != ref.h) {
            return {};
        }

        constexpr int gridCols = 8;
        constexpr int gridRows = 6;
        constexpr int cellCount = gridCols * gridRows;
        constexpr double candidatesPerSample = 7.0;

        std::array<std::vector<GeoSample>, cellCount> cells;

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

                if (quota == 0 || x1 <= x0 || y1 <= y0)
                    continue;

                const double area = static_cast<double>(x1 - x0) * static_cast<double>(y1 - y0);

                const double targetCandidates = std::max(1.0, static_cast<double>(quota) * candidatesPerSample);

                const int step = std::max(1, static_cast<int>(std::sqrt(area / targetCandidates)));

                auto& candidates = cells[static_cast<std::size_t>(cell)];
                candidates.reserve(quota * 8 + 1);

                for (int v = y0; v < y1; v += step) {
                    const cv::Vec3f* pointRow = ref.points.ptr<cv::Vec3f>(v);
                    const cv::Vec3f* normalRow = ref.normals.ptr<cv::Vec3f>(v);

                    for (int u = x0; u < x1; u += step) {
                        const cv::Vec3f& p = pointRow[u];
                        const cv::Vec3f& n = normalRow[u];

                        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || p[2] <= 0.0f) {
                            continue;
                        }

                        if (!std::isfinite(n[0]) || !std::isfinite(n[1]) || !std::isfinite(n[2]) || n.dot(n) < 0.5f) {
                            continue;
                        }

                        candidates.push_back({Vec3(p[0], p[1], p[2]), Vec3(n[0], n[1], n[2])});
                    }
                }
            }
        }

        std::vector<GeoSample> samples;
        samples.reserve(maxSamples);

        for (int i = 0; i < cellCount; ++i) {
            auto& candidates = cells[static_cast<std::size_t>(i)];

            const std::size_t quota = baseQuota + (static_cast<std::size_t>(i) < remainder ? 1 : 0);

            if (quota == 0 || candidates.empty())
                continue;

            if (candidates.size() <= quota) {
                samples.insert(samples.end(), candidates.begin(), candidates.end());
                continue;
            }

            // Select approximately evenly spaced valid candidates from this cell.
            // The multiplication/division formulation avoids a floating-point
            // dependency and keeps selection deterministic.
            for (std::size_t k = 0; k < quota; ++k) {
                const std::size_t index = k * candidates.size() / quota;
                samples.push_back(candidates[index]);
            }
        }

        return samples;
    }

    [[nodiscard]] inline std::optional<Eval> geometricEval(const SE3& T_cr, const GeoSample& s, const PyramidLevel& cur, const GeoGate& gate,bool withJacobian)
    {
        const Vec3 q = T_cr * s.pRef;

        if (!std::isfinite(q.x()) || !std::isfinite(q.y()) || !std::isfinite(q.z()) || q.z() <= 1e-3) {
            return std::nullopt;
        }

        const Pinhole& K = cur.K;

        const int u = static_cast<int>(std::lround(K.fx * q.x() / q.z() + K.cx));
        const int v = static_cast<int>(std::lround(K.fy * q.y() / q.z() + K.cy));

        if (u < 0 || u >= cur.w || v < 0 || v >= cur.h)
            return std::nullopt;

        const cv::Vec3f& Qf = cur.points.at<cv::Vec3f>(v, u);
        const cv::Vec3f& nf = cur.normals.at<cv::Vec3f>(v, u);

        if (!std::isfinite(Qf[0]) || !std::isfinite(Qf[1]) || !std::isfinite(Qf[2]) || Qf[2] <= 0.0f) {
            return std::nullopt;
        }

        if (!std::isfinite(nf[0]) || !std::isfinite(nf[1]) || !std::isfinite(nf[2]) || nf.dot(nf) < 0.5f) {
            return std::nullopt;
        }

        const Vec3 Q(Qf[0], Qf[1], Qf[2]);
        const Vec3 n(nf[0], nf[1], nf[2]);
        const Vec3 diff = q - Q;

        if (diff.norm() > gate.maxDist)
            return std::nullopt;

        if ((T_cr.so3() * s.nRef).dot(n) < gate.minNormalCos)
            return std::nullopt;

        Eval out;
        out.r = n.dot(diff);

        if (withJacobian)
            out.J = poseJacobian(q, n);

        return out;
    }

} // namespace vo
