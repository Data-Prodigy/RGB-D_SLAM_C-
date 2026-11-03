#pragma once

// Experimental CUDA residual evaluator + CPU 6x6 Levenberg-style pose solve.
// This is a separate path; keep RgbdOdometry (g2o) as the correctness baseline.
#include <sleekslam/odometry/odometry.hpp>
#include <sleekslam/odometry/residuals_cuda.hpp>

namespace vo {

    class RgbdOdometryCuda {
    public:
        explicit RgbdOdometryCuda(OdometryConfig cfg = {}) : cfg_(std::move(cfg)) {}

        [[nodiscard]] const OdometryConfig& config() const noexcept { return cfg_; }

        [[nodiscard]] OdometryResult estimate(const Frame& ref, const Frame& cur,const SE3& initialGuess) const {
            using Clock = std::chrono::steady_clock;
            const auto begin = Clock::now();
            auto ms = [](Clock::time_point a, Clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
                };
            OdometryResult out; out.T_cr = initialGuess;
            const int levels = static_cast<int>(std::min(ref.levels.size(), cur.levels.size()));
            for (int l = levels - 1; l >= 0; --l) {
                const auto levelBegin = Clock::now();
                const PyramidLevel& R = ref.levels[static_cast<std::size_t>(l)];
                const PyramidLevel& C = cur.levels[static_cast<std::size_t>(l)];
                LevelStats st{}; st.level = l;
                const std::size_t budget = samplesForLevel(l);

                auto t = Clock::now();
                const auto photo = cfg_.usePhotometric ? selectPhotometricSamples(R, cfg_.minGradient, budget) : std::vector<PhotoSample>{};
                st.photoSelectionMs = ms(t, Clock::now());
                t = Clock::now();
                const auto geo = cfg_.useGeometric ? selectGeometricSamples(R, budget) : std::vector<GeoSample>{};
                st.geoSelectionMs = ms(t, Clock::now()); st.sampleSelectionMs = st.photoSelectionMs + st.geoSelectionMs;
                st.photoEdges = photo.size(); st.geoEdges = geo.size();
                if (photo.size() + geo.size() < cfg_.minEdges) {
                    st.levelTotalMs = ms(levelBegin, Clock::now()); out.stats.push_back(st);
                    if (l == 0) out.ok = false;
                    continue;
                }

                std::vector<cuda_backend::PhotoSample> pSamples; pSamples.reserve(photo.size());
                for (const auto& s : photo) pSamples.push_back({ {s.pRef.x(),s.pRef.y(),s.pRef.z()},s.refIntensity });
                std::vector<cuda_backend::GeoSample> gSamples; gSamples.reserve(geo.size());
                for (const auto& s : geo) gSamples.push_back({ {s.pRef.x(),s.pRef.y(),s.pRef.z()}, {s.nRef.x(),s.nRef.y(),s.nRef.z()} });

                // The .cu API expects contiguous CV_32F arrays. Clone only if needed.
                cv::Mat gray = C.gray.isContinuous() ? C.gray : C.gray.clone();
                cv::Mat gx = C.gradX.isContinuous() ? C.gradX : C.gradX.clone();
                cv::Mat gy = C.gradY.isContinuous() ? C.gradY : C.gradY.clone();
                cv::Mat points = C.points.isContinuous() ? C.points : C.points.clone();
                cv::Mat normals = C.normals.isContinuous() ? C.normals : C.normals.clone();
                if (gray.type() != CV_32FC1 || gx.type() != CV_32FC1 || gy.type() != CV_32FC1 ||
                    points.type() != CV_32FC3 || normals.type() != CV_32FC3)
                    throw std::runtime_error("CUDA odometry: unexpected PyramidLevel matrix type");

                cuda_backend::Camera K{ C.K.fx,C.K.fy,C.K.cx,C.K.cy };
                std::unique_ptr<cuda_backend::LevelHandle, void(*)(cuda_backend::LevelHandle*)> gpu(
                    cuda_backend::createLevel(C.w, C.h, K, gray.ptr<float>(), gx.ptr<float>(), gy.ptr<float>(),
                        reinterpret_cast<const float*>(points.ptr<cv::Vec3f>()),
                        reinterpret_cast<const float*>(normals.ptr<cv::Vec3f>()),
                        pSamples.data(), pSamples.size(), gSamples.data(), gSamples.size()),
                    cuda_backend::destroyLevel);
                st.graphConstructionMs = ms(levelBegin, Clock::now()) - st.sampleSelectionMs;

                const double infoP = 1.0 / (cfg_.sigmaPhoto * cfg_.sigmaPhoto);
                const double infoG = 1.0 / (cfg_.sigmaGeo * cfg_.sigmaGeo);
                const int robustKind = cfg_.robust.kind == RobustKind::None ? 0 :
                    (cfg_.robust.kind == RobustKind::Huber ? 1 : 2);
                double lambda = 1e-3;
                auto evaluateAt = [&](const SE3& pose, bool withJ, Eigen::Matrix<double, 6, 6>* H,
                    Eigen::Matrix<double, 6, 1>* b)->double {
                        const auto gpuPose = packPose(pose);
                        double hRaw[36]{}, bRaw[6]{}, cost = 0.0;
                        cuda_backend::evaluateNormalEquations(gpu.get(), gpuPose,
                            { cfg_.gate.maxDist,cfg_.gate.minNormalCos }, withJ ? 1 : 0,
                            infoP, infoG, robustKind, cfg_.robust.delta, hRaw, bRaw, &cost);
                        if (H) for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) (*H)(r, c) = hRaw[r * 6 + c];
                        if (b) for (int i = 0; i < 6; ++i) (*b)[i] = bRaw[i];
                        return cost;
                    };

                Eigen::Matrix<double, 6, 6> H;
                Eigen::Matrix<double, 6, 1> b;
                st.optimizerSetupMs = 0.0;
                st.chi2Start = evaluateAt(out.T_cr, true, &H, &b);
                st.optimizeMs = 0.0;
                for (int iter = 0; iter < iterationsFor(l); ++iter) {
                    auto optStart = Clock::now();
                    // Recompute linearization at the current accepted pose.
                    const double currentCost = evaluateAt(out.T_cr, true, &H, &b);
                    Eigen::Matrix<double, 6, 6> Hd = H;
                    for (int d = 0; d < 6; ++d) Hd(d, d) += lambda * std::max(H(d, d), 1e-9);
                    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver(Hd);
                    if (solver.info() != Eigen::Success) { lambda = std::min(lambda * 10.0, 1e12); st.optimizeMs += ms(optStart, Clock::now()); continue; }
                    Vec6 delta = solver.solve(-b);
                    if (solver.info() != Eigen::Success || !delta.allFinite()) { lambda = std::min(lambda * 10.0, 1e12); st.optimizeMs += ms(optStart, Clock::now()); continue; }
                    const SE3 trial = perturbLeft(out.T_cr, delta);
                    const double trialCost = evaluateAt(trial, false, nullptr, nullptr);
                    if (std::isfinite(trialCost) && trialCost < currentCost) {
                        out.T_cr = trial; lambda = std::max(lambda * 0.5, 1e-9);
                    }
                    else lambda = std::min(lambda * 10.0, 1e12);
                    st.optimizeMs += ms(optStart, Clock::now());
                    if (delta.norm() < 1e-7) break;
                }
                st.chi2End = evaluateAt(out.T_cr, false, nullptr, nullptr);
                st.levelTotalMs = ms(levelBegin, Clock::now()); out.stats.push_back(st);
                if (l == 0) out.ok = true;
            }
            out.totalMs = ms(begin, Clock::now()); return out;
        }

    private:
        static cuda_backend::Pose packPose(const SE3& T) {
            cuda_backend::Pose p{}; const auto R = T.rotationMatrix(); const auto t = T.translation();
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) p.R[r * 3 + c] = R(r, c);
            for (int i = 0; i < 3; ++i) p.t[i] = t[i]; return p;
        }
        [[nodiscard]] std::size_t samplesForLevel(int l) const {
            if (l >= 0 && static_cast<std::size_t>(l) < cfg_.maxSamplesPerLevelByLevel.size()) return cfg_.maxSamplesPerLevelByLevel[static_cast<std::size_t>(l)];
            return cfg_.maxSamplesPerLevel;
        }
        [[nodiscard]] int iterationsFor(int l) const {
            if (cfg_.iterationsPerLevel.empty()) return 10;
            return cfg_.iterationsPerLevel[std::min(static_cast<std::size_t>(std::max(l, 0)), cfg_.iterationsPerLevel.size() - 1)];
        }
        OdometryConfig cfg_;
    };

} // namespace vo
