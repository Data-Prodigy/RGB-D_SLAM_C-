
#pragma once

#include "../src/pch.h"
#include <sleekslam/backend/g20_types.hpp>
#include <sleekslam/backend/robust.hpp>


namespace vo {

    // For each pyramid level, coarsest first:
    // 1. Select samples from the reference pyramid level.
    // 2. Build a g2o graph with one pose vertex and robustified unary edges.
    // 3. Optimize T_cr and pass the estimate to the next finer level.
    //
    // Convention: T_cr maps reference-camera coordinates to current-camera
    // coordinates: p_cur = T_cr * p_ref.

    struct OdometryConfig {
        // Pyramid / iterations (index = level, 0 = finest).
        std::vector<int> iterationsPerLevel{ 1, 2, 3, 4 };

        // Sampling.
        // Used for every level unless maxSamplesPerLevelByLevel is populated.
        std::size_t maxSamplesPerLevel = 7500;

        // Optional per-level budgets (index 0 = finest).
        // Example: {600, 800, 1000, 1200}.
        // Leave empty to use maxSamplesPerLevel at every level.
        std::vector<std::size_t> maxSamplesPerLevelByLevel{};

        double minGradient = 14.0;
        std::size_t minEdges = 130;

        // Combined cost:
        // sum_photo rho(e_p^2 / sigma_p^2)
        // + sum_geo rho(e_g^2 / sigma_g^2).
        bool usePhotometric = true;
        bool useGeometric = true;
        double sigmaPhoto = 15.0;
        double sigmaGeo = 0.02;

        RobustConfig robust;
        GeoGate gate;
    };

    struct LevelStats {
        int level{};
        std::size_t photoEdges{};
        std::size_t geoEdges{};

        double chi2Start{};
        double chi2End{};

        // Timing breakdown, in milliseconds.
        double photoSelectionMs{};
        double geoSelectionMs{};
        double sampleSelectionMs{};
        double graphConstructionMs{};
        double optimizerSetupMs{};
        double optimizeMs{};
        double levelTotalMs{};
    };

    struct OdometryResult {
        SE3 T_cr;
        bool ok = false;
        std::vector<LevelStats> stats;

        // Total time spent inside estimate(), in milliseconds.
        double totalMs{};
    };

    class RgbdOdometry {
    public:
        explicit RgbdOdometry(OdometryConfig cfg = {})
            : cfg_(std::move(cfg))
        {
        }

        [[nodiscard]]
        const OdometryConfig& config() const noexcept
        {
            return cfg_;
        }

        [[nodiscard]]
        OdometryResult estimate(const Frame& ref, const Frame& cur, const SE3& initialGuess) const{
            using Clock = std::chrono::steady_clock;
            using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>>;
            using LinearSolver = g2o::LinearSolverDense<BlockSolver::PoseMatrixType>;

            const auto estimateStart = Clock::now();

            const auto elapsedMs = [](Clock::time_point start, Clock::time_point end) {
                    return std::chrono::duration<double, std::milli>(
                        end - start).count();
                };

            OdometryResult out;
            out.T_cr = initialGuess;

            const int numLevels = static_cast<int>(
                std::min(ref.levels.size(), cur.levels.size()));

            for (int l = numLevels - 1; l >= 0; --l) {
                const auto levelStart = Clock::now();

                const PyramidLevel& R = ref.levels[static_cast<std::size_t>(l)];
                const PyramidLevel& C = cur.levels[static_cast<std::size_t>(l)];

                LevelStats st{};
                st.level = l;

                const std::size_t sampleBudget = samplesForLevel(l);

                // -------------------------------------------------
                // 1. Sample selection
                // -------------------------------------------------
                const auto photoStart = Clock::now();

                const auto photo = cfg_.usePhotometric ? selectPhotometricSamples(R, cfg_.minGradient, sampleBudget) : std::vector<PhotoSample>{};

                st.photoSelectionMs = elapsedMs(photoStart, Clock::now());

                const auto geoStart = Clock::now();

                const auto geo = cfg_.useGeometric? selectGeometricSamples(R, sampleBudget) : std::vector<GeoSample>{};

                st.geoSelectionMs = elapsedMs(geoStart, Clock::now());

                st.sampleSelectionMs = st.photoSelectionMs + st.geoSelectionMs;

                st.photoEdges = photo.size();
                st.geoEdges = geo.size();

                // Preserve the existing behavior: skip levels with too
                // few total residual edges.
                if (photo.size() + geo.size() < cfg_.minEdges) {
                    st.levelTotalMs = elapsedMs(levelStart, Clock::now());

                    out.stats.push_back(st);

                    if (l == 0)
                        out.ok = false;

                    continue;
                }

                // -------------------------------------------------
                // 2. Graph construction
                // -------------------------------------------------
                const auto graphStart = Clock::now();

                g2o::SparseOptimizer optimizer;

                optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(
                        std::make_unique<BlockSolver>(std::make_unique<LinearSolver>())));

                optimizer.setVerbose(false);

                auto* vertex = new VertexPose;
                vertex->setId(0);
                vertex->setEstimate(out.T_cr);
                optimizer.addVertex(vertex);

                Eigen::Matrix<double, 1, 1> infoPhoto;
                Eigen::Matrix<double, 1, 1> infoGeo;

                infoPhoto(0, 0) = 1.0 / (cfg_.sigmaPhoto * cfg_.sigmaPhoto);

                infoGeo(0, 0) = 1.0 / (cfg_.sigmaGeo * cfg_.sigmaGeo);

                int edgeId = 0;

                for (const PhotoSample& s : photo) {
                    auto* e = new EdgePhotometric(&C, s.pRef);

                    e->setId(edgeId++);
                    e->setVertex(0, vertex);
                    e->setMeasurement(s.refIntensity);
                    e->setInformation(infoPhoto);

                    if (auto* k = makeRobustKernel(cfg_.robust))
                        e->setRobustKernel(k);

                    optimizer.addEdge(e);
                }

                for (const GeoSample& s : geo) {
                    auto* e = new EdgeGeometric(&C, s, cfg_.gate);

                    e->setId(edgeId++);
                    e->setVertex(0, vertex);
                    e->setMeasurement(0.0);
                    e->setInformation(infoGeo);

                    if (auto* k = makeRobustKernel(cfg_.robust))
                        e->setRobustKernel(k);

                    optimizer.addEdge(e);
                }

                st.graphConstructionMs = elapsedMs(graphStart, Clock::now());

                // -------------------------------------------------
                // 3. Optimizer initialization and initial cost
                // -------------------------------------------------
                const auto setupStart = Clock::now();

                optimizer.initializeOptimization();
                optimizer.computeActiveErrors();

                st.chi2Start = optimizer.activeChi2();

                st.optimizerSetupMs =
                    elapsedMs(setupStart, Clock::now());

               
                // 4. Nonlinear optimization
                
                const auto optimizeStart = Clock::now();

                optimizer.optimize(iterationsFor(l));

                st.optimizeMs = elapsedMs(optimizeStart, Clock::now());

                
                // 5. Save the level result and timing
              
                st.chi2End = optimizer.activeChi2();
                out.T_cr = vertex->estimate();

                st.levelTotalMs = elapsedMs(levelStart, Clock::now());

                out.stats.push_back(st);

                // Preserve the existing tracking-success semantics.
                if (l == 0)
                    out.ok = true;
            }

            out.totalMs = elapsedMs(estimateStart, Clock::now());

            return out;
        }

    private:
        [[nodiscard]]
        std::size_t samplesForLevel(int level) const
        {
            if (level >= 0 &&
                static_cast<std::size_t>(level) < cfg_.maxSamplesPerLevelByLevel.size()) {
                return cfg_.maxSamplesPerLevelByLevel[
                    static_cast<std::size_t>(level)];
            }

            return cfg_.maxSamplesPerLevel;
        }

        [[nodiscard]] int iterationsFor(int level) const{
            const auto& v = cfg_.iterationsPerLevel;

            if (v.empty())
                return 10;

            return v[std::min(static_cast<std::size_t>(std::max(level, 0)),v.size() - 1)];
        }

        OdometryConfig cfg_;
    };

} // namespace vo
