
#include <gtest/gtest.h>
#include <sleekslam/backend/pose_graph.hpp>

namespace {

    using sfs::GraphPose;
    using sfs::GraphVec6;

    // Plain translation RMSE. Pose 0 is fixed at the origin in both trajectories, so both live in the same frame and no evo-style alignment is needed.
    double translationRmse(const std::vector<GraphPose>& a, const std::vector<GraphPose>& b) {
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            sum += (a[i].translation() - b[i].translation()).squaredNorm();
        return std::sqrt(sum / static_cast<double>(a.size()));
    }

} // namespace

TEST(PoseGraph, LoopClosureRemovesDriftOnSquareTrajectory) {
    constexpr int kSide = 10;      // steps per side
    constexpr int kN = 4 * kSide;

    // True motion: 0.5 m forward per step, 90 deg yaw on the last step of each side.
    // After kN steps the camera is back at the start.
    const GraphPose forward(Sophus::SO3d(), Eigen::Vector3d(0, 0, 0.5));
    const GraphPose turn(
        Sophus::SO3d::exp(Eigen::Vector3d(0, std::numbers::pi / 2, 0)),
        Eigen::Vector3d::Zero());

    std::vector<GraphPose> relTrue;
    for (int i = 0; i < kN; ++i)
        relTrue.push_back(i % kSide == kSide - 1 ? forward * turn : forward);

    std::vector<GraphPose> truth{ GraphPose() };
    for (int i = 0; i + 1 < kN; ++i)
        truth.push_back(truth.back() * relTrue[i]);

    // Biased odometry: every step carries the same small systematic error
    // (5 mm translation, 0.002 rad yaw). Deterministic, so no RNG in CI.
    GraphVec6 drift;
    drift << 0.005, 0, 0, 0, 0.002, 0;
    const GraphPose driftStep = GraphPose::exp(drift);

    std::vector<GraphPose> odom;
    std::vector<GraphPose> deadReckoned{ GraphPose() };
    for (int i = 0; i + 1 < kN; ++i) {
        odom.push_back(relTrue[i] * driftStep);
        deadReckoned.push_back(deadReckoned.back() * odom[i]);
    }

    // Build the graph: initial guesses are the drifted poses, pose 0 is fixed.
    sfs::PoseGraph graph;
    for (int i = 0; i < kN; ++i)
        ASSERT_TRUE(graph.addPose(i, deadReckoned[i], /*fixed=*/i == 0));
    for (int i = 0; i + 1 < kN; ++i)
        ASSERT_TRUE(graph.addSequentialEdge(i, i + 1, odom[i]));
    // Loop closure: place recognition observes the last pose from pose 0.
    ASSERT_TRUE(graph.addLoopEdge(0, kN - 1, truth[kN - 1], 100.0, 1.0));

    ASSERT_GT(graph.optimize(50), 0);

    std::vector<GraphPose> optimized(kN);
    for (int i = 0; i < kN; ++i)
        ASSERT_TRUE(graph.getPose(i, optimized[i]));

    const double before = translationRmse(deadReckoned, truth);
    const double after = translationRmse(optimized, truth);
    const double finalErr = (optimized.back().translation() - truth.back().translation()).norm();

    EXPECT_GT(before, 0.10);          // sanity: the drift is real, test isn't vacuous
    EXPECT_LT(after, 0.06);           // absolute accuracy after closing the loop
    EXPECT_LT(after, before / 3.0);   // and a clear improvement over dead reckoning
    EXPECT_LT(finalErr, 0.02);        // the closed pose lands on the truth
}