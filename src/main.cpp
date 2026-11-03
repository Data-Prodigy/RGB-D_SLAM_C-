
// main.cpp -- RGB-D VO + GPU TSDF fusion + FBoW loop closure + PLY export.

#include "../src/pch.h"
#include <sleekslam/core/frame.hpp>
#include <sleekslam/backend/jacobian_check.hpp>
#include <sleekslam/odometry/odometry.hpp>
#include <sleekslam/backend/safe_queue.hpp>
#include <sleekslam/viewer/point_cloud_viewer.hpp>
#include <sleekslam/backend/pose_graph.hpp>
#include <sleekslam/mapping/fusion.hpp>
#include <sleekslam/backend/loop_closure_rgbd.hpp>
#include <sleekslam/backend/keyframe_record.hpp>
#include <sleekslam/odometry/odometry_cuda.hpp>

namespace {

    using namespace vo;
    using SteadyClock = std::chrono::steady_clock;

    struct Options {
        std::filesystem::path dir = "C:/tum_ready1";
        std::filesystem::path frameDir = "C:/dev/tum_ready1";
        int firstFrame = 0;
        int maxFrames = 0;
        int keyframeEvery = 5;
        bool selfTest = false;
        bool noViewer = false;
    };

    [[nodiscard]] Options parseArgs(std::span<char* const> args) {
        Options o;

        for (std::size_t i = 1; i < args.size(); ++i) {
            const std::string_view a = args[i];
            const bool hasValue = i + 1 < args.size();

            if (a == "--selftest") {
                o.selfTest = true;
            }
            else if (a == "--no-viewer") {
                o.noViewer = true;
            }
            else if (a == "--dir" && hasValue) {
                o.dir = args[++i];
            }
            else if (a == "--frames-dir" && hasValue) {
                o.frameDir = args[++i];
            }
            else if (a == "--first" && hasValue) {
                o.firstFrame = std::atoi(args[++i]);
            }
            else if (a == "--max-frames" && hasValue) {
                o.maxFrames = std::atoi(args[++i]);
            }
            else if (a == "--keyframe-every" && hasValue) {
                o.keyframeEvery = std::max(1, std::atoi(args[++i]));
            }
            else {
                std::println(stderr, "ignoring unknown argument '{}'", a);
            }
        }

        return o;
    }

    struct RawFrame {
        cv::Mat rgb;
        cv::Mat depth;
    };

    [[nodiscard]] std::optional<RawFrame> readRawFrame(const std::filesystem::path& frameDir, int index){
        RawFrame f;

        f.depth = cv::imread((frameDir / std::format("frame_{:04d}.png", index)).string(), cv::IMREAD_UNCHANGED);

        f.rgb = cv::imread((frameDir / std::format("frame_{:04d}_rgb.png", index)).string(), cv::IMREAD_COLOR);

        if (f.depth.empty() || f.rgb.empty())
            return std::nullopt;

        return f;
    }

    [[nodiscard]] sfs::Pose4f toPose4f(const SE3& T_wc) {
        sfs::Pose4f out{};
        const Eigen::Matrix4d M = T_wc.matrix();

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out.m[r * 4 + c] = static_cast<float>(M(r, c));
            }
        }

        return out;
    }

    struct TrajectorySample {
        double timestamp = 0;
        int graphId = -1;
        SE3 originalPose;
        bool hasGraphPose = false;
    };

    // Calculate a percentile using linear interpolation between sorted samples.
    // Input and output units are milliseconds.
    [[nodiscard]] double percentileMs( std::vector<double> values, double percentile){
        if (values.empty())
            return 0.0;

        std::sort(values.begin(), values.end());

        const double position = std::clamp(percentile, 0.0, 1.0) * static_cast<double>(values.size() - 1);

        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, values.size() - 1);
        const double fraction = position - static_cast<double>(lower);

        return values[lower] * (1.0 - fraction) + values[upper] * fraction;}

    [[nodiscard]] Result<void> trackSequence(const Options& opt, const CameraParams& P, const UndistortMaps& maps, TrajectoryViewer* viewer, std::stop_token stop){
        const PyramidConfig pyramidCfg{};
        const RgbdOdometryCuda odometry{ OdometryConfig{} };

        // ------------------------------------------------------------
        // Loop closure detector and vocabulary
        // ------------------------------------------------------------

        sfs::RgbdLoopConfig loopCfg;

        const std::string vocabPath = "C:/dev/RGB-D_SLAM_C++/RGB-D_SLAM_C++/config/orb_mur.fbow";
        sfs::RgbdLoopClosureDetector loopDetector(vocabPath, loopCfg);

        // ------------------------------------------------------------
        // TSDF configuration
        // ------------------------------------------------------------

        sfs::FusionConfig fusionCfg;

        fusionCfg.volume.nx = 512;
        fusionCfg.volume.ny = 512;
        fusionCfg.volume.nz = 512;
        fusionCfg.volume.voxelSize = 0.015f;
        fusionCfg.volume.truncation = 0.045f;
        fusionCfg.volume.originX = -5.12f;
        fusionCfg.volume.originY = -5.12f;
        fusionCfg.volume.originZ = 0.0f;

        fusionCfg.keyframes.minTranslation = 0.015f;
        fusionCfg.keyframes.minRotationRadians = 0.0349066f;
        fusionCfg.keyframes.maxFrameGap = 30;

        fusionCfg.maxMeshTriangles = 10000;

        sfs::FusionMapper mapper(fusionCfg);

        std::println("TSDF device: {}", mapper.deviceSummary());

        // ------------------------------------------------------------
        // Pose graph and mapping queue
        // ------------------------------------------------------------

        sfs::PoseGraph graph;
        std::mutex graphMutex;
        sfs::ThreadSafeQueue<sfs::KeyframeRecord> mappingQueue;

        int nextGraphId = 0;
        int lastGraphId = -1;
        SE3 lastGraphPose;

        std::optional<Frame> ref;

        SE3 T_wc;
        SE3 lastDelta;

        std::vector<std::string> tum;
        std::vector<TrajectorySample> trajectory;
        std::vector<sfs::KeyframeRecord> savedKeyframes;

        // ------------------------------------------------------------
        // Background mapping and loop closure worker
        // ------------------------------------------------------------

        std::jthread mappingWorker([&](std::stop_token st) {
            sfs::KeyframeRecord kf;

            while (mappingQueue.pop(kf)) {
                if (st.stop_requested())
                    break;

                // 1. GPU TSDF voxel integration.
                const bool integrated = mapper.consider(kf.depth.ptr<float>(), kf.depth.cols, kf.depth.rows, kf.fx, kf.fy, kf.cx, kf.cy, toPose4f(kf.pose),
                    true);

                if (!integrated)
                    continue;

                // 2. Loop detection and verification.
                if (auto loopCandidate = loopDetector.addKeyframe(kf)) {
                    std::println("Loop detected! Query Frame {} matched Old Frame {} " "with {} inliers (RMSE: {:.4f}m)",
                        loopCandidate->queryFrameIndex, loopCandidate->matchedFrameIndex, loopCandidate->inliers,
                        loopCandidate->inlierRmseMeters);



                    bool loopEdgeAdded = false;
                    int loopOptimizationIterations = 0;

                    {
                        // Serialize all pose-graph access. The tracking thread uses
                        // this same mutex when adding poses and sequential constraints.
                        std::lock_guard<std::mutex> lock(graphMutex);
                        const auto printLoopResidual = [&](const char* label) {
                            sfs::GraphPose T_wm;
                            sfs::GraphPose T_wq;

                            if (!graph.getPose(loopCandidate->matchedGraphId, T_wm) ||  !graph.getPose(loopCandidate->queryGraphId, T_wq)) {
                                std::println("{}: unable to retrieve loop poses", label);
                                return;
                            }

                            // Graph's predicted relative transform:
                            // T_matched_from_query = T_wm.inverse() * T_wq
                            const sfs::GraphPose T_predicted = T_wm.inverse() * T_wq;

                            // Error should approach identity when the edge is satisfied.
                            const sfs::GraphPose T_error = loopCandidate->T_matched_from_query.inverse() * T_predicted;

                            const double translationError = T_error.translation().norm();

                            const double rotationErrorDeg = T_error.so3().log().norm() * 180.0 / std::numbers::pi;

                            std::println("{}: translation residual = {:.4f} m, " "rotation residual = {:.3f} deg", label, translationError, rotationErrorDeg);
                            };

                        printLoopResidual("Before optimization");

                        loopEdgeAdded = graph.addLoopEdge(loopCandidate->matchedGraphId, loopCandidate->queryGraphId, loopCandidate->T_matched_from_query, 1.0, 1.2);

                        if (loopEdgeAdded) {
                           loopOptimizationIterations = graph.optimize(40);
                           printLoopResidual("After optimization");
                        }
                    }

                    if (loopEdgeAdded) {
                        std::println("Loop constraint added; pose graph optimized, " "poses={}, iterations={}", graph.size(), loopOptimizationIterations);
                    }
                    else {
                        std::println(stderr, "Failed to add loop constraint to pose graph.");
                    }


                }

                savedKeyframes.push_back(std::move(kf));
            }
            });

        // ------------------------------------------------------------
        // Performance instrumentation
        // ------------------------------------------------------------

        int processed = 0;

        std::vector<double> frameLatencyMs;

        frameLatencyMs.reserve(
            opt.maxFrames > 0
            ? static_cast<std::size_t>(opt.maxFrames)
            : 1024U);

        // Start immediately before the frame-processing loop.
        // End after the mapping worker drains its queue and joins.
        const auto sequenceStart = SteadyClock::now();

        // Sum of individual main-loop frame durations.
        double trackingSeconds = 0.0;

        // ------------------------------------------------------------
        // Main tracking loop
        // ------------------------------------------------------------

        for (int i = opt.firstFrame;
            !stop.stop_requested();
            ++i)
        {
            if (opt.maxFrames > 0 && processed >= opt.maxFrames)
                break;

            // Latency begins before image reading.
            const auto frameStart = SteadyClock::now();

            // 1. Read the RGB and depth images.
            const auto raw = readRawFrame(opt.frameDir, i);

            if (!raw) {
                std::println("End of input at frame {}", i);
                break;
            }

            // 2. Prepare the frame and its pyramid.
            auto cur = makeFrame(raw->rgb, raw->depth, P, maps, pyramidCfg);

            if (!cur) {
                mappingQueue.setFinished();
                mappingWorker.join();

                return std::unexpected(
                    std::format("frame {}: {}", i, cur.error()));
            }

            // 3. Estimate relative pose.
            const bool firstFrame = !ref.has_value();

            bool trackingOk = true;
            SE3 measuredT_cr;

            if (!firstFrame) {
                const OdometryResult res = odometry.estimate(*ref, *cur, lastDelta);
                std::println("Odometry: {:.2f} ms total",  res.totalMs);

                for (const auto& s : res.stats) {
                    std::println("  L{}: photo-select={:.2f} ms, "
                        "geo-select={:.2f} ms, total-select={:.2f} ms, " "graph={:.2f} ms, setup={:.2f} ms, " "optimize={:.2f} ms | photo={} geo={}",
                        s.level, s.photoSelectionMs, s.geoSelectionMs, s.sampleSelectionMs, s.graphConstructionMs, s.optimizerSetupMs, s.optimizeMs,
                        s.photoEdges, s.geoEdges);}

                measuredT_cr = res.T_cr;

                trackingOk = res.ok && measuredT_cr.translation().norm() < 0.5 && rotationAngleDeg(measuredT_cr) < 30.0;

                if (trackingOk) {
                    T_wc = T_wc * measuredT_cr.inverse();
                    lastDelta = measuredT_cr;
                }
                else {
                    lastDelta = SE3();

                    std::println(stderr, "frame {}: tracking lost; skipping fusion", i);
                }
            }

            // 4. Update pose graph.

            int currentGraphId = -1;
            bool currentFrameHasGraphPose = false;

            if (trackingOk) {
                const int id = nextGraphId++;
                const sfs::GraphPose graphPose = T_wc;

                {
                    std::lock_guard<std::mutex> lock(graphMutex);

                    graph.addPose(id, graphPose, id == 0);

                    if (lastGraphId >= 0) {
                        const sfs::GraphPose measurement = lastGraphPose.inverse() * graphPose;

                        graph.addSequentialEdge(lastGraphId, id, measurement);}
                }

                lastGraphId = id;
                lastGraphPose = graphPose;
                currentGraphId = id;
                currentFrameHasGraphPose = true;
            }


            // 5. Save trajectory samples.
            tum.push_back(tumLine(static_cast<double>(i), T_wc));

            trajectory.push_back({ static_cast<double>(i), currentGraphId, T_wc, currentFrameHasGraphPose});


            // 6. Enqueue keyframe data for background mapping.
            const PyramidLevel& L = cur->levels.front();
            if (viewer) {
                viewer->pushFrame(cur->colorBgr, L.gray, L.depth);}
            if (trackingOk && L.depth.type() == CV_32FC1 && processed % opt.keyframeEvery == 0){
                sfs::KeyframeRecord kf;

                kf.frameIndex = i;
                kf.graphId = currentGraphId;
                kf.pose = T_wc;

                kf.depth = L.depth.clone();

                kf.fx = static_cast<float>(L.K.fx);
                kf.fy = static_cast<float>(L.K.fy);
                kf.cx = static_cast<float>(L.K.cx);
                kf.cy = static_cast<float>(L.K.cy);

                kf.gray = L.gray.clone();

                mappingQueue.push(std::move(kf));
            }


            if (viewer) {
                viewer->pushPose(T_wc);

                if (processed % opt.keyframeEvery == 0) {
                    viewer->pushPoints(makeColoredPoints(*cur, 6), T_wc);
                }
            }

            // 8. Advance the reference frame on successful tracking.
            ref = std::move(*cur);

            // --------------------------------------------------------
            // Record frame latency.
            //
            // Includes:
            //   - image reading and decoding
            //   - frame preparation
            //   - odometry
            //   - pose graph and trajectory bookkeeping
            //   - keyframe enqueueing
            //   - viewer work, if enabled
            //
            // Does not include background mapping queue drain.
            // --------------------------------------------------------

            const auto frameEnd = SteadyClock::now();

            const double latencyMs = std::chrono::duration<double, std::milli>(
                    frameEnd - frameStart).count();

            frameLatencyMs.push_back(latencyMs);
            trackingSeconds += latencyMs / 1000.0;

            ++processed;
        }

       
        // End sequence timing after mapping backlog has been processed.
        

        mappingQueue.setFinished();
        mappingWorker.join();

        const auto sequenceEnd = SteadyClock::now();

        const double endToEndSeconds = std::chrono::duration<double>(sequenceEnd - sequenceStart).count();

        // Active tracking throughput based on summed frame durations.
        const double trackingFps = trackingSeconds > 0.0? static_cast<double>(processed) / trackingSeconds : 0.0;

        // End-to-end throughput includes mapping queue drain and join.
        const double endToEndFps = endToEndSeconds > 0.0 ? static_cast<double>(processed) / endToEndSeconds : 0.0;

        // ------------------------------------------------------------
        // Performance report
        // ------------------------------------------------------------

        std::println("\n========== SLAM PERFORMANCE ==========");
        std::println("Frames processed:       {}", processed);
        std::println("Tracking time:          {:.3f} s", trackingSeconds);
        std::println("Tracking FPS:           {:.2f}", trackingFps);
        std::println("End-to-end time:        {:.3f} s", endToEndSeconds);
        std::println("End-to-end throughput:  {:.2f} FPS", endToEndFps);

        std::println("Tracking latency p50:   {:.2f} ms", percentileMs(frameLatencyMs, 0.50));

        std::println("Tracking latency p95:   {:.2f} ms", percentileMs(frameLatencyMs, 0.95));

        std::println("Tracking latency p99:   {:.2f} ms", percentileMs(frameLatencyMs, 0.99));

        std::println("======================================\n");

        if (trajectory.empty())
            return std::unexpected("no frames processed");

       
        // Export trajectory and mesh before pose graph optimization. These offline stages are excluded from the timing report.
       

        const auto trajectoryBeforePath = opt.dir / "trajectory_before.txt";

        {
            std::ofstream out(trajectoryBeforePath);

            for (const auto& line : tum)
                std::println(out, "{}", line);
        }

        const auto meshBeforePath =  opt.dir / "mesh_before_optimization.ply";

        mapper.savePly(meshBeforePath);

        // ------------------------------------------------------------
        // Pose graph optimization.
        // ------------------------------------------------------------

        if (graph.size() > 1) {
            const int iterations = graph.optimize(40);

            std::println("Pose graph optimized: {} poses, iterations={}", graph.size(), iterations);
        }

        // ------------------------------------------------------------
        // Export optimized trajectory.
        // ------------------------------------------------------------

        const auto trajectoryAfterPath =
            opt.dir / "trajectory_after.txt";

        {
            std::ofstream out(trajectoryAfterPath);

            SE3 lastOut; bool haveLast = false;
            for (const auto& s : trajectory) {
                SE3 pose = haveLast ? lastOut : s.originalPose;
                SE3 opt;
                if (s.hasGraphPose && s.graphId >= 0 && graph.getPose(s.graphId, opt)) {
                    pose = opt; lastOut = opt; haveLast = true;
                }
                std::println(out, "{}", tumLine(s.timestamp, pose));
            }
        }

        // ------------------------------------------------------------
        // Rebuild TSDF using optimized keyframe poses.
        // ------------------------------------------------------------

        mapper.reset();

        for (const auto& saved : savedKeyframes) {
            SE3 optimizedPose;

            if (graph.getPose(saved.graphId, optimizedPose)) {
                mapper.integrateKeyframe(saved.depth.ptr<float>(), saved.depth.cols, saved.depth.rows, saved.fx, saved.fy, saved.cx, saved.cy, toPose4f(optimizedPose));}
        }

        const auto meshPath = opt.dir / "mesh.ply";

        mapper.savePly(meshPath);

        std::println("Optimized mesh saved to {}",meshPath.string());

        return {};
    }

} // namespace

int main(int argc, char** argv) {
    const Options opt = parseArgs({ argv, static_cast<std::size_t>(argc) });

    if (opt.selfTest)
        return runSelfTests() ? 0 : 1;

    std::error_code ec;

    std::filesystem::create_directories(opt.dir, ec);

    if (ec) {
        std::println(stderr, "Failed to create output directory: {}", ec.message());
        return 1;
    }

    const auto params = loadCameraParams(opt.dir / "camera_params.yaml");

    if (!params)
        return 1;

    const auto maps = loadUndistortMaps(*params);

    if (!maps)
        return 1;

    if (opt.noViewer) {
        if (const auto r = trackSequence(
            opt, *params, *maps, nullptr, std::stop_token{}); !r)
        {
            std::println(stderr, "error: {}", r.error());
            return 1;
        }

        return 0;
    }
    TrajectoryViewer viewer;

    std::jthread trackingWorker([&](std::stop_token st) {
        try {
            const auto result = trackSequence(opt, *params, *maps, &viewer, st);

            if (!result) {
                std::fprintf(stderr, "[Tracking] Failed: %s\n",
                    result.error().c_str());
            }

            std::fprintf(stderr,  "[Tracking] Worker finished. Viewer remains open until closed.\n");
        }
        catch (const std::exception& e) {
            std::fprintf(stderr, "[Tracking] Worker exception: %s\n", e.what());
        }
        catch (...) {
            std::fprintf(stderr, "[Tracking] Unknown worker exception.\n");
        }
        });

    viewer.run();


    return 0;
}
