#pragma once

#include "../src/pch.h"
#include <sleekslam/core/frame.hpp>

namespace sfs {

    struct LoopConfig {
        int maxFeatures = 1200;
        int minFrameGap = 30;
        int maxCandidates = 5;
        int minGoodMatches = 35;
        int minInliers = 25;
        float ratioTest = 0.75f;
        double minInlierRatio = 0.35;
    };

    struct LoopCandidate {
        int queryFrame = -1;
        int matchedFrame = -1;
        int goodMatches = 0;
        int inliers = 0;
        double inlierRatio = 0.0;
        cv::Mat queryDescriptors;
        cv::Mat matchedDescriptors;
        std::vector<cv::KeyPoint> queryKeypoints;
        std::vector<cv::KeyPoint> matchedKeypoints;
        std::vector<cv::DMatch> inlierMatches;
    };

    // Stage 1: ORB candidate generation + 2D geometric verification with a
    // fundamental matrix. This deliberately does NOT add pose-graph edges yet.
    // The RGB-D 3D verification stage should be added after this passes tests.
    class LoopClosureDetector {
    public:
        explicit LoopClosureDetector(const LoopConfig& cfg = {})
            : cfg_(cfg),
            orb_(cv::ORB::create(cfg.maxFeatures)),
            matcher_(cv::BFMatcher::create(cv::NORM_HAMMING, false)) {
        }

        void reset() {
            entries_.clear();
        }

        // Add one successfully tracked frame. Returns candidates from earlier
        // frames, sorted by inlier count, that pass descriptor and RANSAC checks.
        std::vector<LoopCandidate> addFrame(int frameIndex, const vo::Frame& frame) {
            std::vector<LoopCandidate> accepted;
            if (frame.colorBgr.empty() || frame.levels.empty()) {
                return accepted;
            }

            cv::Mat gray;
            cv::cvtColor(frame.colorBgr, gray, cv::COLOR_BGR2GRAY);

            Entry current;
            current.frameIndex = frameIndex;
            orb_->detectAndCompute(gray, cv::noArray(),
                current.keypoints, current.descriptors);
            if (current.descriptors.empty() || current.keypoints.size() < 10) {
                return accepted;
            }

            for (const Entry& old : entries_) {
                if (frameIndex - old.frameIndex < cfg_.minFrameGap ||
                    old.descriptors.empty()) {
                    continue;
                }

                std::vector<std::vector<cv::DMatch>> knn;
                matcher_->knnMatch(current.descriptors, old.descriptors, knn, 2);

                std::vector<cv::DMatch> good;
                good.reserve(knn.size());
                for (const auto& pair : knn) {
                    if (pair.size() == 2 &&
                        pair[0].distance < cfg_.ratioTest * pair[1].distance) {
                        good.push_back(pair[0]);
                    }
                }
                if (static_cast<int>(good.size()) < cfg_.minGoodMatches) {
                    continue;
                }

                std::vector<cv::Point2f> queryPts, oldPts;
                queryPts.reserve(good.size());
                oldPts.reserve(good.size());
                for (const auto& m : good) {
                    queryPts.push_back(current.keypoints[m.queryIdx].pt);
                    oldPts.push_back(old.keypoints[m.trainIdx].pt);
                }

                cv::Mat inlierMask;
                cv::findFundamentalMat(
                    queryPts, oldPts, cv::FM_RANSAC, 2.0, 0.99, inlierMask);

                if (inlierMask.empty() || inlierMask.total() != good.size()) {
                    continue;
                }

                LoopCandidate candidate;
                candidate.queryFrame = frameIndex;
                candidate.matchedFrame = old.frameIndex;
                candidate.goodMatches = static_cast<int>(good.size());
                candidate.queryKeypoints = current.keypoints;
                candidate.matchedKeypoints = old.keypoints;
                candidate.queryDescriptors = current.descriptors.clone();
                candidate.matchedDescriptors = old.descriptors.clone();

                for (std::size_t i = 0; i < good.size(); ++i) {
                    if (inlierMask.at<uchar>(static_cast<int>(i)) != 0) {
                        candidate.inlierMatches.push_back(good[i]);
                    }
                }

                candidate.inliers = static_cast<int>(candidate.inlierMatches.size());
                candidate.inlierRatio = candidate.goodMatches > 0
                    ? static_cast<double>(candidate.inliers) / candidate.goodMatches
                    : 0.0;

                if (candidate.inliers >= cfg_.minInliers &&
                    candidate.inlierRatio >= cfg_.minInlierRatio) {
                    accepted.push_back(std::move(candidate));
                }
            }

            // Store only after comparing against earlier frames, so a frame
            // cannot match itself.
            entries_.push_back(std::move(current));

            std::sort(accepted.begin(), accepted.end(),
                [](const LoopCandidate& a, const LoopCandidate& b) {
                    if (a.inliers != b.inliers) return a.inliers > b.inliers;
                    return a.inlierRatio > b.inlierRatio;
                });
            if (static_cast<int>(accepted.size()) > cfg_.maxCandidates) {
                accepted.resize(static_cast<std::size_t>(cfg_.maxCandidates));
            }
            return accepted;
        }

    private:
        struct Entry {
            int frameIndex = -1;
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
        };

        LoopConfig cfg_;
        cv::Ptr<cv::ORB> orb_;
        cv::Ptr<cv::BFMatcher> matcher_;
        std::vector<Entry> entries_;
    };

} // namespace sfs
