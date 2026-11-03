#pragma once

#include <sleekslam/core/frame.hpp>
#include <sleekslam/backend/keyframe_record.hpp>

namespace sfs {

    struct RgbdLoopConfig {
        int maxFeatures = 1200;
        int minFrameGap = 30;
        int minGoodMatches = 25;
        int min3dCorrespondences = 30;
        int minRansacInliers = 35;
        float ratioTest = 0.65f;
        float maxDepthMeters = 3.7f;
        double ransacInlierThresholdMeters = 0.06;
        double maxInlierRmseMeters = 0.035;
        int ransacIterations = 1200;
        float minBoWScore = 0.01f;
    };

    struct RgbdLoopCandidate {
        int queryFrameIndex = -1;
        int matchedFrameIndex = -1;
        int queryGraphId = -1;
        int matchedGraphId = -1;
        int goodMatches = 0;
        int valid3dCorrespondences = 0;
        int inliers = 0;
        double inlierRmseMeters = std::numeric_limits<double>::infinity();
        Sophus::SE3d T_matched_from_query;
    };

    class RgbdLoopClosureDetector {
    public:
        explicit RgbdLoopClosureDetector(const std::string& vocabPath, const RgbdLoopConfig& cfg = {}) : cfg_(cfg), orb_(cv::ORB::create(cfg.maxFeatures)) {
            vocab_.readFromFile(vocabPath);
        }

        std::optional<RgbdLoopCandidate> addKeyframe(KeyframeRecord& kf) {
            if (kf.depth.empty() || kf.depth.type() != CV_32FC1 || kf.gray.empty() || kf.fx <= 0.0)
                return std::nullopt;

            cv::Mat gray8;
            if (kf.gray.type() == CV_32F)
                kf.gray.convertTo(gray8, CV_8U);
            else
                gray8 = kf.gray;

            orb_->detectAndCompute(gray8, cv::noArray(), kf.keypoints, kf.descriptors);

            if (kf.descriptors.empty() || kf.keypoints.size() < 20) {
                entries_.push_back(kf);
                return std::nullopt;
            }

            // 1. Transform ORB descriptors into FBoW representation
            kf.bowVec = vocab_.transform(kf.descriptors);

            std::optional<RgbdLoopCandidate> best;

            // 2. Scan past keyframe records using FBoW scoring
            for (const KeyframeRecord& old : entries_) {
                if (kf.frameIndex - old.frameIndex < cfg_.minFrameGap || old.descriptors.empty() || old.bowVec.empty())
                    continue;

                double score = fbow::fBow::score(kf.bowVec, old.bowVec);
                if (score < cfg_.minBoWScore) continue;

                // 3. Geometric Verification
                cv::BFMatcher matcher(cv::NORM_HAMMING, false);
                std::vector<std::vector<cv::DMatch>> knn;
                matcher.knnMatch(kf.descriptors, old.descriptors, knn, 2);
                std::vector<cv::DMatch> good;
                for (const auto& pair : knn) {
                    if (pair.size() >= 2 && pair[0].distance < cfg_.ratioTest * pair[1].distance)
                        good.push_back(pair[0]);
                }
                if (static_cast<int>(good.size()) < cfg_.minGoodMatches) continue;

                std::vector<Eigen::Vector3d> srcQuery, dstOld;
                srcQuery.reserve(good.size());
                dstOld.reserve(good.size());
                for (const auto& m : good) {
                    Eigen::Vector3d pq, po;
                    if (backProject(kf, kf.keypoints[m.queryIdx].pt, pq) && backProject(old, old.keypoints[m.trainIdx].pt, po)) {
                        srcQuery.push_back(pq);
                        dstOld.push_back(po);
                    }
                }
                if (static_cast<int>(srcQuery.size()) < cfg_.min3dCorrespondences) continue;

                const RigidEstimate est = estimateRigidRansac(srcQuery, dstOld);
                if (est.inliers < cfg_.minRansacInliers || !std::isfinite(est.rmse) || est.rmse > cfg_.maxInlierRmseMeters) continue;

                RgbdLoopCandidate c;
                c.queryFrameIndex = kf.frameIndex;
                c.matchedFrameIndex = old.frameIndex;
                c.queryGraphId = kf.graphId;
                c.matchedGraphId = old.graphId;
                c.goodMatches = static_cast<int>(good.size());
                c.valid3dCorrespondences = static_cast<int>(srcQuery.size());
                c.inliers = est.inliers;
                c.inlierRmseMeters = est.rmse;
                c.T_matched_from_query = est.T;

                if (!best || c.inliers > best->inliers || (c.inliers == best->inliers && c.inlierRmseMeters < best->inlierRmseMeters))
                    best = c;
            }

            entries_.push_back(kf);
            return best;
        }

        void reset() { entries_.clear(); }
        std::size_t size() const { return entries_.size(); }

    private:
        struct RigidEstimate {
            Sophus::SE3d T;
            int inliers = 0;
            double rmse = std::numeric_limits<double>::infinity();
            std::vector<int> inlierIndices;
        };

        bool backProject(const KeyframeRecord& e, const cv::Point2f& uv, Eigen::Vector3d& p) const {
            const int x = static_cast<int>(std::lround(uv.x));
            const int y = static_cast<int>(std::lround(uv.y));
            if (x < 0 || y < 0 || x >= e.depth.cols || y >= e.depth.rows) return false;
            const float z = e.depth.at<float>(y, x);
            if (!std::isfinite(z) || z <= 0.05f || z > cfg_.maxDepthMeters) return false;
            p = Eigen::Vector3d((uv.x - e.cx) * z / e.fx, (uv.y - e.cy) * z / e.fy, z);
            return p.allFinite();
        }

        static bool fitRigid(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, const std::vector<int>& ids, Sophus::SE3d& T) {
            if (ids.size() < 3) return false;
            Eigen::Vector3d cs = Eigen::Vector3d::Zero(), cd = Eigen::Vector3d::Zero();
            for (int i : ids) { cs += src[i]; cd += dst[i]; }
            cs /= static_cast<double>(ids.size());
            cd /= static_cast<double>(ids.size());
            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (int i : ids) H += (src[i] - cs) * (dst[i] - cd).transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
            Eigen::Matrix3d R = V * U.transpose();
            if (R.determinant() < 0.0) { V.col(2) *= -1.0; R = V * U.transpose(); }
            T = Sophus::SE3d(Eigen::Quaterniond(R), cd - R * cs);
            return T.matrix().allFinite();
        }

        RigidEstimate estimateRigidRansac(const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst) const {
            RigidEstimate best;
            if (src.size() != dst.size() || src.size() < 3) return best;
            std::mt19937 rng(static_cast<std::uint32_t>(0x9e3779b9u ^ src.size()));
            std::uniform_int_distribution<int> pick(0, static_cast<int>(src.size()) - 1);
            for (int iter = 0; iter < cfg_.ransacIterations; ++iter) {
                std::vector<int> sample;
                while (sample.size() < 3) {
                    int id = pick(rng);
                    if (std::find(sample.begin(), sample.end(), id) == sample.end()) sample.push_back(id);
                }
                Sophus::SE3d T;
                if (!fitRigid(src, dst, sample, T)) continue;
                std::vector<int> inliers;
                double sumSq = 0.0;
                for (int i = 0; i < static_cast<int>(src.size()); ++i) {
                    const double e = (T * src[i] - dst[i]).norm();
                    if (e <= cfg_.ransacInlierThresholdMeters) { inliers.push_back(i); sumSq += e * e; }
                }
                if (inliers.size() > best.inlierIndices.size()) {
                    best.inlierIndices = std::move(inliers);
                    best.inliers = static_cast<int>(best.inlierIndices.size());
                    best.rmse = std::sqrt(sumSq / best.inliers);
                    best.T = T;
                }
            }
            if (best.inlierIndices.size() >= 3) {
                Sophus::SE3d refined;
                if (fitRigid(src, dst, best.inlierIndices, refined)) {
                    double sumSq = 0.0;
                    for (int i : best.inlierIndices) sumSq += (refined * src[i] - dst[i]).squaredNorm();
                    best.rmse = std::sqrt(sumSq / best.inlierIndices.size());
                    best.T = refined;
                }
            }
            return best;
        }

        RgbdLoopConfig cfg_;
        cv::Ptr<cv::ORB> orb_;
        fbow::Vocabulary vocab_;
        std::vector<KeyframeRecord> entries_;
    };

} // namespace sfs