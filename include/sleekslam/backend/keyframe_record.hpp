#pragma once

#include "../src/pch.h"

namespace sfs {

    struct KeyframeRecord {
        int frameIndex = -1;
        int graphId = -1;

        // Camera Pose (Camera-to-World, T_wc)
        Sophus::SE3d pose;

        // Intrinsics
        float fx = 0.f;
        float fy = 0.f;
        float cx = 0.f;
        float cy = 0.f;

        // Sensor Data & Features
        cv::Mat depth; // CV_32FC1 in meters
        cv::Mat gray;  // Grayscale image for ORB feature detection
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors; // ORB descriptors
        fbow::fBow bowVec;   // FBoW Bag-of-Words vector
    };

} // namespace sfs