// frame.hpp  -- the coarse-to-fine image pyramid, and everything a VO "frame" carries.
//
// Per pyramid level we keep what the two error terms need:
//   photometric:  gray image + its gradient images
//   geometric:    per-pixel 3D points + surface normals (for point-to-plane ICP)
//
// Pyramid conventions (important for correctness):
//   * level l+1 has size ((w+1)/2, (h+1)/2); pixel i of level l+1 sits on pixel 2i of level l, so the intrinsics simply scale by 0.5 per level (fx, fy, cx, cy all * 0.5).
//   * gray is blurred+decimated with cv::pyrDown; depth is NEVER blurred (that would invent surfaces between objects) -- it is averaged only over neighbours that agree with the centre.
#pragma once

#include "../src/pch.h"
#include <sleekslam/core/camera.hpp>
namespace vo {

    struct PyramidConfig {
        int numLevels = 4;
        double depthMergeTol = 0.05;   // when downsampling depth, only average neighbours within 5% of the centre depth
        int normalStepFine = 2;        // pixel offset for normals at full resolution (coarser levels use 1)
        double maxRelJump = 0.05;      // relative depth jump (per pixel step) that invalidates a normal
    };

    struct PyramidLevel {
        Pinhole K;
        int w{}, h{};
        cv::Mat gray;       // CV_32F, 0..255
        cv::Mat gradX;      // CV_32F, gray levels per pixel
        cv::Mat gradY;
        cv::Mat depth;      // CV_32F, meters, 0 = invalid
        cv::Mat points;     // CV_32FC3, camera-frame 3D points, z = 0 = invalid
        cv::Mat normals;    // CV_32FC3, unit normals facing the camera, (0,0,0) = invalid
    };

    struct Frame {
        std::vector<PyramidLevel> levels;   // [0] = full resolution
        cv::Mat colorBgr;                   // undistorted full-res colour (for visualisation only)
    };

    // ----------------------------------------------------------------------------
    // Building blocks
    // ----------------------------------------------------------------------------
    // Sobel with scale 1/8 gives the gradient in "gray levels per pixel".
    inline void computeGradients(const cv::Mat& gray, cv::Mat& gx, cv::Mat& gy) {
        cv::Sobel(gray, gx, CV_32F, 1, 0, 3, 1.0 / 8.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(gray, gy, CV_32F, 0, 1, 3, 1.0 / 8.0, 0.0, cv::BORDER_REPLICATE);
    }

    [[nodiscard]] inline cv::Mat downsampleDepth(const cv::Mat& src, double relTol) {
        cv::Mat dst(cv::Size((src.cols + 1) / 2, (src.rows + 1) / 2), CV_32F, cv::Scalar(0.0));
        for (int v = 0; v < dst.rows; ++v) {
            for (int u = 0; u < dst.cols; ++u) {
                const int sx = 2 * u, sy = 2 * v;
                const float zc = src.at<float>(sy, sx);
                if (zc <= 0.f) continue;

                const float tol = static_cast<float>(relTol) * zc;
                float sum = 0.f;
                int n = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int x = sx + dx, y = sy + dy;
                        if (x < 0 || y < 0 || x >= src.cols || y >= src.rows) continue;
                        const float z = src.at<float>(y, x);
                        if (z > 0.f && std::abs(z - zc) <= tol) { sum += z; ++n; }
                    }
                }
                dst.at<float>(v, u) = sum / static_cast<float>(n);   // n >= 1: the centre always qualifies
            }
        }
        return dst;
    }

    // 3D points from depth, plus normals from central differences of neighbouring points.
    // Normal = dy x dx so that it points back toward the camera (camera frame: X right, Y down, Z forward).
    inline void computePointsAndNormals(const cv::Mat& depth, const Pinhole& K, int step, double maxRelJump, cv::Mat& points, cv::Mat& normals)
    {
        points = cv::Mat::zeros(depth.size(), CV_32FC3);
        normals = cv::Mat::zeros(depth.size(), CV_32FC3);

        for (int v = 0; v < depth.rows; ++v) {
            for (int u = 0; u < depth.cols; ++u) {
                const float z = depth.at<float>(v, u);
                if (z <= 0.f) continue;
                points.at<cv::Vec3f>(v, u) = cv::Vec3f(static_cast<float>((u - K.cx) * z / K.fx),static_cast<float>((v - K.cy) * z / K.fy), z);
            }
        }

        for (int v = step; v < depth.rows - step; ++v) {
            for (int u = step; u < depth.cols - step; ++u) {
                const cv::Vec3f& pc = points.at<cv::Vec3f>(v, u);
                if (pc[2] <= 0.f) continue;

                const cv::Vec3f& pl = points.at<cv::Vec3f>(v, u - step);
                const cv::Vec3f& pr = points.at<cv::Vec3f>(v, u + step);
                const cv::Vec3f& pu = points.at<cv::Vec3f>(v - step, u);
                const cv::Vec3f& pd = points.at<cv::Vec3f>(v + step, u);
                if (pl[2] <= 0.f || pr[2] <= 0.f || pu[2] <= 0.f || pd[2] <= 0.f) continue;

                const float jump = static_cast<float>(maxRelJump * step) * pc[2];
                if (std::abs(pl[2] - pc[2]) > jump || std::abs(pr[2] - pc[2]) > jump || std::abs(pu[2] - pc[2]) > jump || std::abs(pd[2] - pc[2]) > jump) continue;

                const Eigen::Vector3f dx(pr[0] - pl[0], pr[1] - pl[1], pr[2] - pl[2]);
                const Eigen::Vector3f dy(pd[0] - pu[0], pd[1] - pu[1], pd[2] - pu[2]);
                Eigen::Vector3f n = dy.cross(dx);
                const float norm = n.norm();
                if (norm < 1e-12f) continue;
                n /= norm;
                if (n.dot(Eigen::Vector3f(pc[0], pc[1], pc[2])) > 0.f) n = -n;   // face the camera
                normals.at<cv::Vec3f>(v, u) = cv::Vec3f(n.x(), n.y(), n.z());
            }
        }
    }

    
    // Building my coarse-to--fine image Pyramid
   
    [[nodiscard]] inline std::vector<PyramidLevel> buildPyramid(const cv::Mat& gray32, const cv::Mat& depth32,const Pinhole& K0, const PyramidConfig& cfg){
        std::vector<PyramidLevel> levels;
        levels.reserve(static_cast<std::size_t>(cfg.numLevels));

        cv::Mat g = gray32, d = depth32;
        Pinhole K = K0;
        for (int l = 0; l < cfg.numLevels; ++l) {
            PyramidLevel L;
            L.K = K;
            L.w = g.cols;
            L.h = g.rows;
            L.gray = g;
            L.depth = d;
            computeGradients(g, L.gradX, L.gradY); computePointsAndNormals(d, K, l == 0 ? cfg.normalStepFine : 1, cfg.maxRelJump, L.points, L.normals);
            levels.push_back(std::move(L));

            if (l + 1 < cfg.numLevels) {
                cv::Mat gs;
                cv::pyrDown(g, gs);                       // size ((w+1)/2, (h+1)/2), same as downsampleDepth
                d = downsampleDepth(d, cfg.depthMergeTol);
                g = std::move(gs);
                K = K.scaled(0.5);
            }
        }
        return levels;
    }
    
    // ----------------------------------------------------------------------------
    // Raw sensor images -> Frame (undistort -> register depth to RGB -> gray -> pyramid)
    // ----------------------------------------------------------------------------
    [[nodiscard]] inline Result<Frame> makeFrame(const cv::Mat& rgbRaw, const cv::Mat& depthRaw16,const CameraParams& P, const UndistortMaps& maps,const PyramidConfig& cfg){
        if (rgbRaw.empty() || depthRaw16.empty())
            return std::unexpected("empty input image");
        if (depthRaw16.type() != CV_16UC1)
            return std::unexpected(std::format("depth image must be single-channel 16-bit (got OpenCV type {})",depthRaw16.type()));

        auto depthU = undistortImage(depthRaw16, maps.ir, cv::INTER_NEAREST);   // never blend depths
        if (!depthU) return std::unexpected("depth: " + depthU.error());
        auto rgbU = undistortImage(rgbRaw, maps.rgb, cv::INTER_LINEAR);
        if (!rgbU) return std::unexpected("rgb: " + rgbU.error());

        const cv::Mat depthOnRgb = registerDepthToRgb(*depthU, P, rgbU->size());

        cv::Mat gray8, gray32;
        cv::cvtColor(*rgbU, gray8, cv::COLOR_BGR2GRAY);
        gray8.convertTo(gray32, CV_32F);

        Frame f;
        f.colorBgr = *rgbU;
        f.levels = buildPyramid(gray32, depthOnRgb, P.rgb.K, cfg);
        return f;
    }

    // Sub-sampled coloured points of the full-res level (camera frame), for the viewer.
    [[nodiscard]] inline std::vector<ColoredPoint> makeColoredPoints(const Frame& f, int stride) {
        std::vector<ColoredPoint> out;
        const PyramidLevel& L = f.levels.front();
        for (int v = 0; v < L.h; v += stride) {
            for (int u = 0; u < L.w; u += stride) {
                const cv::Vec3f& p = L.points.at<cv::Vec3f>(v, u);
                if (p[2] <= 0.f) continue;
                const cv::Vec3b& bgr = f.colorBgr.at<cv::Vec3b>(v, u);
                out.push_back({ p[0], p[1], p[2], bgr[2], bgr[1], bgr[0] });
            }
        }
        return out;
    }

}  // namespace vo