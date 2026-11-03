// camera.hpp  --  calibration, undistortion, back-projection / projection, depth->RGB registration.
//
// Everything the VO needs to know about the sensor comes from camera_params.yaml. Visual odometry runs in the RGB camera frame: the depth image is undistorted in the
// IR frame, then forward-projected onto the RGB pixel grid so that (gray, depth) are pixel-aligned.
#pragma once
#include "../src/pch.h"
#include <sleekslam/core/pose.hpp>
namespace vo {
    using SE3 = Sophus::SE3d;
    using Vec3 = Eigen::Vector3d;
    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using JacRow = Eigen::Matrix<double, 1, 6>;   // Jacobian of one scalar residual w.r.t. the 6 pose parameters

    template <class T>
    using Result = std::expected<T, std::string>;

    
    // Types
   
    struct Pinhole {
        double fx{}, fy{}, cx{}, cy{};
        [[nodiscard]] constexpr Pinhole scaled(double s) const noexcept { return { fx * s, fy * s, cx * s, cy * s }; }
    };

    struct CamIntrinsics {
        int w{}, h{};
        Pinhole K;               // intrinsics of the UNDISTORTED image
        std::string mapFile;
    };

    struct CameraParams {
        double depthUnitToM = 0.001;
        CamIntrinsics ir, rgb;
        SE3 T_rgb_from_ir;       // p_rgb = T * p_ir
    };

    struct ColoredPoint {
        float x{}, y{}, z{};
        std::uint8_t r{ 160 }, g{ 160 }, b{ 160 };
    };

    struct UndistortMap {
        cv::Mat map;             // CV_32FC2
        int w{}, h{};
    };

    struct UndistortMaps {
        UndistortMap ir, rgb;
    };

    // *********************
    // loading my camera_params.yaml
    // ********************
    [[nodiscard]] inline Result<CamIntrinsics> readCameraNode(const cv::FileNode& camNode, std::string_view name) {
        const cv::FileNode p = camNode["undistorted_pinhole"];
        if (p.empty())
            return std::unexpected(std::format("'{}' has no undistorted_pinhole block", name));

        CamIntrinsics out;
        p["width_px"] >> out.w;
        p["height_px"] >> out.h;
        p["fx_px"] >> out.K.fx;
        p["fy_px"] >> out.K.fy;
        p["cx_px"] >> out.K.cx;
        p["cy_px"] >> out.K.cy;
        p["map_file"] >> out.mapFile;

        if (out.w <= 0 || out.h <= 0 || out.K.fx <= 0 || out.K.fy <= 0)
            return std::unexpected(std::format("'{}' has invalid undistorted_pinhole values", name));
        return out;
    }

    [[nodiscard]] inline Sophus::SE3d rigidFromMats(const cv::Mat& Rm, const cv::Mat& tm) {
        Eigen::Matrix3d R;
        Vec3 t;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) R(r, c) = Rm.at<double>(r, c);
            t(r) = tm.at<double>(r);
        }
        // The yaml keeps ~8 digits, so R is only approximately orthonormal: go through a normalised quaternion.
        Eigen::Quaterniond q(R);
        q.normalize();
        return SE3(q, t);
    }

    [[nodiscard]] inline Result<CameraParams> loadCameraParams(const std::filesystem::path &path) {
        cv::FileStorage fs(path.string(), cv::FileStorage::READ);
        if (!fs.isOpened())
            return std::unexpected(std::format("cannot open {}", path.string()));

        CameraParams P;
        fs["depth"]["unit_to_meters"] >> P.depthUnitToM;

        auto ir = readCameraNode(fs["ir_camera"], "ir_camera");
        if (!ir) return std::unexpected(std::move(ir).error());
        P.ir = *std::move(ir);

        auto rgb = readCameraNode(fs["rgb_camera"], "rgb_camera");
        if (!rgb) return std::unexpected(std::move(rgb).error());
        P.rgb = *std::move(rgb);

        cv::Mat Rm, tm;
        fs["ir_to_rgb_extrinsics"]["rotation_R_rgb_from_ir"] >> Rm;
        fs["ir_to_rgb_extrinsics"]["translation_t_rgb_from_ir_m"] >> tm;
        if (Rm.rows != 3 || Rm.cols != 3 || tm.rows != 3 || tm.cols != 1)
            return std::unexpected(std::format("extrinsics R/t missing or wrong shape in {}", path.string()));
        Rm.convertTo(Rm, CV_64F);
        tm.convertTo(tm, CV_64F);
        P.T_rgb_from_ir = rigidFromMats(Rm, tm);
        return P;
    }

    
    // Undistortion maps (.bin binary files(ir_map.bin, rgb_map.bin))
   
    [[nodiscard]] inline Result<UndistortMap> loadUndistortMap(const std::filesystem::path &path) {
        static_assert(std::endian::native == std::endian::little, "map files are little-endian");

        std::ifstream f(path, std::ios::binary);
        if (!f) return std::unexpected(std::format("cannot open {}", path.string()));

        std::int32_t w{}, h{};
        std::array<double, 4> intrinsics{};   // unused: the yaml is the source of truth
        f.read(reinterpret_cast<char*>(&w), sizeof w);
        f.read(reinterpret_cast<char*>(&h), sizeof h);
        f.read(reinterpret_cast<char*>(intrinsics.data()), sizeof intrinsics);
        if (!f || w <= 0 || h <= 0 || w > 16384 || h > 16384)
            return std::unexpected(std::format("bad header in {}", path.string()));

        UndistortMap out;
        out.w = w;
        out.h = h;
        out.map.create(h, w, CV_32FC2);
        f.read(reinterpret_cast<char*>(out.map.data),
            static_cast<std::streamsize>(out.map.total() * out.map.elemSize()));
        if (!f) return std::unexpected(std::format("truncated map file {}", path.string()));
        return out;
    }

    [[nodiscard]] inline Result<UndistortMaps> loadUndistortMaps(const CameraParams& P) {
        auto ir = loadUndistortMap(P.ir.mapFile);
        if (!ir) return std::unexpected(std::move(ir).error());
        auto rgb = loadUndistortMap(P.rgb.mapFile);
        if (!rgb) return std::unexpected(std::move(rgb).error());
        return UndistortMaps{ *std::move(ir), *std::move(rgb) };
    }

    [[nodiscard]] inline Result<cv::Mat> undistortImage(const cv::Mat& raw, const UndistortMap& map, int interpolation) {
        if (map.w != raw.cols || map.h != raw.rows)
            return std::unexpected(std::format("map is {}x{} but image is {}x{}", map.w, map.h, raw.cols, raw.rows));
        cv::Mat out;
        cv::remap(raw, out, map.map, cv::noArray(), interpolation, cv::BORDER_CONSTANT, cv::Scalar(0));
        return out;
    }

    // Pinhole geometry
    //   backproject: pixel (u = column, v = row) + depth z  ->  3D point (X right, Y down, Z forward)
    [[nodiscard]] inline Vec3 backproject(const Pinhole& K, double u, double v, double z) noexcept {
        return Vec3((u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z);
    }

    // Depth (undistorted, 16-bit raw units, IR frame)  ->  depth on the RGB pixel grid (float meters, 0 = invalid)
    // Nearest surface wins (z-buffer); each depth sample is splatted to a 2x2 block to avoid rounding holes.
    [[nodiscard]] inline cv::Mat registerDepthToRgb(const cv::Mat& depthUndist16, const CameraParams& P, cv::Size rgbSize) {
        cv::Mat out = cv::Mat::zeros(rgbSize, CV_32F);

        auto put = [&out](int u, int v, float z) {
            if (u < 0 || u >= out.cols || v < 0 || v >= out.rows) return;
            float& dst = out.at<float>(v, u);
            if (dst == 0.f || z < dst) dst = z;
            };

        for (int v = 0; v < depthUndist16.rows; ++v) {
            const auto* row = depthUndist16.ptr<std::uint16_t>(v);
            for (int u = 0; u < depthUndist16.cols; ++u) {
                const std::uint16_t d = row[u];
                if (d == 0) continue;

                const Vec3 q = P.T_rgb_from_ir * backproject(P.ir.K, u, v, d * P.depthUnitToM);
                if (q.z() <= 0.0) continue;

                const double uf = P.rgb.K.fx * q.x() / q.z() + P.rgb.K.cx;
                const double vf = P.rgb.K.fy * q.y() / q.z() + P.rgb.K.cy;
                const int u0 = static_cast<int>(std::floor(uf));
                const int v0 = static_cast<int>(std::floor(vf));
                const auto z = static_cast<float>(q.z());
                put(u0, v0, z);     put(u0 + 1, v0, z);
                put(u0, v0 + 1, z); put(u0 + 1, v0 + 1, z);
            }
        }
        return out;
    }

}  // namespace vo