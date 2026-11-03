
#pragma once

#include "../src/pch.h"
#include <sleekslam/odometry/geometric.hpp>
#include <sleekslam/odometry/photometric.hpp>
namespace vo {

    namespace detail {

        template <class F>
        [[nodiscard]] std::optional<JacRow> numericJacobian(const SE3& T, double eps, F&& residualAt){
            JacRow J;

            for (int i = 0; i < 6; ++i) {
                Vec6 d = Vec6::Zero();
                d[i] = eps;

                const std::optional<double> rp = residualAt(SE3::exp(d) * T);

                const std::optional<double> rm = residualAt(SE3::exp(-d) * T);

                if (!rp || !rm)
                    return std::nullopt;

                J[i] = (*rp - *rm) / (2.0 * eps);
            }

            return J;
        }

        inline bool report(const std::string& name, bool pass, const std::string& detail){
            std::println("[{}] {:<34} {}", pass ? "PASS" : "FAIL", name, detail);

            return pass;
        }

        // Verify that the optimized interpolation helper produces the
        // same intensity and gradient values as the original bilinear().
        [[nodiscard]] inline bool checkPhotometricInterpolation(){
            constexpr int W = 64;
            constexpr int H = 48;

            PyramidLevel cur;
            cur.w = W;
            cur.h = H;

            cur.gray = cv::Mat(H, W, CV_32F);
            cur.gradX = cv::Mat(H, W, CV_32F);
            cur.gradY = cv::Mat(H, W, CV_32F);

            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    cur.gray.at<float>(y, x) = 0.7f * x + 1.3f * y + 0.01f * x * y;

                    cur.gradX.at<float>(y, x) = 0.2f * x - 0.1f * y + 2.0f;

                    cur.gradY.at<float>(y, x) = -0.15f * x + 0.3f * y - 1.0f;
                }
            }

            const std::array<cv::Point2d, 8> positions = {cv::Point2d(1.25, 1.75),cv::Point2d(5.50, 8.25), cv::Point2d(12.10, 19.90), cv::Point2d(21.33, 17.66),
                cv::Point2d(31.50, 23.50), cv::Point2d(42.80, 11.20), cv::Point2d(58.25, 39.75), cv::Point2d(62.90, 45.80) };

            double maxError = 0.0;
            bool valid = true;

            for (const auto& p : positions) {
                const auto optimized = bilinearPhoto(cur, p.x, p.y, true);

                const auto intensity = bilinear(cur.gray, p.x, p.y);

                const auto gx = bilinear(cur.gradX, p.x, p.y);

                const auto gy = bilinear(cur.gradY, p.x, p.y);

                if (!optimized || !intensity || !gx || !gy) {
                    valid = false;
                    break;
                }

                maxError = std::max(maxError, std::abs(static_cast<double>(optimized->intensity) - static_cast<double>(*intensity)));

                maxError = std::max(maxError, std::abs(static_cast<double>(optimized->gx) - static_cast<double>(*gx)));

                maxError = std::max(maxError, std::abs(static_cast<double>(optimized->gy) - static_cast<double>(*gy)));}

            // Floating-point operation order may differ slightly.
            constexpr double tolerance = 1e-4;
            const bool pass = valid && maxError < tolerance;

            return report("photometric interpolation", pass, std::format("max absolute error = {:.3e} (tol {:.1e})", maxError, tolerance));}

        [[nodiscard]] inline bool checkSE3Roundtrip(){
            const Vec6 xis[] = {(Vec6() << 0.01, -0.02, 0.03, 0.001, -0.002, 0.003).finished(), (Vec6() << 0.5, -0.3, 1.2, 0.4, -0.7, 0.2).finished(),

                (Vec6() << -1.0, 2.0, 0.5, 1.5, 1.0, -1.2).finished(),};

            double worst = 0.0;
            bool valid = true;

            for (const Vec6& xi : xis) {
                const SE3 T = expMap(xi);

                worst = std::max(worst, (logMap(T) - xi).norm());
                const Eigen::Matrix3d R = T.rotationMatrix();

                valid = valid && (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() < 1e-12 && std::abs(R.determinant() - 1.0) < 1e-12;
            }

            return report("SE3 exp/log round trip", worst < 1e-9 && valid, std::format("max |log(exp(xi)) - xi| = {:.2e}, rotations valid = {}", worst, valid));}

        [[nodiscard]] inline bool checkPhotometricJacobian() {
            const int W = 160;
            const int H = 120;

            cv::Mat gray(H, W, CV_32F);

            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    gray.at<float>(y, x) = 128.f + 60.f * std::sin(static_cast<float>(2 * std::numbers::pi * x / 90.0 + 0.3)) * std::cos(static_cast<float>(
                            2 * std::numbers::pi * y / 110.0));
                }
            }

            PyramidLevel cur;
            cur.K = { 200.0, 200.0, 80.0, 60.0 };
            cur.w = W;
            cur.h = H;
            cur.gray = gray;

            computeGradients(gray, cur.gradX, cur.gradY);

            const PhotoSample s{Vec3(0.10, -0.05, 1.5), 100.0 };

            const SE3 T = expMap((Vec6() << 0.01, -0.02, 0.015, 0.01, -0.008, 0.006).finished());

            const auto analytic = photometricEval(T, s, cur, true);

            const auto numeric = numericJacobian(T, 1e-4, [&](const SE3& Tp) -> std::optional<double> {
                    const auto e = photometricEval(Tp, s, cur, false);

                    return e ? std::optional<double>(e->r) : std::nullopt;});

            if (!analytic || !numeric) 
            {return report("photometric Jacobian vs numeric", false, "sample left the image");}

            const double rel = (analytic->J - *numeric).norm() / std::max(1e-9, numeric->norm());

            return report("photometric Jacobian vs numeric", rel < 0.15, std::format("relative error = {:.3e} (tol 0.15)", rel));}

        [[nodiscard]] inline bool checkGeometricJacobian(){
            // An exact tilted plane Z = z0 + aX + bY is rendered as
            // a depth map so the point-to-plane residual remains smooth
            // as the projected pixel changes.

            const int W = 160;
            const int H = 120;

            const Pinhole K{ 200.0, 200.0, 80.0, 60.0 };

            const double z0 = 2.0;
            const double a = 0.2;
            const double b = -0.1;

            cv::Mat depth(H, W, CV_32F);

            for (int v = 0; v < H; ++v) {
                for (int u = 0; u < W; ++u) {
                    depth.at<float>(v, u) = static_cast<float>(z0 / (1.0 - a * (u - K.cx) / K.fx - b * (v - K.cy) / K.fy));}
            }

            PyramidLevel cur;
            cur.K = K;
            cur.w = W;
            cur.h = H;
            cur.depth = depth;

            computePointsAndNormals(depth,K,1,0.05,cur.points,cur.normals);

            const Vec3 nRef = Vec3(a, b, -1.0).normalized();

            const GeoSample s{Vec3(0.10, 0.05, z0 + a * 0.10 + b * 0.05),nRef};

            const SE3 T = expMap((Vec6() << 0.01, -0.02, 0.015, 0.005, -0.004, 0.003).finished());

            const GeoGate gate{};

            const auto analytic = geometricEval(T, s, cur, gate, true);

            const auto numeric = numericJacobian(T, 1e-4, [&](const SE3& Tp) -> std::optional<double> {
                    const auto e = geometricEval(Tp, s, cur, gate, false);

                    return e? std::optional<double>(e->r): std::nullopt;});

            if (!analytic || !numeric) {return report("geometric Jacobian vs numeric",false,"correspondence failed");
            }

            const double rel = (analytic->J - *numeric).norm() / std::max(1e-9, numeric->norm());

            return report("geometric Jacobian vs numeric", rel < 1e-2, std::format("relative error = {:.3e} (tol 1e-2)",rel));}

        [[nodiscard]] inline bool checkHuberKernel(){
            const double delta = 1.345;

            g2o::RobustKernelHuber k;
            k.setDelta(delta);

            Eigen::Vector3d rho;

            // chi2 = 100 -> |e|/sigma = 10 (an outlier)
            k.robustify(100.0, rho);

            const double rhoOutlier = rho[0];

            const double expected =
                2.0 * 10.0 * delta - delta * delta;

            const bool outlierOk =
                std::abs(rhoOutlier - expected) < 1e-9;

            // chi2 = 1 -> |e|/sigma = 1 (an inlier)
            k.robustify(1.0, rho);

            const bool inlierOk =
                std::abs(rho[0] - 1.0) < 1e-12 &&
                std::abs(rho[1] - 1.0) < 1e-12;

            return report(
                "g2o Huber kernel semantics",
                outlierOk && inlierOk,
                std::format(
                    "rho(100) = {:.4f} (expected {:.4f}), inlier unchanged = {}",
                    rhoOutlier,
                    expected,
                    inlierOk));
        }

    } // namespace detail

    [[nodiscard]] inline bool runSelfTests()
    {
        std::println("--- self tests ---");

        bool ok = true;

        ok &= detail::checkSE3Roundtrip();
        ok &= detail::checkPhotometricInterpolation();
        ok &= detail::checkPhotometricJacobian();
        ok &= detail::checkGeometricJacobian();
        ok &= detail::checkHuberKernel();

        std::println(
            "--- {} ---",
            ok ? "all passed" : "SOME TESTS FAILED");

        return ok;
    }

} // namespace vo
