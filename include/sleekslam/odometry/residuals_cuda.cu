#include <string>
#include <stdexcept>
#include <sleekslam/odometry/residuals_cuda.hpp>

namespace vo::cuda_backend {
    namespace {
        constexpr int BLOCK = 256;

        inline void check(cudaError_t e, const char* what) {
            if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
        }

        template<class T> T* deviceAlloc(std::size_t n) {
            if (!n) return nullptr;
            T* p = nullptr; check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), "cudaMalloc"); return p;
        }
        template<class T> void upload(T* d, const T* h, std::size_t n) {
            if (n) check(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        }
        template<class T> void download(T* h, const T* d, std::size_t n) {
            if (n) check(cudaMemcpy(h, d, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        }
        __device__ inline bool finite3(double x, double y, double z) { return isfinite(x) && isfinite(y) && isfinite(z); }
        __device__ inline void transform(const Pose& T, const double* p, double* q) {
            q[0] = T.R[0] * p[0] + T.R[1] * p[1] + T.R[2] * p[2] + T.t[0];
            q[1] = T.R[3] * p[0] + T.R[4] * p[1] + T.R[5] * p[2] + T.t[1];
            q[2] = T.R[6] * p[0] + T.R[7] * p[1] + T.R[8] * p[2] + T.t[2];
        }
        __device__ inline void jacobian(const double* q, const double* g, double* J) {
            J[0] = g[0]; J[1] = g[1]; J[2] = g[2];
            J[3] = q[1] * g[2] - q[2] * g[1];
            J[4] = q[2] * g[0] - q[0] * g[2];
            J[5] = q[0] * g[1] - q[1] * g[0];
        }
        __device__ inline float bilerp(const float* img, int w, int x, int y, float dx, float dy) {
            const float* r0 = img + static_cast<std::size_t>(y) * w;
            const float* r1 = r0 + w;
            const float ax = 1.0f - dx, ay = 1.0f - dy;
            return ax * ay * r0[x] + dx * ay * r0[x + 1] + ax * dy * r1[x] + dx * dy * r1[x + 1];
        }

        __global__ void photoKernel(int w, int h, Camera K, const float* gray,
            const float* gx, const float* gy, const PhotoSample* samples, int n,
            Pose T, int withJ, Result* out) {
            int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
            Result z{}; const PhotoSample s = samples[i]; double q[3]; transform(T, s.p, q);
            if (!finite3(q[0], q[1], q[2]) || q[2] <= 1e-3) { out[i] = z; return; }
            const double iz = 1.0 / q[2]; const double u = K.fx * q[0] * iz + K.cx, v = K.fy * q[1] * iz + K.cy;
            if (!isfinite(u) || !isfinite(v) || u < 0.0 || v < 0.0 || u >= w - 1.0 || v >= h - 1.0) { out[i] = z; return; }
            int x = static_cast<int>(u), y = static_cast<int>(v); float dx = static_cast<float>(u - x), dy = static_cast<float>(v - y);
            float I = bilerp(gray, w, x, y, dx, dy); if (!isfinite(I)) { out[i] = z; return; }
            z.r = static_cast<double>(I) - s.intensity; if (!isfinite(z.r)) { out[i] = Result{}; return; }
            z.valid = 1;
            if (withJ) {
                float ix = bilerp(gx, w, x, y, dx, dy), iy = bilerp(gy, w, x, y, dx, dy);
                if (!isfinite(ix) || !isfinite(iy)) { out[i] = Result{}; return; }
                double g[3] = { static_cast<double>(ix) * K.fx * iz,
                    static_cast<double>(iy) * K.fy * iz,
                    -(static_cast<double>(ix) * K.fx * q[0] + static_cast<double>(iy) * K.fy * q[1]) * iz * iz };
                jacobian(q, g, z.J);
            }
            out[i] = z;
        }

        __global__ void geoKernel(int w, int h, Camera K, const float* points, const float* normals,
            const GeoSample* samples, int n, Pose T, GeoGate gate, int withJ, Result* out) {
            int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
            Result z{}; const GeoSample s = samples[i]; double q[3]; transform(T, s.p, q);
            if (!finite3(q[0], q[1], q[2]) || q[2] <= 1e-3) { out[i] = z; return; }
            double uf = K.fx * q[0] / q[2] + K.cx, vf = K.fy * q[1] / q[2] + K.cy;
            if (!isfinite(uf) || !isfinite(vf)) { out[i] = z; return; }
            // lround() semantics: halfway values round away from zero.
            int u = static_cast<int>(round(uf)), v = static_cast<int>(round(vf));
            if (u < 0 || u >= w || v < 0 || v >= h) { out[i] = z; return; }
            std::size_t idx = static_cast<std::size_t>(v) * w + u;
            const float* Qf = points + idx * 3; const float* nf = normals + idx * 3;
            if (!finite3(Qf[0], Qf[1], Qf[2]) || Qf[2] <= 0.0f || !finite3(nf[0], nf[1], nf[2]) ||
                (nf[0] * nf[0] + nf[1] * nf[1] + nf[2] * nf[2]) < 0.5f) {
                out[i] = z; return;
            }
            double dx = q[0] - Qf[0], dy = q[1] - Qf[1], dz = q[2] - Qf[2];
            if (sqrt(dx * dx + dy * dy + dz * dz) > gate.maxDist) { out[i] = z; return; }
            double rn[3] = { T.R[0] * s.n[0] + T.R[1] * s.n[1] + T.R[2] * s.n[2],
                T.R[3] * s.n[0] + T.R[4] * s.n[1] + T.R[5] * s.n[2],
                T.R[6] * s.n[0] + T.R[7] * s.n[1] + T.R[8] * s.n[2] };
            if (rn[0] * nf[0] + rn[1] * nf[1] + rn[2] * nf[2] < gate.minNormalCos) { out[i] = z; return; }
            double normal[3] = { nf[0],nf[1],nf[2] }; z.r = normal[0] * dx + normal[1] * dy + normal[2] * dz;
            if (!isfinite(z.r)) { out[i] = Result{}; return; }
            z.valid = 1; if (withJ) jacobian(q, normal, z.J); out[i] = z;
        }
    }

    __device__ inline void robustifyDevice(double chi2, int kind, double delta,
        double& rho0, double& rho1) {
        if (kind == 0) { rho0 = chi2; rho1 = 1.0; return; }
        const double d = fmax(delta, 1e-12), d2 = d * d;
        if (kind == 1) {
            if (chi2 <= d2) { rho0 = chi2; rho1 = 1.0; }
            else {
                const double s = sqrt(fmax(chi2, 0.0));
                rho0 = 2.0 * d * s - d2;
                rho1 = d / fmax(s, 1e-12);
            }
        }
        else {
            const double x = chi2 / d2;
            rho0 = d2 * log1p(x);
            rho1 = 1.0 / (1.0 + x);
        }
    }

    __global__ void accumulateKernel(const Result* values, int n, double info,
        int robustKind, double robustDelta, int withJ,
        double* H, double* b, double* cost) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const Result e = values[i];
        if (!e.valid) return;
        const double chi2 = e.r * e.r * info;
        double rho0, rho1;
        robustifyDevice(chi2, robustKind, robustDelta, rho0, rho1);
        atomicAdd(cost, rho0);
        if (!withJ) return;
        const double w = info * rho1;
#pragma unroll
        for (int r = 0; r < 6; ++r) {
            atomicAdd(b + r, w * e.J[r] * e.r);
#pragma unroll
            for (int c = 0; c < 6; ++c)
                atomicAdd(H + r * 6 + c, w * e.J[r] * e.J[c]);
        }
    }

    struct LevelHandle {
        int w{}, h{}; Camera K{}; std::size_t np{}, ng{};
        float* gray{}, * gx{}, * gy{}, * points{}, * normals{};
        PhotoSample* photo{}; GeoSample* geo{}; Result* dPhotoOut{}, * dGeoOut{};
        double* dH{}, * dB{}, * dCost{};
    };

    LevelHandle* createLevel(int w, int h, Camera K, const float* gray, const float* gx, const float* gy,
        const float* points, const float* normals, const PhotoSample* photo, std::size_t np,
        const GeoSample* geo, std::size_t ng) {
        if (w <= 0 || h <= 0) throw std::invalid_argument("Invalid CUDA pyramid dimensions");
        auto* c = new LevelHandle{}; c->w = w; c->h = h; c->K = K; c->np = np; c->ng = ng;
        try {
            const std::size_t pixels = static_cast<std::size_t>(w) * h;
            c->gray = deviceAlloc<float>(pixels); c->gx = deviceAlloc<float>(pixels); c->gy = deviceAlloc<float>(pixels);
            c->points = deviceAlloc<float>(pixels * 3); c->normals = deviceAlloc<float>(pixels * 3);
            c->photo = deviceAlloc<PhotoSample>(np); c->geo = deviceAlloc<GeoSample>(ng);
            c->dPhotoOut = deviceAlloc<Result>(np); c->dGeoOut = deviceAlloc<Result>(ng);
            c->dH = deviceAlloc<double>(36); c->dB = deviceAlloc<double>(6); c->dCost = deviceAlloc<double>(1);
            upload(c->gray, gray, pixels); upload(c->gx, gx, pixels); upload(c->gy, gy, pixels);
            upload(c->points, points, pixels * 3); upload(c->normals, normals, pixels * 3);
            upload(c->photo, photo, np); upload(c->geo, geo, ng);
        }
        catch (...) { destroyLevel(c); throw; }
        return c;
    }
    void destroyLevel(LevelHandle* c) noexcept {
        if (!c)return;
        cudaFree(c->gray); cudaFree(c->gx); cudaFree(c->gy); cudaFree(c->points); cudaFree(c->normals);
        cudaFree(c->photo); cudaFree(c->geo); cudaFree(c->dPhotoOut); cudaFree(c->dGeoOut);
        cudaFree(c->dH); cudaFree(c->dB); cudaFree(c->dCost); delete c;
    }
    void evaluateNormalEquations(LevelHandle* c, const Pose& pose, GeoGate gate, int withJ,
        double infoPhoto, double infoGeo, int robustKind, double robustDelta,
        double* H, double* b, double* cost) {
        if (!c) throw std::invalid_argument("Null CUDA level handle");
        check(cudaMemset(c->dH, 0, 36 * sizeof(double)), "cudaMemset H");
        check(cudaMemset(c->dB, 0, 6 * sizeof(double)), "cudaMemset b");
        check(cudaMemset(c->dCost, 0, sizeof(double)), "cudaMemset cost");
        if (c->np) photoKernel << <(static_cast<int>(c->np) + BLOCK - 1) / BLOCK, BLOCK >> > (c->w, c->h, c->K, c->gray, c->gx, c->gy, c->photo, static_cast<int>(c->np), pose, withJ, c->dPhotoOut);
        if (c->ng) geoKernel << <(static_cast<int>(c->ng) + BLOCK - 1) / BLOCK, BLOCK >> > (c->w, c->h, c->K, c->points, c->normals, c->geo, static_cast<int>(c->ng), pose, gate, withJ, c->dGeoOut);
        check(cudaGetLastError(), "CUDA residual kernel launch");
        if (c->np) accumulateKernel << <(static_cast<int>(c->np) + BLOCK - 1) / BLOCK, BLOCK >> > (c->dPhotoOut, static_cast<int>(c->np), infoPhoto, robustKind, robustDelta, withJ, c->dH, c->dB, c->dCost);
        if (c->ng) accumulateKernel << <(static_cast<int>(c->ng) + BLOCK - 1) / BLOCK, BLOCK >> > (c->dGeoOut, static_cast<int>(c->ng), infoGeo, robustKind, robustDelta, withJ, c->dH, c->dB, c->dCost);
        check(cudaGetLastError(), "CUDA normal-equation kernel launch");
        check(cudaDeviceSynchronize(), "CUDA residual/normal-equation evaluation");
        download(cost, c->dCost, 1);
        if (withJ) { download(H, c->dH, 36); download(b, c->dB, 6); }
    }
} // namespace vo::cuda_backend
