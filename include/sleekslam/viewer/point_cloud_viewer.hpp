
#pragma once

// Thread-safe RGB-D visual odometry dashboard.
//
// Layout:
//   +-------------------------+-------------------------+
//   |                         |                         |
//   |  Live camera image      |  Global 3D map          |
//   |  RGB / grayscale /      |  Full trajectory,       |
//   |  depth (left panel)     |  accumulated points,    |
//   |                         |  current camera pose    |
//   |                         +-------------------------+
//   |                         |                         |
//   |                         |  First-person 3D view   |
//   |                         |  Camera-centred view of  |
//   |                         |  the accumulated cloud  |
//   +-------------------------+-------------------------+
//
// Controls:
//   1 : RGB image in left panel
//   2 : grayscale image in left panel
//   3 : depth visualization in left panel
//   p : toggle points in the global map
//   f : toggle points in the first-person view
//   Esc : close viewer
//
// Coordinate convention:
//   T_wc transforms camera-frame points into world coordinates.
//   Camera coordinates: +X right, +Y down, +Z forward.
//   First-person OpenGL coordinates: (X, -Y, -Z).
//
// Threading:
//   Tracking may call pushPose(), pushFrame(), and pushPoints().
//   All OpenGL operations happen in run() on the viewer thread.
//   Published point chunks and frames are immutable after publication.


#include <sleekslam/core/camera.hpp>

namespace vo {

    class TrajectoryViewer {
    public:
        enum class ViewMode : int {
            RGB = 1,
            Grayscale = 2,
            Depth = 3,
            Map3D = 4
        };

        explicit TrajectoryViewer(
            std::size_t maxPoints = 1'500'000,
            double maxRenderFps = 30.0)
            : maxPoints_(maxPoints),
            maxRenderFps_(std::max(1.0, maxRenderFps)) {
        }

        TrajectoryViewer(const TrajectoryViewer&) = delete;
        TrajectoryViewer& operator=(const TrajectoryViewer&) = delete;

        ~TrajectoryViewer() = default;

        // Publish the latest world-from-camera pose.
        void pushPose(const SE3& T_wc) {
            std::lock_guard<std::mutex> lock(mutex_);

            latestPose_ = T_wc;
            hasPose_ = true;
            trajectory_.push_back(T_wc.translation());

            // Invalidate the first-person cache so it follows the latest pose.
            firstPersonDirty_ = true;
        }

        // Publish the current RGB-D frame for the left panel.
        //
        // Expected:
        //   colorBgr : CV_8UC3, BGR order
        //   gray32f  : CV_32FC1, intensity range 0..255
        //   depth32f : CV_32FC1, depth in metres; 0 = invalid
        //
        // These cv::Mat headers are shallow copies. Do not modify the underlying
        // pixel buffers after publishing them.
        void pushFrame(
            const cv::Mat& colorBgr,
            const cv::Mat& gray32f,
            const cv::Mat& depth32f) {

            if (colorBgr.empty() || gray32f.empty() || depth32f.empty()) {
                return;
            }

            auto frame = std::make_shared<FrameImages>();
            frame->colorBgr = colorBgr;
            frame->gray32f = gray32f;
            frame->depth32f = depth32f;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                latestFrame_ = std::move(frame);
                ++frameVersion_;
            }
        }

        // Append a camera-frame cloud after transforming it into world space.
        // The input span may refer to a temporary vector.
        void pushPoints(
            std::span<const ColoredPoint> cameraPoints,
            const SE3& T_wc) {

            if (cameraPoints.empty()) {
                return;
            }

            std::size_t remaining = 0;

            {
                std::lock_guard<std::mutex> lock(mutex_);

                if (pointCount_ >= maxPoints_) {
                    return;
                }

                remaining = maxPoints_ - pointCount_;
            }

            const std::size_t count =
                std::min(cameraPoints.size(), remaining);

            auto worldPoints =
                std::make_shared<std::vector<ColoredPoint>>();

            worldPoints->reserve(count);

            for (std::size_t i = 0; i < count; ++i) {
                const ColoredPoint& source = cameraPoints[i];

                const Eigen::Vector3d pWorld =
                    T_wc * Eigen::Vector3d(
                        static_cast<double>(source.x),
                        static_cast<double>(source.y),
                        static_cast<double>(source.z));

                ColoredPoint p = source;
                p.x = static_cast<decltype(p.x)>(pWorld.x());
                p.y = static_cast<decltype(p.y)>(pWorld.y());
                p.z = static_cast<decltype(p.z)>(pWorld.z());

                worldPoints->push_back(p);
            }

            if (worldPoints->empty()) {
                return;
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);

                // Recheck capacity in case another producer published points.
                const std::size_t available =
                    pointCount_ < maxPoints_
                    ? maxPoints_ - pointCount_
                    : 0;

                if (available == 0) {
                    return;
                }

                if (worldPoints->size() > available) {
                    worldPoints->resize(available);
                }

                if (worldPoints->empty()) {
                    return;
                }

                PointChunk chunk;
                chunk.firstIndex = pointCount_;
                chunk.points = std::move(worldPoints);

                pointCount_ += chunk.points->size();
                pointChunks_.push_back(std::move(chunk));

                firstPersonDirty_ = true;
            }
        }

        bool closed() const noexcept {
            return closed_.load(std::memory_order_relaxed);
        }

        // Kept for compatibility with existing code.
        // The layout remains three panels regardless of this setting.
        ViewMode viewMode() const noexcept {
            return viewMode_.load(std::memory_order_relaxed);
        }

        void setViewMode(ViewMode mode) noexcept {
            viewMode_.store(mode, std::memory_order_relaxed);
        }

        void run() {
            closed_.store(false, std::memory_order_relaxed);

            pangolin::CreateWindowAndBind(
                "RGB-D Visual Odometry", 1440, 900);

            glewExperimental = GL_TRUE;
            const GLenum glewStatus = glewInit();

            if (glewStatus != GLEW_OK) {
                std::fprintf(
                    stderr,
                    "GLEW initialization failed: %s\n",
                    glewGetErrorString(glewStatus));

                closed_.store(true, std::memory_order_relaxed);
                return;
            }

            // Clear any error that GLEW may leave on some drivers.
            while (glGetError() != GL_NO_ERROR) {
            }

            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            // -----------------------------
            // Global 3D map camera
            // -----------------------------
            pangolin::OpenGlRenderState globalCamera(
                pangolin::ProjectionMatrix(
                    1440, 900,
                    700.0, 700.0,
                    720.0, 450.0,
                    0.05, 1000.0),
                pangolin::ModelViewLookAt(
                    0.0, -3.0, -6.0,
                    0.0, 0.0, 0.0,
                    pangolin::AxisNegY));

            // -----------------------------
            // Three persistent panels
            // -----------------------------
            pangolin::View& imageView =
                pangolin::Display("vo_image")
                .SetBounds(0.0, 1.0, 0.0, 0.48);

            pangolin::View& globalView =
                pangolin::Display("vo_global_map")
                .SetBounds(0.50, 1.0, 0.48, 1.0)
                .SetHandler(new pangolin::Handler3D(globalCamera));

            pangolin::View& firstPersonView =
                pangolin::Display("vo_first_person")
                .SetBounds(0.0, 0.50, 0.48, 1.0);

            pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0)
                .AddDisplay(imageView)
                .AddDisplay(globalView)
                .AddDisplay(firstPersonView);

            // -----------------------------
            // OpenGL resources
            // -----------------------------
            createImageTexture();

            glGenBuffers(1, &globalPointVbo_);
            glBindBuffer(GL_ARRAY_BUFFER, globalPointVbo_);
            glBufferData(
                GL_ARRAY_BUFFER,
                static_cast<GLsizeiptr>(
                    maxPoints_ * sizeof(ColoredPoint)),
                nullptr,
                GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            glGenBuffers(1, &firstPersonVbo_);
            glBindBuffer(GL_ARRAY_BUFFER, firstPersonVbo_);
            glBufferData(
                GL_ARRAY_BUFFER,
                static_cast<GLsizeiptr>(
                    maxFirstPersonPoints_ * sizeof(ColoredPoint)),
                nullptr,
                GL_STREAM_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            // -----------------------------
            // Keyboard controls
            // -----------------------------
            pangolin::RegisterKeyPressCallback(
                '1', [this]() {
                    setViewMode(ViewMode::RGB);
                });

            pangolin::RegisterKeyPressCallback(
                '2', [this]() {
                    setViewMode(ViewMode::Grayscale);
                });

            pangolin::RegisterKeyPressCallback(
                '3', [this]() {
                    setViewMode(ViewMode::Depth);
                });

            pangolin::RegisterKeyPressCallback(
                'p', [this]() {
                    showGlobalPoints_.store(
                        !showGlobalPoints_.load(
                            std::memory_order_relaxed),
                        std::memory_order_relaxed);
                });

            pangolin::RegisterKeyPressCallback(
                'f', [this]() {
                    showFirstPersonPoints_.store(
                        !showFirstPersonPoints_.load(
                            std::memory_order_relaxed),
                        std::memory_order_relaxed);
                });

            printControls();

            using Clock = std::chrono::steady_clock;

            const auto framePeriod =
                std::chrono::duration<double>(1.0 / maxRenderFps_);

            const auto firstPersonRefreshPeriod =
                std::chrono::milliseconds(100);

            auto nextFrameTime = Clock::now();
            auto lastFirstPersonRefresh = Clock::time_point{};

            while (!pangolin::ShouldQuit()) {
                nextFrameTime +=
                    std::chrono::duration_cast<Clock::duration>(
                        framePeriod);

                Snapshot snapshot = takeSnapshot();

                glClearColor(0.035f, 0.035f, 0.045f, 1.0f);
                glClear(
                    GL_COLOR_BUFFER_BIT |
                    GL_DEPTH_BUFFER_BIT);

                // Panel 1: image used by the odometry pipeline.
                drawImagePanel(imageView, snapshot);

                // Panel 2: global map and complete trajectory.
                globalView.Activate(globalCamera);
                drawGlobalMap(snapshot);

                // Refresh the first-person cloud at a lower rate than the UI.
                // This avoids transforming the full accumulated cloud every
                // rendered frame.
                const auto now = Clock::now();

                if (snapshot.hasPose &&
                    (snapshot.firstPersonDirty ||
                        now - lastFirstPersonRefresh >=
                        firstPersonRefreshPeriod)) {

                    rebuildFirstPersonCloud(snapshot);
                    lastFirstPersonRefresh = now;
                }

                // Panel 3: current camera perspective.
                firstPersonView.Activate();
                drawFirstPersonPanel();

                pangolin::FinishFrame();

                const auto afterRender = Clock::now();

                if (nextFrameTime > afterRender) {
                    std::this_thread::sleep_until(nextFrameTime);
                }
                else {
                    // Avoid accumulating timing debt after a slow render.
                    nextFrameTime = afterRender;
                }
            }

            destroyGlResources();

            closed_.store(true, std::memory_order_relaxed);
        }

    private:
        struct FrameImages {
            cv::Mat colorBgr;
            cv::Mat gray32f;
            cv::Mat depth32f;
        };

        struct PointChunk {
            std::size_t firstIndex = 0;
            std::shared_ptr<const std::vector<ColoredPoint>> points;
        };

        struct Snapshot {
            std::vector<PointChunk> pointChunks;
            std::vector<Eigen::Vector3d> trajectory;
            std::shared_ptr<const FrameImages> frame;

            SE3 pose;
            bool hasPose = false;

            std::uint64_t frameVersion = 0;
            bool firstPersonDirty = false;
            std::size_t pointCount = 0;
        };

        static constexpr std::size_t maxFirstPersonPoints_ = 180'000;

        static void printControls() {
            std::puts("");
            std::puts("RGB-D Visual Odometry dashboard");
            std::puts("  Left panel : live image");
            std::puts("  Top-right  : global 3D map and trajectory");
            std::puts("  Bottom-right: first-person 3D point cloud");
            std::puts("");
            std::puts("  1 : RGB image");
            std::puts("  2 : grayscale image");
            std::puts("  3 : depth visualization");
            std::puts("  p : toggle global-map points");
            std::puts("  f : toggle first-person points");
            std::puts("  Esc : close viewer");
            std::puts("");
        }

        Snapshot takeSnapshot() {
            Snapshot snapshot;

            std::lock_guard<std::mutex> lock(mutex_);

            // These are shared references to immutable point chunks, not copies
            // of all accumulated point data.
            snapshot.pointChunks = pointChunks_;
            snapshot.trajectory = trajectory_;
            snapshot.frame = latestFrame_;
            snapshot.pose = latestPose_;
            snapshot.hasPose = hasPose_;
            snapshot.frameVersion = frameVersion_;
            snapshot.firstPersonDirty = firstPersonDirty_;
            snapshot.pointCount = pointCount_;

            firstPersonDirty_ = false;

            return snapshot;
        }

        // -----------------------------
        // Image panel
        // -----------------------------
        void createImageTexture() {
            glGenTextures(1, &imageTexture_);
            glBindTexture(GL_TEXTURE_2D, imageTexture_);

            glTexParameteri(
                GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(
                GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(
                GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(
                GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

            glBindTexture(GL_TEXTURE_2D, 0);
        }

        void drawImagePanel(
            pangolin::View& view,
            const Snapshot& snapshot) {

            view.Activate();

            if (!snapshot.frame) {
                drawPanelMessage("Waiting for camera frames");
                return;
            }

            cv::Mat image;

            const ViewMode mode =
                viewMode_.load(std::memory_order_relaxed);

            switch (mode) {
            case ViewMode::RGB:
                image = snapshot.frame->colorBgr;
                break;

            case ViewMode::Depth:
                makeDepthDisplay(
                    snapshot.frame->depth32f,
                    depthDisplay_);
                image = depthDisplay_;
                break;

            case ViewMode::Grayscale:
            case ViewMode::Map3D:
            default:
                if (snapshot.frame->gray32f.type() == CV_8UC1) {
                    image = snapshot.frame->gray32f;
                }
                else {
                    snapshot.frame->gray32f.convertTo(
                        grayDisplay_, CV_8UC1);
                    image = grayDisplay_;
                }
                break;
            }

            if (image.empty()) {
                drawPanelMessage("No image available");
                return;
            }

            uploadImage(image, mode);

            glDisable(GL_DEPTH_TEST);
            glDisable(GL_LIGHTING);
            glDisable(GL_CULL_FACE);

            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();

            const double imageAspect =
                static_cast<double>(image.cols) /
                static_cast<double>(image.rows);

            const double panelAspect =
                static_cast<double>(std::max(1, view.v.w)) /
                static_cast<double>(std::max(1, view.v.h));

            double halfWidth = 1.0;
            double halfHeight = 1.0;

            if (imageAspect > panelAspect) {
                halfHeight = panelAspect / imageAspect;
            }
            else {
                halfWidth = imageAspect / panelAspect;
            }

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, imageTexture_);
            glColor4f(1.f, 1.f, 1.f, 1.f);

            glBegin(GL_QUADS);
            glTexCoord2f(0.f, 1.f);
            glVertex2d(-halfWidth, -halfHeight);

            glTexCoord2f(1.f, 1.f);
            glVertex2d(halfWidth, -halfHeight);

            glTexCoord2f(1.f, 0.f);
            glVertex2d(halfWidth, halfHeight);

            glTexCoord2f(0.f, 0.f);
            glVertex2d(-halfWidth, halfHeight);
            glEnd();

            glBindTexture(GL_TEXTURE_2D, 0);
            glDisable(GL_TEXTURE_2D);

            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);

            glEnable(GL_DEPTH_TEST);
        }

        void uploadImage(
            const cv::Mat& image,
            ViewMode mode) {

            if (image.empty()) {
                return;
            }

            GLenum inputFormat = GL_LUMINANCE;
            GLint internalFormat = GL_LUMINANCE8;

            if (mode == ViewMode::RGB) {
                CV_Assert(image.type() == CV_8UC3);
                inputFormat = GL_BGR;
                internalFormat = GL_RGB8;
            }
            else {
                CV_Assert(image.type() == CV_8UC1);
            }

            // OpenGL expects consecutive rows when no row length is specified.
            cv::Mat contiguous;
            const cv::Mat* upload = &image;

            if (!image.isContinuous()) {
                contiguous = image.clone();
                upload = &contiguous;
            }

            glBindTexture(GL_TEXTURE_2D, imageTexture_);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            const bool resized =
                textureWidth_ != image.cols ||
                textureHeight_ != image.rows ||
                textureMode_ != static_cast<int>(mode);

            if (resized) {
                glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    internalFormat,
                    image.cols,
                    image.rows,
                    0,
                    inputFormat,
                    GL_UNSIGNED_BYTE,
                    upload->data);

                textureWidth_ = image.cols;
                textureHeight_ = image.rows;
                textureMode_ = static_cast<int>(mode);
            }
            else {
                glTexSubImage2D(
                    GL_TEXTURE_2D,
                    0,
                    0, 0,
                    image.cols,
                    image.rows,
                    inputFormat,
                    GL_UNSIGNED_BYTE,
                    upload->data);
            }

            glBindTexture(GL_TEXTURE_2D, 0);
        }

        static void makeDepthDisplay(
            const cv::Mat& depth32f,
            cv::Mat& output) {

            if (depth32f.empty()) {
                output.release();
                return;
            }

            CV_Assert(depth32f.type() == CV_32FC1);

            constexpr float nearMeters = 0.30f;
            constexpr float farMeters = 5.00f;
            constexpr float scale =
                255.0f / (farMeters - nearMeters);

            output.create(depth32f.size(), CV_8UC1);

            for (int y = 0; y < depth32f.rows; ++y) {
                const float* src = depth32f.ptr<float>(y);
                std::uint8_t* dst = output.ptr<std::uint8_t>(y);

                for (int x = 0; x < depth32f.cols; ++x) {
                    const float z = src[x];

                    if (!std::isfinite(z) || z <= 0.0f) {
                        dst[x] = 0;
                        continue;
                    }

                    const float value = (farMeters - z) * scale;

                    dst[x] = static_cast<std::uint8_t>(
                        std::clamp(value, 0.0f, 255.0f));
                }
            }
        }

        static void drawPanelMessage(const char* /*message*/) {
            // Keep the panel clean while no frame is available.
            glDisable(GL_DEPTH_TEST);
            glClearColor(0.035f, 0.035f, 0.045f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
        }

        // -----------------------------
        // Global map panel
        // -----------------------------
        void drawGlobalMap(const Snapshot& snapshot) {
            drawGroundGrid();
            drawTrajectory(snapshot.trajectory);

            if (showGlobalPoints_.load(std::memory_order_relaxed)) {
                drawGlobalPoints(snapshot.pointChunks);
            }

            if (snapshot.hasPose) {
                drawCameraPose(snapshot.pose);
            }
        }

        static void drawGroundGrid() {
            glDisable(GL_LIGHTING);
            glLineWidth(1.0f);
            glColor3f(0.20f, 0.20f, 0.23f);

            glBegin(GL_LINES);

            constexpr int extent = 20;
            constexpr float step = 0.5f;

            for (int i = -extent; i <= extent; ++i) {
                const float p = static_cast<float>(i) * step;

                glVertex3f(p, 0.0f, -extent * step);
                glVertex3f(p, 0.0f, extent * step);

                glVertex3f(-extent * step, 0.0f, p);
                glVertex3f(extent * step, 0.0f, p);
            }

            glEnd();
        }

        static void drawTrajectory(
            const std::vector<Eigen::Vector3d>& trajectory) {

            if (trajectory.size() < 2) {
                return;
            }

            glDisable(GL_LIGHTING);
            glDisable(GL_TEXTURE_2D);

            // Solid, thick cyan trajectory
            glDisable(GL_LINE_STIPPLE);
            glLineWidth(4.0f);
            glColor3f(0.0f, 0.95f, 1.0f);

            glBegin(GL_LINE_STRIP);
            for (const auto& p : trajectory) {
                glVertex3d(p.x(), p.y(), p.z());
            }
            glEnd();

            glDisable(GL_LINE_STIPPLE);


            // Mark the starting position.
            const auto& start = trajectory.front();

            glPointSize(7.0f);
            glColor3f(0.2f, 1.0f, 0.3f);

            glBegin(GL_POINTS);
            glVertex3d(start.x(), start.y(), start.z());
            glEnd();
        }

        void drawGlobalPoints(
            const std::vector<PointChunk>& chunks) {

            if (chunks.empty() || globalPointVbo_ == 0) {
                return;
            }

            glBindBuffer(GL_ARRAY_BUFFER, globalPointVbo_);

            // Incrementally upload only the point data not yet on the GPU.
            for (const auto& chunk : chunks) {
                if (!chunk.points || chunk.points->empty()) {
                    continue;
                }

                const std::size_t endIndex =
                    chunk.firstIndex + chunk.points->size();

                if (endIndex <= uploadedGlobalPointCount_) {
                    continue;
                }

                const std::size_t skip =
                    uploadedGlobalPointCount_ > chunk.firstIndex
                    ? uploadedGlobalPointCount_ - chunk.firstIndex
                    : 0;

                if (skip >= chunk.points->size()) {
                    continue;
                }

                const std::size_t uploadCount =
                    chunk.points->size() - skip;

                const std::size_t uploadStart =
                    chunk.firstIndex + skip;

                glBufferSubData(
                    GL_ARRAY_BUFFER,
                    static_cast<GLintptr>(
                        uploadStart * sizeof(ColoredPoint)),
                    static_cast<GLsizeiptr>(
                        uploadCount * sizeof(ColoredPoint)),
                    chunk.points->data() + skip);

                uploadedGlobalPointCount_ =
                    std::max(uploadedGlobalPointCount_, endIndex);
            }

            const GLsizei count = static_cast<GLsizei>(
                std::min(uploadedGlobalPointCount_, maxPoints_));

            if (count > 0) {
                glEnableClientState(GL_VERTEX_ARRAY);
                glEnableClientState(GL_COLOR_ARRAY);

                glVertexPointer(
                    3, GL_FLOAT, sizeof(ColoredPoint),
                    reinterpret_cast<const void*>(
                        offsetof(ColoredPoint, x)));

                glColorPointer(
                    3, GL_UNSIGNED_BYTE, sizeof(ColoredPoint),
                    reinterpret_cast<const void*>(
                        offsetof(ColoredPoint, r)));

                glPointSize(2.0f);
                glDrawArrays(GL_POINTS, 0, count);

                glDisableClientState(GL_COLOR_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);
            }

            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        static void drawCameraPose(const SE3& T_wc) {
            const Eigen::Vector3d origin = T_wc.translation();
            const Eigen::Matrix3d R = T_wc.so3().matrix();

            constexpr double axisLength = 0.20;

            const Eigen::Vector3d xEnd =
                origin + R * Eigen::Vector3d(axisLength, 0, 0);

            const Eigen::Vector3d yEnd =
                origin + R * Eigen::Vector3d(0, axisLength, 0);

            const Eigen::Vector3d zEnd =
                origin + R * Eigen::Vector3d(0, 0, axisLength);

            glLineWidth(3.0f);
            glBegin(GL_LINES);

            glColor3f(1.f, 0.1f, 0.1f);
            glVertex3d(origin.x(), origin.y(), origin.z());
            glVertex3d(xEnd.x(), xEnd.y(), xEnd.z());

            glColor3f(0.1f, 1.f, 0.1f);
            glVertex3d(origin.x(), origin.y(), origin.z());
            glVertex3d(yEnd.x(), yEnd.y(), yEnd.z());

            glColor3f(0.1f, 0.3f, 1.f);
            glVertex3d(origin.x(), origin.y(), origin.z());
            glVertex3d(zEnd.x(), zEnd.y(), zEnd.z());

            glEnd();
        }

        // -----------------------------
        // First-person point-cloud panel
        // -----------------------------
        void rebuildFirstPersonCloud(const Snapshot& snapshot) {
            firstPersonPoints_.clear();

            if (!snapshot.hasPose || snapshot.pointCount == 0) {
                return;
            }

            // Convert world points into the latest camera frame.
            const SE3 T_cw = snapshot.pose.inverse();

            // The first-person view uses a perspective projection. Keep only
            // points in front of the camera and within a practical viewing range.
            constexpr double minDepth = 0.10;
            constexpr double maxDepth = 8.0;
            constexpr double verticalHalfFovRadians = 35.0 * 3.141592653589793 / 180.0;

            const double tanVertical =
                std::tan(verticalHalfFovRadians);

            const double panelAspect =
                static_cast<double>(std::max(1, lastFirstPersonWidth_)) /
                static_cast<double>(std::max(1, lastFirstPersonHeight_));

            const double tanHorizontal = tanVertical * panelAspect;

            firstPersonPoints_.reserve(
                std::min(snapshot.pointCount, maxFirstPersonPoints_));

            // Sampling limits the first-person buffer size even when the global
            // map contains hundreds of thousands of points.
            const std::size_t stride = std::max<std::size_t>(
                1,
                (snapshot.pointCount + maxFirstPersonPoints_ - 1) /
                maxFirstPersonPoints_);

            std::size_t globalIndex = 0;

            for (const auto& chunk : snapshot.pointChunks) {
                if (!chunk.points || chunk.points->empty()) {
                    continue;
                }

                for (std::size_t i = 0; i < chunk.points->size(); ++i) {
                    const std::size_t index = globalIndex++;

                    if ((index % stride) != 0) {
                        continue;
                    }

                    const ColoredPoint& pWorld = (*chunk.points)[i];

                    const Eigen::Vector3d pCamera =
                        T_cw * Eigen::Vector3d(
                            static_cast<double>(pWorld.x),
                            static_cast<double>(pWorld.y),
                            static_cast<double>(pWorld.z));

                    const double x = pCamera.x();
                    const double y = pCamera.y();
                    const double z = pCamera.z();

                    if (!std::isfinite(x) ||
                        !std::isfinite(y) ||
                        !std::isfinite(z)) {
                        continue;
                    }

                    if (z < minDepth || z > maxDepth) {
                        continue;
                    }

                    // Camera-frame +Y points down; the OpenGL camera's +Y
                    // points up. The projection looks down OpenGL -Z.
                    if (std::abs(x) > z * tanHorizontal ||
                        std::abs(y) > z * tanVertical) {
                        continue;
                    }

                    ColoredPoint p = pWorld;
                    p.x = static_cast<decltype(p.x)>(x);
                    p.y = static_cast<decltype(p.y)>(-y);
                    p.z = static_cast<decltype(p.z)>(-z);

                    firstPersonPoints_.push_back(p);

                    if (firstPersonPoints_.size() >=
                        maxFirstPersonPoints_) {
                        break;
                    }
                }

                if (firstPersonPoints_.size() >=
                    maxFirstPersonPoints_) {
                    break;
                }
            }

            firstPersonDirty_ = false;
            uploadFirstPersonCloud();
        }

        void uploadFirstPersonCloud() {
            if (firstPersonVbo_ == 0) {
                return;
            }

            glBindBuffer(GL_ARRAY_BUFFER, firstPersonVbo_);

            const std::size_t count = std::min(
                firstPersonPoints_.size(),
                maxFirstPersonPoints_);

            if (count > 0) {
                glBufferData(
                    GL_ARRAY_BUFFER,
                    static_cast<GLsizeiptr>(
                        count * sizeof(ColoredPoint)),
                    firstPersonPoints_.data(),
                    GL_STREAM_DRAW);
            }
            else {
                glBufferData(
                    GL_ARRAY_BUFFER,
                    0,
                    nullptr,
                    GL_STREAM_DRAW);
            }

            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        void drawFirstPersonPanel() {
            lastFirstPersonWidth_ = std::max(1, pangolin::DisplayBase().v.w);
            lastFirstPersonHeight_ = std::max(1, pangolin::DisplayBase().v.h);

            glDisable(GL_LIGHTING);
            glEnable(GL_DEPTH_TEST);

            // This panel intentionally has no world trajectory or orbit camera.
            // The camera is fixed at the origin, looking along camera +Z,
            // represented as OpenGL -Z.
            const double aspect =
                static_cast<double>(std::max(1, lastFirstPersonWidth_)) /
                static_cast<double>(std::max(1, lastFirstPersonHeight_));

            constexpr double nearPlane = 0.05;
            constexpr double farPlane = 12.0;
            constexpr double verticalFovRadians =
                70.0 * 3.141592653589793 / 180.0;

            const double halfHeight =
                nearPlane * std::tan(verticalFovRadians * 0.5);

            const double halfWidth = halfHeight * aspect;

            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glFrustum(
                -halfWidth, halfWidth,
                -halfHeight, halfHeight,
                nearPlane, farPlane);

            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();

            // A small fixed camera-axis indicator at the origin.
            drawFirstPersonAxes();

            if (showFirstPersonPoints_.load(std::memory_order_relaxed) &&
                !firstPersonPoints_.empty() &&
                firstPersonVbo_ != 0) {

                glBindBuffer(GL_ARRAY_BUFFER, firstPersonVbo_);

                glEnableClientState(GL_VERTEX_ARRAY);
                glEnableClientState(GL_COLOR_ARRAY);

                glVertexPointer(
                    3, GL_FLOAT, sizeof(ColoredPoint),
                    reinterpret_cast<const void*>(
                        offsetof(ColoredPoint, x)));

                glColorPointer(
                    3, GL_UNSIGNED_BYTE, sizeof(ColoredPoint),
                    reinterpret_cast<const void*>(
                        offsetof(ColoredPoint, r)));

                glPointSize(2.0f);

                glDrawArrays(
                    GL_POINTS,
                    0,
                    static_cast<GLsizei>(firstPersonPoints_.size()));

                glDisableClientState(GL_COLOR_ARRAY);
                glDisableClientState(GL_VERTEX_ARRAY);

                glBindBuffer(GL_ARRAY_BUFFER, 0);
            }

            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
        }

        static void drawFirstPersonAxes() {
            // Short axes centred at the camera. They help orient the viewer
            // without drawing the full trajectory in this panel.
            constexpr float length = 0.12f;

            glLineWidth(2.0f);
            glBegin(GL_LINES);

            glColor3f(1.f, 0.2f, 0.2f);
            glVertex3f(0.f, 0.f, 0.f);
            glVertex3f(length, 0.f, 0.f);

            glColor3f(0.2f, 1.f, 0.2f);
            glVertex3f(0.f, 0.f, 0.f);
            glVertex3f(0.f, length, 0.f);

            glColor3f(0.2f, 0.4f, 1.f);
            glVertex3f(0.f, 0.f, 0.f);
            glVertex3f(0.f, 0.f, -length);

            glEnd();
        }

        void destroyGlResources() {
            if (imageTexture_ != 0) {
                glDeleteTextures(1, &imageTexture_);
                imageTexture_ = 0;
            }

            if (globalPointVbo_ != 0) {
                glDeleteBuffers(1, &globalPointVbo_);
                globalPointVbo_ = 0;
            }

            if (firstPersonVbo_ != 0) {
                glDeleteBuffers(1, &firstPersonVbo_);
                firstPersonVbo_ = 0;
            }
        }

    private:
        const std::size_t maxPoints_;
        const double maxRenderFps_;

        mutable std::mutex mutex_;

        std::shared_ptr<const FrameImages> latestFrame_;
        std::uint64_t frameVersion_ = 0;

        std::vector<PointChunk> pointChunks_;
        std::vector<Eigen::Vector3d> trajectory_;
        std::size_t pointCount_ = 0;

        SE3 latestPose_;
        bool hasPose_ = false;
        bool firstPersonDirty_ = true;

        std::atomic<ViewMode> viewMode_{ ViewMode::Grayscale };
        std::atomic<bool> showGlobalPoints_{ true };
        std::atomic<bool> showFirstPersonPoints_{ true };
        std::atomic<bool> closed_{ true };

        // OpenGL resources: accessed only from run().
        GLuint imageTexture_ = 0;
        GLuint globalPointVbo_ = 0;
        GLuint firstPersonVbo_ = 0;

        int textureWidth_ = 0;
        int textureHeight_ = 0;
        int textureMode_ = -1;

        std::size_t uploadedGlobalPointCount_ = 0;

        // Viewer-thread-only scratch buffers.
        cv::Mat grayDisplay_;
        cv::Mat depthDisplay_;
        std::vector<ColoredPoint> firstPersonPoints_;

        // The panel dimensions are updated by the viewer thread.
        int lastFirstPersonWidth_ = 640;
        int lastFirstPersonHeight_ = 360;
    };

} // namespace vo
