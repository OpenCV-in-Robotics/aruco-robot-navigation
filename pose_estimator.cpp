// ============================================================================
//  SECTION 2 — POSE ESTIMATOR
//  Member: Krish
//
//  KEY METHOD ANALYSED: estimatePoseSingleMarkers()
//  In the very latest OpenCV builds this function was removed.
//  FIX: We implement it manually using cv::solvePnP — which is exactly
//  what estimatePoseSingleMarkers() called internally anyway.
//  Same result, same accuracy, works on all OpenCV versions.
// ============================================================================

struct CameraParams {
    cv::Mat  cameraMatrix;
    cv::Mat  distCoeffs;
    cv::Size imageSize;

    static std::optional<CameraParams> fromFile(const std::string& path) {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) return std::nullopt;
        CameraParams p;
        fs["camera_matrix"]           >> p.cameraMatrix;
        fs["distortion_coefficients"] >> p.distCoeffs;
        int w=0, h=0;
        fs["image_width"]  >> w;
        fs["image_height"] >> h;
        p.imageSize = {w, h};
        return (p.cameraMatrix.empty() || p.distCoeffs.empty())
               ? std::nullopt : std::optional<CameraParams>(p);
    }

    void save(const std::string& path) const {
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        fs << "camera_matrix"           << cameraMatrix
           << "distortion_coefficients" << distCoeffs
           << "image_width"             << imageSize.width
           << "image_height"            << imageSize.height;
    }

    static CameraParams defaultWebcam() {
        CameraParams p;
        p.imageSize    = {640, 480};
        p.cameraMatrix = (cv::Mat_<double>(3,3) <<
            600,   0, 320,
              0, 600, 240,
              0,   0,   1);
        p.distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
        return p;
    }
};

struct MarkerPose {
    int       id;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat   R;
    double    distance;
    double    roll, pitch, yaw;
};

class PoseEstimator {
public:
    PoseEstimator(const CameraParams& cam, double markerLength = 0.05)
        : cam_(cam), markerLength_(markerLength)
    {
        // Build 3D object points for a square marker centred at origin.
        // This is exactly what estimatePoseSingleMarkers() used internally.
        // Half-length of marker side:
        float h = static_cast<float>(markerLength_ / 2.0);
        // Four corners in marker's local frame (Z=0, flat on table)
        objPoints_.push_back({-h,  h, 0});
        objPoints_.push_back({ h,  h, 0});
        objPoints_.push_back({ h, -h, 0});
        objPoints_.push_back({-h, -h, 0});
    }

    std::vector<MarkerPose> estimate(const DetectionResult& det) const {
        std::vector<MarkerPose> out;
        if (det.markerIds.empty()) return out;

        out.reserve(det.markerIds.size());
        for (size_t i = 0; i < det.markerIds.size(); ++i) {
            // solvePnP: given 4 known 3D points and their 2D projections,
            // recover the rotation + translation that explains the projection.
            // SOLVEPNP_IPPE_SQUARE is the same algorithm that
            // estimatePoseSingleMarkers() used — closed-form, fast, accurate.
            cv::Mat rvec, tvec;
            cv::solvePnP(
                objPoints_,
                det.markerCorners[i],
                cam_.cameraMatrix,
                cam_.distCoeffs,
                rvec, tvec,
                false,
                cv::SOLVEPNP_IPPE_SQUARE
            );

            MarkerPose mp;
            mp.id   = det.markerIds[i];
            mp.rvec = rvec;
            mp.tvec = tvec;
            cv::Rodrigues(rvec, mp.R);
            mp.distance = cv::norm(tvec);
            toEuler(mp.rvec, mp.roll, mp.pitch, mp.yaw);
            out.push_back(mp);
        }
        return out;
    }

    cv::Mat visualise(const cv::Mat& frame,
                      const std::vector<MarkerPose>& poses) const {
        cv::Mat out = frame.clone();
        for (const auto& mp : poses) {
            // NEW API: cv::aruco::drawAxis is removed.
            // Replacement: cv::drawFrameAxes (in calib3d)
            cv::drawFrameAxes(out,
                              cam_.cameraMatrix, cam_.distCoeffs,
                              mp.rvec, mp.tvec,
                              static_cast<float>(markerLength_ * 0.5f));

            std::vector<cv::Point3f> obj3 = {{0,0,0}};
            std::vector<cv::Point2f> img2;
            cv::projectPoints(obj3, mp.rvec, mp.tvec,
                              cam_.cameraMatrix, cam_.distCoeffs, img2);
            if (!img2.empty()) {
                cv::Point pt((int)img2[0].x, (int)img2[0].y);
                char buf[64];
                std::snprintf(buf, sizeof(buf),
                              "ID%d  %.2fm", mp.id, mp.distance);
                cv::putText(out, buf, pt+cv::Point(6,-6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,255}, 2);
                std::snprintf(buf, sizeof(buf), "yaw %.1fdeg", mp.yaw);
                cv::putText(out, buf, pt+cv::Point(6,16),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, {255,200,0}, 1);
            }
        }
        return out;
    }

    const CameraParams& camera()       const { return cam_; }
    double              markerLength() const { return markerLength_; }

private:
    CameraParams              cam_;
    double                    markerLength_;
    std::vector<cv::Point3f>  objPoints_;  // 3D corners of one marker

    static void toEuler(const cv::Vec3d& rvec,
                        double& roll, double& pitch, double& yaw) {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        pitch = std::atan2(-R.at<double>(2,0),
                            std::hypot(R.at<double>(2,1),
                                       R.at<double>(2,2)));
        roll  = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        yaw   = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
        const double r2d = 180.0 / CV_PI;
        roll  *= r2d;
        pitch *= r2d;
        yaw   *= r2d;
    }
};

