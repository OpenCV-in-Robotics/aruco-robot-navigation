// ============================================================================
//  SECTION 3 — TRAJECTORY PREDICTOR  (NEW METHOD — Mitul)
//  Kalman filter per marker. State = [px,py,pz,vx,vy,vz].
//  Smooths noisy pose + predicts 10 future positions.
// ============================================================================

struct PredictionResult {
    int                      markerId;
    cv::Point3d              smoothedPos;
    cv::Point3d              velocity;
    std::vector<cv::Point3d> futurePath;
    double                   horizonMs;
    bool                     reliable;
};

struct KFState {
    cv::KalmanFilter        kf;
    bool                    initialised = false;
    double                  lastMs      = 0.0;
    std::deque<cv::Point3d> history;
    static constexpr size_t MAX_HIST = 60;
};

class TrajectoryPredictor {
public:
    explicit TrajectoryPredictor(int    steps  = 10,
                                 double pNoise = 1e-4,
                                 double mNoise = 1e-2)
        : steps_(steps), pNoise_(pNoise), mNoise_(mNoise) {}

    std::vector<PredictionResult> update(
        const std::vector<MarkerPose>& poses, double nowMs)
    {
        std::vector<PredictionResult> results;
        for (const auto& mp : poses) {
            auto& s = kfs_[mp.id];
            cv::Point3d meas(mp.tvec[0], mp.tvec[1], mp.tvec[2]);

            if (!s.initialised) { initKF(s, meas); s.lastMs = nowMs; }

            double dt = clampDt(s.lastMs, nowMs);
            s.lastMs  = nowMs;
            s.kf.transitionMatrix.at<double>(0,3) = dt;
            s.kf.transitionMatrix.at<double>(1,4) = dt;
            s.kf.transitionMatrix.at<double>(2,5) = dt;

            s.kf.predict();

            cv::Mat z = (cv::Mat_<double>(3,1) << meas.x, meas.y, meas.z);
            cv::Mat corrected = s.kf.correct(z);

            cv::Point3d smoothed(corrected.at<double>(0),
                                 corrected.at<double>(1),
                                 corrected.at<double>(2));
            cv::Point3d vel(corrected.at<double>(3),
                            corrected.at<double>(4),
                            corrected.at<double>(5));

            s.history.push_back(meas);
            if (s.history.size() > KFState::MAX_HIST) s.history.pop_front();

            cv::Mat future = corrected.clone();
            std::vector<cv::Point3d> path;
            path.reserve(steps_);
            for (int k = 0; k < steps_; ++k) {
                future = s.kf.transitionMatrix * future;
                path.emplace_back(future.at<double>(0),
                                  future.at<double>(1),
                                  future.at<double>(2));
            }

            PredictionResult pr;
            pr.markerId    = mp.id;
            pr.smoothedPos = smoothed;
            pr.velocity    = vel;
            pr.futurePath  = std::move(path);
            pr.horizonMs   = steps_ * dt * 1000.0;
            pr.reliable    = (s.history.size() >= 5);
            results.push_back(std::move(pr));
        }
        return results;
    }

    void drawPredictions(cv::Mat& frame,
                         const std::vector<PredictionResult>& results,
                         const cv::Mat& K, const cv::Mat& D) const {
        for (const auto& pr : results) {
            if (!pr.reliable || pr.futurePath.empty()) continue;

            std::vector<cv::Point3f> pts3;
            pts3.emplace_back((float)pr.smoothedPos.x,
                              (float)pr.smoothedPos.y,
                              (float)pr.smoothedPos.z);
            for (const auto& p : pr.futurePath)
                pts3.emplace_back((float)p.x,(float)p.y,(float)p.z);

            std::vector<cv::Point2f> pts2;
            cv::Mat r0 = cv::Mat::zeros(3,1,CV_64F);
            cv::Mat t0 = cv::Mat::zeros(3,1,CV_64F);
            cv::projectPoints(pts3, r0, t0, K, D, pts2);

            for (size_t i = 1; i < pts2.size(); ++i) {
                double a = (double)i / pts2.size();
                cv::Scalar col((int)(255*(1-a)), 255, (int)(255*a));
                cv::line(frame,
                         {(int)pts2[i-1].x,(int)pts2[i-1].y},
                         {(int)pts2[i  ].x,(int)pts2[i  ].y},
                         col, 2, cv::LINE_AA);
            }
            if (!pts2.empty())
                cv::circle(frame,
                           {(int)pts2.back().x,(int)pts2.back().y},
                           5, {0,140,255}, cv::FILLED);

            double spd = cv::norm(
                cv::Vec3d(pr.velocity.x,pr.velocity.y,pr.velocity.z));
            char buf[48];
            std::snprintf(buf, sizeof(buf), "v=%.2fm/s", spd);
            cv::putText(frame, buf,
                        {(int)pts2[0].x+8,(int)pts2[0].y-22},
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, {0,255,255}, 1);
        }
    }

    void resetMarker(int id) { kfs_.erase(id); }
    void resetAll()           { kfs_.clear();  }

private:
    int    steps_;
    double pNoise_, mNoise_;
    std::map<int,KFState> kfs_;

    void initKF(KFState& s, const cv::Point3d& p0) const {
        s.kf.init(6, 3, 0, CV_64F);
        s.kf.transitionMatrix = (cv::Mat_<double>(6,6) <<
            1,0,0, 1,0,0,
            0,1,0, 0,1,0,
            0,0,1, 0,0,1,
            0,0,0, 1,0,0,
            0,0,0, 0,1,0,
            0,0,0, 0,0,1);
        s.kf.measurementMatrix = cv::Mat::zeros(3,6,CV_64F);
        s.kf.measurementMatrix.at<double>(0,0) = 1;
        s.kf.measurementMatrix.at<double>(1,1) = 1;
        s.kf.measurementMatrix.at<double>(2,2) = 1;
        cv::setIdentity(s.kf.processNoiseCov,     cv::Scalar(pNoise_));
        for (int i=3;i<6;++i)
            s.kf.processNoiseCov.at<double>(i,i) = pNoise_*10.0;
        cv::setIdentity(s.kf.measurementNoiseCov, cv::Scalar(mNoise_));
        cv::setIdentity(s.kf.errorCovPost,         cv::Scalar(1.0));
        s.kf.statePost.at<double>(0) = p0.x;
        s.kf.statePost.at<double>(1) = p0.y;
        s.kf.statePost.at<double>(2) = p0.z;
        s.initialised = true;
    }

    static double clampDt(double lastMs, double nowMs) {
        return std::max(0.001, std::min((nowMs-lastMs)/1000.0, 0.5));
    }
};
