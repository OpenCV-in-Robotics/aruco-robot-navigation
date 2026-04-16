// ============================================================================
//  SECTION 1 — ArUco DETECTOR
//  Member: Shresth
//  Uses new OpenCV 4.7+ ArucoDetector class.
//  detectMarkers() is now a method of the ArucoDetector object.
// ============================================================================

struct DetectionResult {
    std::vector<int>                       markerIds;
    std::vector<std::vector<cv::Point2f>>  markerCorners;
    std::vector<std::vector<cv::Point2f>>  rejectedCandidates;
    double                                 detectionTimeMs;
    int                                    frameW, frameH;
};

class ArucoDetector {
public:
    ArucoDetector() {
        // NEW API: Dictionary is a plain value type (not Ptr<>)
        dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);

        // NEW API: DetectorParameters constructed directly (no ::create())
        cv::aruco::DetectorParameters params;
        params.cornerRefinementMethod    = cv::aruco::CORNER_REFINE_SUBPIX;
        params.adaptiveThreshWinSizeMin  = 3;
        params.adaptiveThreshWinSizeMax  = 23;
        params.adaptiveThreshWinSizeStep = 10;
        params.minMarkerPerimeterRate    = 0.03f;
        params.errorCorrectionRate       = 0.10;

        // NEW API: ArucoDetector is now a proper object
        detector_ = cv::aruco::ArucoDetector(dictionary_, params);
    }

    DetectionResult detect(const cv::Mat& frame) {
        if (frame.empty()) throw std::runtime_error("Empty frame");

        DetectionResult res;
        res.frameW = frame.cols;
        res.frameH = frame.rows;

        cv::Mat grey;
        if (frame.channels() == 3)
            cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
        else
            grey = frame;

        auto t0 = HRC::now();

        // NEW API: detectMarkers is a method on the ArucoDetector object
        detector_.detectMarkers(grey,
                                 res.markerCorners,
                                 res.markerIds,
                                 res.rejectedCandidates);

        res.detectionTimeMs =
            std::chrono::duration<double, std::milli>(HRC::now()-t0).count();
        return res;
    }

    cv::Mat visualise(const cv::Mat& frame, const DetectionResult& res) const {
        cv::Mat out = frame.clone();
        if (!res.markerIds.empty())
            cv::aruco::drawDetectedMarkers(out, res.markerCorners,
                                           res.markerIds);
        if (!res.rejectedCandidates.empty())
            cv::aruco::drawDetectedMarkers(out, res.rejectedCandidates,
                                           cv::noArray(),
                                           cv::Scalar(180,0,180));
        char buf[80];
        std::snprintf(buf, sizeof(buf), "Markers: %d  |  Det: %.1f ms",
                      (int)res.markerIds.size(), res.detectionTimeMs);
        cv::putText(out, buf, {10,26},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,230,0}, 2);
        return out;
    }

private:
    cv::aruco::Dictionary    dictionary_;
    cv::aruco::ArucoDetector detector_;
};
