// ============================================================================
//  SECTION 6 — LOGGER
// ============================================================================

class Logger {
public:
    explicit Logger(const std::string& path) : file_(path) {
        if (file_.is_open())
            file_ << "frame,det_ms,pose_ms,pred_ms,total_ms,markers\n";
    }
    ~Logger() { if (file_.is_open()) file_.close(); }

    void log(int frame, double det, double pose,
             double pred, double total, int markers) {
        dV_.push_back(det); pV_.push_back(pose);
        prV_.push_back(pred); tV_.push_back(total);
        if (file_.is_open())
            file_<<frame<<","<<std::fixed<<std::setprecision(3)
                 <<det<<","<<pose<<","<<pred<<","<<total<<","<<markers<<"\n";
    }

    void printSummary() const {
        if (tV_.empty()) return;
        auto avg=[](const std::vector<double>& v){
            return std::accumulate(v.begin(),v.end(),0.0)/v.size();
        };
        double mT=avg(tV_);
        std::cout<<"\n========== Performance Summary ==========\n"
                 <<std::fixed<<std::setprecision(2)
                 <<"  Frames      : "<<tV_.size()            <<"\n"
                 <<"  Avg detect  : "<<avg(dV_)              <<" ms\n"
                 <<"  Avg pose    : "<<avg(pV_)              <<" ms\n"
                 <<"  Avg predict : "<<avg(prV_)             <<" ms\n"
                 <<"  Avg total   : "<<mT                    <<" ms\n"
                 <<"  Avg FPS     : "<<(mT>0?1000.0/mT:0)   <<"\n"
                 <<"=========================================\n";
    }

private:
    std::ofstream       file_;
    std::vector<double> dV_,pV_,prV_,tV_;
};


// ============================================================================
//  SECTION 7 — UTILITIES
// ============================================================================

void generateMarkers(int count=10, int sizePx=200) {
    MAKE_DIR("markers");
    cv::aruco::Dictionary dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
    for (int id=0; id<count; ++id) {
        cv::Mat img;
        // NEW API: generateImageMarker replaces drawMarker
        cv::aruco::generateImageMarker(dict, id, sizePx, img, 1);
        int brd=sizePx/8;
        cv::Mat out(sizePx+2*brd+24, sizePx+2*brd, CV_8UC1, cv::Scalar(255));
        img.copyTo(out(cv::Rect(brd,brd,sizePx,sizePx)));
        cv::putText(out,"ID "+std::to_string(id),
                    {brd,sizePx+2*brd+16},
                    cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0),1);
        std::string fn="markers/marker_"+std::to_string(id)+".png";
        cv::imwrite(fn,out);
        std::cout<<"Saved: "<<fn<<"\n";
    }
    std::cout<<"\nDone. Print markers/marker_0.png and hold "
               "it in front of the webcam.\n";
}

cv::Mat makeTopDown(int sz=400) {
    cv::Mat c(sz,sz,CV_8UC3,cv::Scalar(18,18,18));
    for (int i=0;i<sz;i+=50) {
        cv::line(c,{i,0},{i,sz},{35,35,35},1);
        cv::line(c,{0,i},{sz,i},{35,35,35},1);
    }
    cv::putText(c,"Top-Down View",{6,14},
                cv::FONT_HERSHEY_SIMPLEX,0.38,{80,80,80},1);
    return c;
}

