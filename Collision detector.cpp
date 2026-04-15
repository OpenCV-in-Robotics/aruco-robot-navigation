// ============================================================================
//  SECTION 4 — COLLISION DETECTOR  (NEW METHOD — Tushar)
// ============================================================================

enum class AlertLevel { SAFE=0, WARNING=1, DANGER=2 };

struct CollisionZone {
    std::string  name;
    cv::Point3d  centre;
    double       dangerRadius;
    double       warnRadius;
    cv::Scalar   colour;
};

struct CollisionAlert {
    int         markerId;
    std::string zoneName;
    AlertLevel  level;
    double      distToZone;
    double      timeToImpactMs;
    cv::Point3d robotPos;
};

using AlertCB = std::function<void(const CollisionAlert&)>;

class CollisionDetector {
public:
    void addZone(const CollisionZone& z) { zones_.push_back(z); }
    void setCallback(AlertCB cb)         { cb_ = std::move(cb); }

    std::vector<CollisionAlert> check(
        const std::vector<PredictionResult>& preds,
        double stepDtMs = 100.0)
    {
        std::vector<CollisionAlert> alerts;
        for (const auto& pr : preds) {
            for (const auto& z : zones_) {
                double     curDist = distToBoundary(pr.smoothedPos, z);
                AlertLevel level   = AlertLevel::SAFE;
                double     tti     = -1.0;

                if (curDist <= 0.0) {
                    level = AlertLevel::DANGER;
                } else {
                    double dw = cv::norm(pr.smoothedPos-z.centre)-z.warnRadius;
                    if (dw <= 0.0) level = AlertLevel::WARNING;
                    for (size_t step=0; step<pr.futurePath.size(); ++step)
                        if (distToBoundary(pr.futurePath[step],z) <= 0.0) {
                            tti   = (double)(step+1)*stepDtMs;
                            level = AlertLevel::WARNING;
                            break;
                        }
                }
                if (level==AlertLevel::SAFE) continue;

                CollisionAlert a;
                a.markerId       = pr.markerId;
                a.zoneName       = z.name;
                a.level          = level;
                a.distToZone     = curDist;
                a.timeToImpactMs = tti;
                a.robotPos       = pr.smoothedPos;
                alerts.push_back(a);
                if (cb_) cb_(a);
            }
        }
        return alerts;
    }

    void drawHUD(cv::Mat& frame,
                 const std::vector<CollisionAlert>& alerts) const {
        int y = frame.rows-14;
        for (auto it=alerts.rbegin(); it!=alerts.rend(); ++it, y-=22) {
            bool danger = (it->level==AlertLevel::DANGER);
            cv::Scalar col = danger ? cv::Scalar(0,0,255):cv::Scalar(0,140,255);
            std::ostringstream ss;
            if (danger)
                ss<<"[DANGER]  ID"<<it->markerId<<" inside "<<it->zoneName;
            else {
                ss<<"[WARN]    ID"<<it->markerId<<" near "<<it->zoneName;
                if (it->timeToImpactMs>0)
                    ss<<"  ~"<<(int)it->timeToImpactMs<<"ms";
            }
            std::string msg=ss.str();
            cv::rectangle(frame,{0,y-16},{(int)msg.size()*9+10,y+4},
                          {0,0,0},cv::FILLED);
            cv::putText(frame,msg,{6,y},
                        cv::FONT_HERSHEY_SIMPLEX,0.52,col,1);
        }
    }

    void drawTopDown(cv::Mat& canvas,
                     const std::vector<PredictionResult>& preds,
                     const std::vector<CollisionAlert>& alerts,
                     double ppm, cv::Point origin) const {
        std::map<std::string,AlertLevel> lut;
        for (const auto& a:alerts) lut[a.zoneName]=a.level;

        for (const auto& z:zones_) {
            cv::Point ctr(origin.x+(int)(z.centre.x*ppm),
                          origin.y-(int)(z.centre.z*ppm));
            cv::Scalar col=z.colour;
            auto it=lut.find(z.name);
            if (it!=lut.end())
                col=(it->second==AlertLevel::DANGER)
                    ?cv::Scalar(0,0,255):cv::Scalar(0,140,255);
            cv::circle(canvas,ctr,(int)(z.warnRadius*ppm),col,1,cv::LINE_AA);
            cv::Mat ov=canvas.clone();
            cv::circle(ov,ctr,(int)(z.dangerRadius*ppm),col,cv::FILLED);
            cv::addWeighted(ov,0.25,canvas,0.75,0,canvas);
            cv::circle(canvas,ctr,(int)(z.dangerRadius*ppm),col,2,cv::LINE_AA);
            cv::putText(canvas,z.name,ctr+cv::Point(4,-4),
                        cv::FONT_HERSHEY_SIMPLEX,0.38,col,1);
        }
        for (const auto& pr:preds) {
            cv::Point cur(origin.x+(int)(pr.smoothedPos.x*ppm),
                          origin.y-(int)(pr.smoothedPos.z*ppm));
            cv::circle(canvas,cur,7,{0,220,0},cv::FILLED);
            cv::putText(canvas,"R"+std::to_string(pr.markerId),
                        cur+cv::Point(9,4),
                        cv::FONT_HERSHEY_SIMPLEX,0.38,{0,220,0},1);
            cv::Point prev=cur;
            for (const auto& p:pr.futurePath) {
                cv::Point next(origin.x+(int)(p.x*ppm),
                               origin.y-(int)(p.z*ppm));
                cv::line(canvas,prev,next,{0,180,255},1);
                prev=next;
            }
        }
    }

private:
    std::vector<CollisionZone> zones_;
    AlertCB                    cb_;

    static double distToBoundary(const cv::Point3d& p,
                                  const CollisionZone& z) {
        cv::Point3d d=p-z.centre;
        return std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z)-z.dangerRadius;
    }
};

