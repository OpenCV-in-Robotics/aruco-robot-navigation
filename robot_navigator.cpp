// ============================================================================
//  SECTION 5 — ROBOT NAVIGATOR  (Karthikeya)
//
//  FIX: Config struct moved OUTSIDE RobotNavigator class.
//  MinGW/GCC does not allow default member initialisers in a struct
//  that is used as a default argument of the same enclosing class.
//  Moving it outside resolves this completely.
// ============================================================================

enum class NavCmd {
    STOP, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, HOLD
};

std::string navCmdName(NavCmd c) {
    switch(c) {
        case NavCmd::STOP:          return "STOP";
        case NavCmd::MOVE_FORWARD:  return "FWD";
        case NavCmd::MOVE_BACKWARD: return "BWD";
        case NavCmd::TURN_LEFT:     return "TURN_L";
        case NavCmd::TURN_RIGHT:    return "TURN_R";
        case NavCmd::HOLD:          return "HOLD";
        default:                    return "?";
    }
}

struct NavOutput {
    NavCmd      cmd;
    double      distToTarget;
    double      yawError;
    std::string statusMsg;
    bool        targetVisible;
    bool        dangerActive;
};

// FIX: Config is now a standalone struct, NOT nested inside RobotNavigator
struct NavConfig {
    int    targetId       = 0;
    double stopDistM      = 0.30;
    double alignThreshDeg = 5.0;
    double posThreshM     = 0.05;
};

class RobotNavigator {
public:
    // FIX: takes NavConfig (external struct) — no default-arg issue
    explicit RobotNavigator(NavConfig cfg = NavConfig())
        : cfg_(cfg), state_(State::SEARCHING), lostFrames_(0) {}

    NavOutput tick(const std::vector<MarkerPose>& poses,
                   const std::vector<CollisionAlert>& alerts)
    {
        for (const auto& a : alerts)
            if (a.level == AlertLevel::DANGER) {
                state_ = State::EMERGENCY;
                return makeOut(NavCmd::STOP, -1, 0,
                               "!!! EMERGENCY STOP", false, true);
            }
        if (state_ == State::EMERGENCY) state_ = State::SEARCHING;

        const MarkerPose* tgt = nullptr;
        for (const auto& mp : poses)
            if (mp.id == cfg_.targetId) { tgt = &mp; break; }

        if (!tgt) {
            if (++lostFrames_ > 30) state_ = State::SEARCHING;
            return makeOut(NavCmd::TURN_RIGHT, -1, 0,
                           "SEARCHING for ID "+std::to_string(cfg_.targetId),
                           false, false);
        }
        lostFrames_ = 0;

        switch (state_) {
            case State::SEARCHING:
                state_ = State::APPROACHING;
                // fall through
            case State::APPROACHING:
                if (tgt->distance <= cfg_.stopDistM + cfg_.posThreshM)
                    state_ = State::ALIGNING;
                return doApproach(*tgt, alerts);

            case State::ALIGNING:
                if (std::fabs(tgt->yaw) <= cfg_.alignThreshDeg)
                    state_ = State::HOLDING;
                return doAlign(*tgt);

            case State::HOLDING:
                if (tgt->distance > cfg_.stopDistM + cfg_.posThreshM*3.0) {
                    state_ = State::APPROACHING;
                    return doApproach(*tgt, alerts);
                }
                return makeOut(NavCmd::HOLD, tgt->distance, tgt->yaw,
                               "HOLDING  dist="
                               +std::to_string(tgt->distance).substr(0,4)+"m",
                               true, false);
            default:
                return makeOut(NavCmd::STOP,-1,0,"UNKNOWN",false,false);
        }
    }

    void drawHUD(cv::Mat& frame, const NavOutput& out) const {
        cv::Mat ov=frame.clone();
        cv::rectangle(ov,{0,0},{frame.cols,52},{20,20,20},cv::FILLED);
        cv::addWeighted(ov,0.72,frame,0.28,0,frame);
        cv::Scalar col=out.dangerActive?cv::Scalar(0,0,255):cv::Scalar(60,230,80);
        cv::putText(frame,out.statusMsg,{10,22},
                    cv::FONT_HERSHEY_SIMPLEX,0.55,col,1);
        cv::putText(frame,"CMD: "+navCmdName(out.cmd),{10,44},
                    cv::FONT_HERSHEY_SIMPLEX,0.48,{255,220,0},1);
        std::string sn=stateName(state_);
        int base=0;
        cv::Size sz=cv::getTextSize(sn,cv::FONT_HERSHEY_SIMPLEX,0.48,1,&base);
        cv::putText(frame,sn,{frame.cols-sz.width-10,22},
                    cv::FONT_HERSHEY_SIMPLEX,0.48,{180,180,255},1);
    }

    void reset() { state_=State::SEARCHING; lostFrames_=0; }

private:
    enum class State { SEARCHING, APPROACHING, ALIGNING, HOLDING, EMERGENCY };
    NavConfig cfg_;
    State     state_;
    int       lostFrames_;

    NavOutput doApproach(const MarkerPose& t,
                         const std::vector<CollisionAlert>& alerts) {
        bool warn=false;
        for (const auto& a:alerts)
            if (a.level==AlertLevel::WARNING){warn=true;break;}
        NavCmd cmd;
        if (std::fabs(t.yaw)>cfg_.alignThreshDeg*2.0)
            cmd=(t.yaw>0)?NavCmd::TURN_RIGHT:NavCmd::TURN_LEFT;
        else
            cmd=warn?NavCmd::HOLD:NavCmd::MOVE_FORWARD;
        std::ostringstream ss;
        ss<<"APPROACHING  dist="<<std::fixed<<std::setprecision(2)
          <<t.distance<<"m  yaw="<<std::setprecision(1)<<t.yaw<<"deg";
        if (warn) ss<<"  [COLLISION WARNING]";
        return makeOut(cmd,t.distance,t.yaw,ss.str(),true,warn);
    }

    NavOutput doAlign(const MarkerPose& t) {
        NavCmd cmd=(t.yaw>0)?NavCmd::TURN_RIGHT:NavCmd::TURN_LEFT;
        std::ostringstream ss;
        ss<<"ALIGNING  yaw="<<std::fixed<<std::setprecision(1)<<t.yaw<<"deg";
        return makeOut(cmd,t.distance,t.yaw,ss.str(),true,false);
    }

    static NavOutput makeOut(NavCmd cmd, double dist, double yaw,
                              const std::string& msg,
                              bool visible, bool danger) {
        return {cmd,dist,yaw,msg,visible,danger};
    }

    static std::string stateName(State s) {
        switch(s) {
            case State::SEARCHING:   return "SEARCHING";
            case State::APPROACHING: return "APPROACHING";
            case State::ALIGNING:    return "ALIGNING";
            case State::HOLDING:     return "HOLDING";
            case State::EMERGENCY:   return "EMERGENCY STOP";
            default:                 return "UNKNOWN";
        }
    }
};
