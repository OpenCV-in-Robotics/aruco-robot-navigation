# 🤖 ArUco Robot Navigation
### Real-time Robot Navigation using OpenCV ArUco Markers with Kalman Filter Trajectory Prediction & Collision Avoidance

<div align="center">

![C++](https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=c%2B%2B)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7%2B-green?style=flat-square&logo=opencv)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**Case Study — OpenCV ArUco Library | Domain: Robotics**

</div>

---

## 👥 Team Members

| Member | Role | Module |
|--------|------|--------|
| **Shresth** | ArUco Marker Detection | `Section 1 — ArucoDetector` |
| **Krish** | Pose Estimation & Camera Calibration | `Section 2 — PoseEstimator` |
| **Mitul** | Trajectory Prediction ⭐ *New Method* | `Section 3 — TrajectoryPredictor` |
| **Tushar** | Collision Zone Detection ⭐ *New Method* | `Section 4 — CollisionDetector` |
| **Karthikeya** | Robot Navigator & System Integration | `Section 5 — RobotNavigator + main()` |

> ⭐ = New enhancement method added on top of the base OpenCV ArUco library

---

## 📌 Project Overview

This project implements a **real-time autonomous robot navigation system** built on top of OpenCV's ArUco fiducial marker library. A fixed camera observes a robot that carries a printed ArUco marker. The system:

1. **Detects** the marker in every camera frame
2. **Estimates** the robot's exact 6-DoF pose in 3D space (position + orientation)
3. **Predicts** where the robot is heading using a Kalman filter *(new enhancement)*
4. **Alerts** if the robot is on a collision course with a forbidden zone *(new enhancement)*
5. **Issues** navigation commands — Forward, Turn, Hold, Emergency Stop

Everything runs in **real time at 30+ FPS** using a single webcam and printed paper markers.

---

## 🧠 Library: OpenCV ArUco

**OpenCV** (Open Source Computer Vision Library) is the world's most widely used computer vision library. The **ArUco module** provides:

- 100 unique printable square binary markers
- Robust detection under varying lighting, angles, and partial occlusion
- 6-DoF pose estimation from a single monocular camera

### 🔍 Key Method Analysed: `estimatePoseSingleMarkers()`

The core function studied in this project computes the **full 3D position and orientation** of each detected marker:

```
Input : 2D corner pixels + physical marker size + camera intrinsics
Output: rvec (rotation) + tvec (translation in metres) per marker
```

Internally it uses the **IPPE algorithm** (Infinitesimal Plane-based Pose Estimation) — a closed-form mathematical solution that converts 2D pixel coordinates into a 3D rigid-body transform without iterative optimisation. In our implementation we call `cv::solvePnP()` with `SOLVEPNP_IPPE_SQUARE` directly (equivalent, compatible with all OpenCV 4.7+ versions).

---

## ⭐ New Enhancements

### Enhancement 1 — Kalman Filter Trajectory Prediction (Mitul)

Raw pose measurements are noisy (±1–3 cm jitter). This module adds:

| Without Prediction | With Prediction |
|--------------------|-----------------|
| Noisy ±3 cm position | Smooth < 0.5 cm position |
| No velocity info | Real-time velocity in m/s |
| Zero look-ahead | 300–1000 ms look-ahead |
| Reacts only when IN danger | Warns BEFORE entering danger |

**How it works:** A 6-state Kalman filter `[x, y, z, vx, vy, vz]` is maintained per tracked marker. Each frame: predict the next state using a constant-velocity model, then correct with the new measurement. The corrected state is projected forward N steps to generate the predicted path — shown as a **cyan-to-orange trail** on screen.

### Enhancement 2 — Collision Zone Detection (Tushar)

Defines spherical exclusion zones in the robot's workspace. Checks both the **current position** and all **10 predicted future positions** against every zone:

- 🟠 **WARNING** — robot will enter a zone within the prediction horizon
- 🔴 **DANGER** — robot is currently inside a zone right now

Danger alerts trigger an immediate **Emergency Stop** in the navigator.

---

## 🏗️ System Architecture

```
Camera Frame
     │
     ▼
┌─────────────────────┐
│   ArucoDetector     │  ← Shresth
│   detectMarkers()   │  Finds marker IDs + corner pixels
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   PoseEstimator     │  ← Krish
│   solvePnP (IPPE)   │  Computes 3D position + yaw/pitch/roll
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ TrajectoryPredictor │  ← Mitul  ⭐ NEW
│   Kalman Filter     │  Smooths pose + predicts 10 future positions
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ CollisionDetector   │  ← Tushar  ⭐ NEW
│  Zone Intersection  │  WARNING / DANGER alerts with time-to-impact
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RobotNavigator     │  ← Karthikeya
│  5-State FSM        │  STOP / FWD / TURN / HOLD / EMERGENCY
└─────────────────────┘
```

---

## 📁 File Structure

```
aruco-robotics/
├── aruco_robot_navigation_final.cpp   ← Single file, entire project
├── README.md                          ← This file
└── evaluation/
    └── metrics.csv                    ← Auto-generated on first run
```

> The entire project is a **single self-contained `.cpp` file**. No separate headers, no CMake, no external project files needed beyond OpenCV.

---

## ⚙️ Installation & Setup

### Prerequisites

**Windows (MSYS2/MinGW) — recommended for this project:**
```bash
# Install OpenCV via MSYS2 pacman
pacman -S mingw-w64-x86_64-opencv
pacman -S mingw-w64-x86_64-gcc
```

**Ubuntu / Debian / WSL2:**
```bash
sudo apt update
sudo apt install -y build-essential libopencv-dev libopencv-contrib-dev
```

**macOS:**
```bash
brew install opencv
```

### Verify OpenCV is installed
```bash
pkg-config --modversion opencv4
# Should print something like: 4.8.0
```

---

## 🔨 Compile

**Windows (MSYS2 terminal) / Linux / macOS — same command:**
```bash
g++ -std=c++17 aruco_robot_navigation_final.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -o aruco_navigation
```

**If the above fails, try explicit flags:**
```bash
g++ -std=c++17 aruco_robot_navigation_final.cpp \
    -I/usr/include/opencv4 \
    -lopencv_core -lopencv_highgui -lopencv_imgproc \
    -lopencv_videoio -lopencv_calib3d -lopencv_objdetect \
    -o aruco_navigation
```

You should see no errors and a new file called `aruco_navigation` (or `aruco_navigation.exe` on Windows).

---

## 🚀 Running the Project

### Step 1 — Generate markers to print
```bash
./aruco_navigation --generate-markers
```
This creates a `markers/` folder with `marker_0.png` through `marker_9.png`.
**Print `marker_0.png` on paper.** This is your "robot".

### Step 2 — Run the demo
```bash
# Basic — uses webcam, default settings
./aruco_navigation

# With a recorded video file
./aruco_navigation --video myvideo.mp4

# Navigate toward marker ID 2 instead of ID 0
./aruco_navigation --target 2

# If your printed marker is 8 cm wide (measure the black border)
./aruco_navigation --marker-len 0.08

# Disable the top-down bird's-eye window
./aruco_navigation --no-topdown
```

### Step 3 — Hold the printed marker in front of the webcam

Two windows appear immediately:
- **Main window** — camera feed with all overlays
- **Top-Down View** — bird's-eye map of the workspace

---

## 🖥️ What You See on Screen

| Visual Element | Colour | Who Built It | What It Means |
|---|---|---|---|
| Box around marker + ID | Green | Shresth | Marker detected |
| 3D axes on marker | Red/Green/Blue | Krish | Robot orientation |
| Distance label | Yellow | Krish | Metres from camera |
| Predicted path trail | Cyan → Orange | Mitul | Next 10 predicted positions |
| Speed label | Cyan | Mitul | Robot velocity in m/s |
| Alert banner | Orange / Red | Tushar | WARNING / DANGER zone |
| State label (top right) | Purple | Balla | Navigator FSM state |
| CMD label (top left) | Yellow | Balla | Current navigation command |
| FPS counter | Grey | All | Pipeline performance |

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit the program |
| `r` | Reset navigator state + Kalman filters |
| `s` | Save current frame as PNG |

---

## 📊 Evaluation / Performance Metrics

After running, metrics are saved automatically to `evaluation/metrics.csv`.

The CSV contains per-frame timing for each pipeline stage:

| Column | Description |
|--------|-------------|
| `frame` | Frame number |
| `det_ms` | Detection time (Shresth's module) |
| `pose_ms` | Pose estimation time (Krish's module) |
| `pred_ms` | Kalman prediction time (Mitul's module) |
| `total_ms` | Full pipeline time per frame |
| `markers` | Number of markers detected |

**Expected performance on a modern laptop:**

| Metric | Expected Value |
|--------|---------------|
| Detection time | 3 – 8 ms |
| Pose estimation | 0.5 – 2 ms |
| Kalman prediction | < 1 ms |
| Total FPS | 30 – 60 FPS |
| Position accuracy (with calibration) | ±1 – 2 cm |

---

## 🗂️ Navigator State Machine

```
         ┌─────────────┐
  start  │  SEARCHING  │ ──── marker found ────►
         └─────────────┘
                                  ┌──────────────┐
                                  │  APPROACHING │ ──── close enough ────►
                                  └──────────────┘
                                                        ┌───────────┐
                                                        │  ALIGNING │ ──── yaw OK ────►
                                                        └───────────┘
                                                                           ┌─────────┐
                                                                           │ HOLDING │
                                                                           └─────────┘

              ANY STATE ──── DANGER alert ────► EMERGENCY STOP ──── danger clear ────► SEARCHING
```

---

## 🐙 GitHub Workflow

All features are developed on individual branches and merged via reviewed Pull Requests.

### Branch structure
```
main
 ├── feature/shresth-aruco-detector
 ├── feature/krish-pose-estimator
 ├── feature/mitul-trajectory-predictor
 ├── feature/tushar-collision-detector
 └── feature/karthikeya-navigator
```

### Commit message format
```
feat: implement Kalman filter trajectory prediction with 10-step look-ahead
fix: clamp Kalman dt to prevent instability after frame drops
docs: update README with calibration instructions
eval: add per-stage latency breakdown to evaluation metrics
```

---

## 🔧 Troubleshooting

**`pkg-config: command not found`**
```bash
# Windows MSYS2:
pacman -S pkg-config
# Ubuntu:
sudo apt install pkg-config
```

**`Cannot open video source` error**
- Make sure a webcam is connected
- On Windows, try changing `cap.open(0)` to `cap.open(1)` if you have multiple cameras

**Markers not being detected**
- Make sure the printed marker has a clear white border around it
- Try better lighting — avoid direct glare on the paper
- Hold the marker flat and face-on to the camera first
- Use `--marker-len` to match your actual printed size

**Very inaccurate distance readings**
- Run with proper camera calibration (use a chessboard calibration tool)
- The default parameters assume a standard 640×480 webcam

**Slow FPS on Windows**
- Make sure you compiled with `-O2` optimisation flag
- Use Release build settings

---

## 📚 References

1. Garrido-Jurado, S. et al. (2014). *Automatic generation and detection of highly reliable fiducial markers under occlusion.* Pattern Recognition, 47(6), 2280–2292.
2. Collins, T. & Devereux, A. (2014). *Infinitesimal Plane-Based Pose Estimation.* International Journal of Computer Vision.
3. Welch, G. & Bishop, G. (2006). *An Introduction to the Kalman Filter.* UNC Chapel Hill Technical Report TR 95-041.
4. Bradski, G. & Kaehler, A. (2008). *Learning OpenCV.* O'Reilly Media.
5. OpenCV ArUco Tutorial: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

---

## 📄 License

This project was developed as an academic case study.

---

<div align="center">
Made with ❤️ by <strong>Shresth · Krish · Mitul · Tushar · Karthikeya</strong>
</div>
