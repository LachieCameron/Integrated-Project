# Track A: TurtleBot3 EKF + LQR Navigation and ArUco Docking

**41014 Sensors and Control in Mechatronics Systems, Assessment 4: Integrated Project**

Team: Lachlan Cameron, Jonathan Wijaya, William Hsu

> Demo video (5 min): _TODO link_
> Platform: simulation (MuJoCo). Hardware: _TODO yes/no_

## 1. Overview

TODO: 3 to 4 sentences. What the robot does, the task, and the key result (e.g. docking success rate and RMSE from Section 6).

Autonomous Docking TurtleBot3 Project where demonstrated using MuJoCo simulation. The robot integrates wheel encoder odometry, camera-based ArUco marker detection, LiDAR obstacle detection, Extended Kalman Filter (EKF) state estimation, and Linear Quadratic Regulator (LQR) trajectory tracking. 

The system initially detects ArUco marker ID 0 and uses marker ID 99 for fine docking alignment. The EKF estimates the robot's position and orientation, while LQR-based navigation controls its approach toward the pre-docking location.

## 2. Install and run
See [SETUP.md](SETUP.md) for the environment setup on Mac and Windows.

TODO: add the exact commands to run the project once the code exists. The marking guide requires that the code runs from a fresh environment as described here.

## 3. System architecture

TODO: block diagram (simulator, sensors, EKF, LQR, back to simulator).

**Constraint:** the controller uses the estimated state only. Simulator ground truth is used for evaluation only, and this should be visible in the code.

## 4. Modelling (Part A)

- State, input and equations: TODO
- Assumptions: TODO
- Controllability and observability analysis: TODO. Include which states are observable from which sensors, and what happens when the marker leaves the camera view.

### 4.1 State, input and equations

**State Vector:**

$$
\mathbf{x} =
\begin{bmatrix}
x \\
y \\
\theta
\end{bmatrix}
$$

**Control Vector:**

$$
\mathbf{u} =
\begin{bmatrix}
v \\
\omega
\end{bmatrix}
$$

Where:
- $v$: Linear velocity (m/s).
- $\omega$: Angular velocity (rad/s).

**Differential-drive kinematic equations:**

$$
\dot{x} = v\cos\theta
$$

$$
\dot{y} = v\sin\theta
$$

$$
\dot{\theta} = \omega
$$

$$
v = \frac{r}{2}(\omega_R+\omega_L)
$$

$$
\omega = \frac{r}{L}(\omega_R-\omega_L)
$$

Where:
- $r$: Wheel radius (m).
- $L$: Distance between the left and right wheels (m).
- $\omega_R$: Right wheel angular velocity (rad/s).
- $\omega_L$: Left wheel angular velocity (rad/s).

### 4.2 Assumptions

### 4.3 Controllability and observability analysis

## 5. Sensing and calibration (Part B)

- Frame diagram: TODO
- Camera calibration and ArUco detection: TODO
- LiDAR processing: TODO
- Noise characterisation: TODO

## 5. Sensing and Calibration (Part B)

### 5.1 Sensor Configuration

three main sensors implemented for navigation, localisation and obstacle avoidance.

| Sensor | Measurement | Purpose |
|---|---|---|
| Wheel Encoders | Left and right wheel displacement | Predict robot motion in the EKF |
| RGB Camera | ArUco marker ID, corners and relative pose | Detect navigation targets and correct the EKF estimate |
| LiDAR | Distance to nearby obstacles | Detect obstacles and support collision avoidance |

### 5.2 Coordinate Frames

The system uses four coordinate frames:

1. **World Frame:** Global reference for the robot's position and navigation targets.
2. **Robot Base Frame:** Local coordinate system for TurtleBot3.
3. **Camera Frame:** Coordinate system used for camera measurements and marker detection.
4. **ArUco Marker Frame:** Local coordinate system attached to each marker.

The coordinate transformation from the camera to the robot and world frames is required to convert marker observations into global robot pose measurements.

The relationship between coordinate frames can be represented as:

$$
T_{WM} = T_{WB}T_{BC}T_{CM}
$$

Where:

- $T_{WM}$: Marker pose in the world frame.
- $T_{WB}$: Robot base pose in the world frame.
- $T_{BC}$: Camera pose relative to the robot base.
- $T_{CM}$: Marker pose relative to the camera.

Rearranging gives:

$$
T_{WB}=T_{WM}T_{CM}^{-1}T_{BC}^{-1}
$$

This allows the camera measurement to be transformed into the robot's global pose for EKF correction.

### 5.3 Camera Calibration and ArUco Detection

The RGB camera is used to detect ArUco markers and estimate the robot's pose relative to known landmarks.

The camera detection process consists of:

1. Detects ArUco markers using OpenCV.
2. Extracting marker IDs and corner coordinates.
3. Estimating marker pose using camera calibration parameters and known marker dimensions.
4. Providing the resulting robot pose measurement to the EKF.

The system uses:

| Parameter | Value |
|---|---|
| ArUco dictionary | `DICT_5X5_100` |
| Coarse navigation marker | ID 0 |
| Fine docking marker | ID 99 |
| Camera update interval | 1/30 s |
| Coarse marker size | 0.20 m |
| Fine marker size | 0.05 m |

Camera calibration requires the intrinsic matrix:

$$
K=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
$$

Where:

- $f_x,f_y$: Camera focal lengths in pixels.
- $c_x,c_y$: Principal point coordinates.

The camera model may also include distortion coefficients to correct lens distortion.

The camera measurement is accepted by the EKF only when it passes the innovation-gating test.

### 5.4 LiDAR Processing

LiDAR are implemented for obstacle detection and avoidance across robot's forward facing region. It was integrated across -90° to +90° at 15° intervals.

The measurement then grouped into three different sectors, where:

| Sector | Angular Range | Purpose |
|---|---|---|
| Front | -30° to +30° | Detect obstacles directly ahead |
| Left | +45° to +90° | Evaluate clearance on the left |
| Right | -90° to -45° | Evaluate clearance on the right |

The minimum valid distance in each sector is used for obstacle detection.

The main obstacle avoidance parameters are:

| Parameter | Value |
|---|---|
| Stop distance | 0.20 m |
| Clearance distance | 0.40 m |
| Avoidance distance | 0.50 m |
| Avoidance turn angle | 35° |
| Avoidance angular velocity | 0.45 rad/s |

When an obstacle is detected, the robot can stop, turn toward the clearer direction and move forward to bypass it.

During pre-docking, LiDAR measurements are also supplied to the navigation controller.

### 5.5 Sensor Noise Characterisation

Sensor measurements contain uncertainty caused by simulation noise, numerical errors and limitations of the sensing models.

The EKF represents uncertainty using process noise covariance $Q$ and measurement noise covariance $R$.

For camera pose measurements, the integrated script uses:

$$
R_{\text{camera}}=
\begin{bmatrix}
0.08^2 & 0 & 0\\
0 & 0.08^2 & 0\\
0 & 0 & (5\pi/180)^2
\end{bmatrix}
$$

This corresponds to assumed standard deviations of:

- X-position: 0.08 m.
- Y-position: 0.08 m.
- Heading: 5°.

The camera correction uses a chi-square innovation gate with a threshold of 11.345, corresponding to a 99% confidence threshold for three measurement dimensions.

## 6. Estimation (Part C)

- EKF design and parameter table (Q, R, P0) with justification: TODO
- Tuning, consistency checks and reported performance: TODO


## 6. Estimation (Part C)

### 6.1 EKF Design and Parameters

An Extended Kalman Filter (EKF) estimates the robot's position $(x,y)$ and heading $(\theta)$ by combining wheel encoder odometry with ArUco camera measurements.

- **Prediction:** Wheel encoders estimate the robot's movement.
- **Correction:** Camera measurements correct accumulated odometry drift when an ArUco marker is visible.

**EKF Parameter Table:**

| Parameter | Value | Justification |
|---|---|---|
| Q (Process Noise) | Calculated from wheel-motion uncertainty | Accounts for encoder and motion-model errors |
| R (Camera Noise) | diag(0.08², 0.08², 5°²) | Represents camera position and heading uncertainty |
| P0 (Initial Covariance) | diag(0.15², 0.15², 15°²) | Represents uncertainty in the initial robot pose |

### 6.2 Tuning, Consistency Checks and Performance

The EKF uses a **chi-square innovation gate of 11.345** to reject inconsistent camera measurements.

Performance can be evaluated by comparing the EKF-estimated trajectory against MuJoCo ground truth using position and heading RMSE.

When the ArUco marker is not visible, the EKF continues predicting from wheel encoders, but estimation uncertainty can increase.

**Results:** Quantitative RMSE and consistency results still need to be recorded from simulation trials.

## 7. Control (Part D)

- Controller design (state feedback or LQR), gain or weight selection with justification: TODO
- Tracking results: TODO

### 7.1 Controller Architecture

The project uses a hybrid control system consisting of ArUco marker following, LQR-based navigation and visual docking.

| Control Stage | Controller | Function |
|---|---|---|
| Initial navigation | `TagFollower` | Track coarse ArUco marker ID 0 |
| Obstacle avoidance | LiDAR-based navigation | Avoid obstacles during movement |
| Pre-docking | LQR trajectory tracking | Navigate to the pre-docking pose |
| Fine docking | Visual proportional control | Align with ArUco marker ID 99 |
| Docking completion | Stop command | Stop wheel motion when docking is completed |

The LQR controller operates during the `PRE_DOCK` stage and uses the EKF-estimated robot pose as feedback.

### 7.2 LQR Controller Design

The Linear Quadratic Regulator (LQR) minimises tracking error while penalising excessive control corrections.

The linearised tracking-error system is:

$$
\mathbf{e}_{k+1}
=
A_d\mathbf{e}_k+B_d\Delta\mathbf{u}_k
$$

The LQR objective function is:

$$
J=
\sum_{k=0}^{\infty}
\left(
\mathbf{e}_k^TQ\mathbf{e}_k+
\Delta\mathbf{u}_k^TR\Delta\mathbf{u}_k
\right)
$$

Where:

- $\mathbf{e}$: Robot tracking error.
- $Q$: State tracking-error weighting matrix.
- $R$: Control-effort weighting matrix.
- $\Delta\mathbf{u}$: Control correction.

The optimal feedback gain is obtained by solving the discrete algebraic Riccati equation.

The controller applies a state-feedback correction using the estimated pose and reference trajectory.

The generated linear and angular velocities are converted into left and right wheel angular velocities.

### 7.3 LQR Parameters and Weight Selection

| Weight | Function |
|---|---|
| $Q_{11}$ | Penalises longitudinal tracking error |
| $Q_{22}$ | Penalises lateral tracking error |
| $Q_{33}$ | Penalises heading error |
| $R_{11}$ | Penalises linear velocity correction |
| $R_{22}$ | Penalises angular velocity correction |

Increasing a state-error weight generally encourages stronger correction of that error, while increasing a control-effort weight discourages aggressive control corrections.

The pre-docking controller limits the commanded velocities to:

| Parameter | Maximum |
|---|---|
| Linear velocity | 0.10 m/s |
| Angular velocity | 0.45 rad/s |

These limits help prevent excessively aggressive movements during docking.

### 7.4 LQR Tuning Method

To justify the selected LQR parameters, different Q and R configurations should be tested under identical initial conditions.

The main performance metrics are:

1. Position tracking RMSE.
2. Heading tracking error.
3. Settling time.
4. Control effort.
5. Pre-docking success rate.

Control effort can be evaluated using:

$$
J_u=
\sum_{k=1}^{N}
\left(
v_k^2+\omega_k^2
\right)\Delta t
$$

For a dimensionally consistent comparison, the linear and angular terms should be normalised or weighted before being combined.

The selected Q and R values should provide an appropriate balance between tracking accuracy, settling time and control activity.

### 7.5 LQR Tracking Results

**[RESULT REQUIRED]**

| Configuration | Q Diagonal | R Diagonal | Position RMSE (m) | Settling Time (s) |
|---|---|---|---|---|
| Baseline | (1, 1, 0.5) | (0.1, 0.1) | TBD | TBD |
| Higher heading penalty | TBD | TBD | TBD | TBD |
| Higher control penalty | TBD | TBD | TBD | TBD |
| Selected configuration | TBD | TBD | TBD | TBD |

The following graphs should be included:

- Position tracking error versus time.
- Heading tracking error versus time.
- Linear and angular velocity commands versus time.
- Actual robot trajectory compared with the reference trajectory.

The results should demonstrate whether the LQR controller achieves accurate and stable pre-docking navigation.

### 7.6 Final Visual Docking

After the robot reaches the pre-docking pose, control switches from LQR to visual docking.

The pre-docking target is:

$$
\mathbf{x}_{dock}=
\begin{bmatrix}
2.4\\
2.15\\
\pi/2
\end{bmatrix}
$$

The transition conditions are:

- Position error no greater than 0.10 m.
- Heading error no greater than 8°.

During visual docking, the robot will follow ArUco marker ID 99 to estimate proximity using the detected marker width. 

The visual controller adjusts linear and angular velocities to align the robot with the charging station.

When the configured docking-completion conditions are met, the robot stops and enters the simulated `CHARGED` state.

For final-docking state, the LQR not implemented instead the robot uses visual feedback with LiDAR logic.

## 8. Integration and robustness (Part E)

- Obstacle avoidance approach: TODO
- Results over repeated trials with varied conditions, and failure diagnosis: TODO

### 8.1 Obstacle Avoidance

LiDAR are provided with configuration between -90° and 90° with 15° gap in between.

**Obstacle avoidance parameters:**

| Parameter | Value |
|---|---|
| Stop distance | 0.20 m |
| Clearance threshold | 0.40 m |
| Avoidance distance | 0.50 m |
| Avoidance turn angle | 35° |
| Avoidance forward velocity | 0.15 m/s |
| Avoidance angular velocity | 0.45 rad/s |

**Avoidance sequence:**

1. **Obstacle Detection:** When an obstacle is detected within the clearance threshold during forward movement, the robot enters `AVOID_TURN`.
2. **Direction Selection:** The robot compares left and right LiDAR clearance and selects the direction with greater available space.
3. **Turning:** The robot rotates approximately 35° to avoid the obstacle.
4. **Forward Movement:** After turning, the robot enters `AVOID_FORWARD` and travels approximately 0.50 m.
5. **Obstacle Rechecking:** If another obstacle is detected within the stop distance, the robot returns to `AVOID_TURN`.
6. **Navigation Resumption:** Once the avoidance movement is completed, the robot returns to normal navigation.

During the `PRE_DOCK` stage, LiDAR readings are supplied directly to the `Navigator` subsystem, which handles navigation and obstacle-related decisions.

**Integration Testing**

Integration testing evaluates whether the sensing, estimation, navigation, and docking subsystems operate together correctly.

The following tests should be performed:

| Test | Objective | Expected Behaviour |
|---|---|---|
| Sensor integration | Verify encoder, camera and LiDAR measurements | Sensors provide valid data |
| EKF integration | Verify odometry prediction and camera correction | Robot pose estimate updates correctly |
| LQR integration | Verify EKF-based pre-docking navigation | Robot approaches the reference pose |
| Obstacle avoidance | Place obstacles along the navigation path | Robot detects and avoids obstacles |
| Marker switching | Test transitions from ID 0 to ID 99 | Robot enters the appropriate docking stage |
| Docking completion | Approach the charging station | Robot stops when docking conditions are satisfied |

These tests assess the interaction between subsystems rather than evaluating each component independently.

### 8.2 Robustness Testing Under Varied Conditions

To evaluate system reliability, the robot were tested in different environment condition where obstacle's position were randomly changed.

The simulation supports randomised obstacle placement using:

```bash
python3 tests/ekf_docking_test.py --random-scene --scene-seed 92 --obstacle-count 3
```

Different scene seeds and obstacle counts can be used to evaluate the system under varied conditions.

**Recommended experiments:**

| Experiment | Test Condition | Evaluation Metric |
|---|---|---|
| Baseline | Fixed starting pose and obstacle layout | Docking success and completion time |
| Random obstacles | Different obstacle positions | Success rate and collision count |
| Initial pose variation | Different robot starting positions | Navigation success and tracking error |
| Sensor noise | Increased encoder or camera noise | EKF estimation accuracy |
| LQR tuning | Different Q and R values | Tracking error and settling time |

Each configuration should be tested over multiple trials to evaluate repeatability.

## 9. Limitations and extensions

TODO: what does not work, what the results do and do not support, and any independent extension.

## 10. Contribution statement

| Member | Contribution |
|--------|--------------|
| Lachlan | TODO |
| Jonathan | TODO |
| William | TODO |

## 11. Generative AI declaration

TODO: state the tools used and what they were used for. Update this as your usage changes. Non-declaration is an academic integrity matter.

Integrating multiple state logic machine into a single running code, where combine navigation, obstacle avoidance, EKF, and LQR seamlessly.

## 12. Acknowledgements

TODO: list code adapted from the weekly tutorials, MuJoCo examples, the Robotics Toolbox, and the source of the TurtleBot3 MuJoCo model.

ArUco Marker Generation: [text](https://chev.me/arucogen/)

MuJoCo Documentation: 
