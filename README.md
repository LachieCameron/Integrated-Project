# Track A: TurtleBot3 EKF + LQR Navigation and ArUco Docking

**41014 Sensors and Control in Mechatronics Systems, Assessment 4: Integrated Project**

Team: Lachlan Cameron, Jonathan ([@Jojo-988](https://github.com/Jojo-988)), William ([@Willl-Je-Suis](https://github.com/Willl-Je-Suis))

> Demo video (5 min): _TODO link_
> Platform: simulation (MuJoCo). Hardware: _TODO yes/no_

## 1. Overview

TODO: 3 to 4 sentences. What the robot does, the task, and the key result (e.g. docking success rate and RMSE from Section 6).

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

## 5. Sensing and calibration (Part B)

- Frame diagram: TODO
- Camera calibration and ArUco detection: TODO
- LiDAR processing: TODO
- Noise characterisation: TODO

## 6. Estimation (Part C)

- EKF design and parameter table (Q, R, P0) with justification: TODO
- Tuning, consistency checks and reported performance: TODO

## 7. Control (Part D)

- Controller design (state feedback or LQR), gain or weight selection with justification: TODO
- Tracking results: TODO

## 8. Integration and robustness (Part E)

- Obstacle avoidance approach: TODO
- Results over repeated trials with varied conditions, and failure diagnosis: TODO

## 9. Limitations and extensions

TODO: what does not work, what the results do and do not support, and any independent extension.

## 10. Contribution statement

| Member | Contribution |
|--------|--------------|
| Lachlan Cameron | TODO |
| Jonathan (Jojo-988) | TODO |
| William (Willl-Je-Suis) | TODO |

## 11. Generative AI declaration

TODO: state the tools used and what they were used for. Update this as your usage changes. Non-declaration is an academic integrity matter.

## 12. Acknowledgements

TODO: list code adapted from the weekly tutorials, MuJoCo examples, the Robotics Toolbox, and the source of the TurtleBot3 MuJoCo model.
