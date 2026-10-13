
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path(
    "models/scene_turtlebot3_waffle_pi.xml"
)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.monotonic()

        # Forward for 4 seconds
        while viewer.is_running() and time.monotonic() - start < 4:
            data.ctrl[0] = 3.0
            data.ctrl[1] = 3.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # Rotate for 3 seconds
        start = time.monotonic()

        while viewer.is_running() and time.monotonic() - start < 3:
            data.ctrl[0] = -2.0
            data.ctrl[1] = 2.0

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # Stop
        data.ctrl[:] = 0

        mujoco.mj_step(model, data)
        viewer.sync()

        break
