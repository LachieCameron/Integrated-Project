
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path(
    "assets/scene_turtlebot3_waffle_pi.xml"
)

data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data, camera="raspi_cam_v2.1")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.monotonic()

        # Forward for 4 seconds
        while viewer.is_running() and time.monotonic() - start < 4:
            with viewer.lock():
                data.ctrl[0] = 3.0
                data.ctrl[1] = 3.0
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # Rotate for 10 seconds
        start = time.monotonic()

        while viewer.is_running() and time.monotonic() - start < 10:
            with viewer.lock():
                data.ctrl[0] = -2.0
                data.ctrl[1] = 2.0
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # Stop
        with viewer.lock():
            data.ctrl[:] = 0
            mujoco.mj_step(model, data)
        viewer.sync()

        # Keep the viewer alive so closing the window performs the shutdown.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)
