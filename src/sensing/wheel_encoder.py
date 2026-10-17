import mujoco
import numpy as np

class WheelEncoder:
    def __init__(self, model, 
                 left_pos_name="wheel_left_encoder", 
                 right_pos_name="wheel_right_encoder",
                 left_vel_name="wheel_left_velocity",
                 right_vel_name="wheel_right_velocity", 
                 radius=0.033):
        
        # Look up flat memory array slice addresses for position sensors
        self.lp_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, left_pos_name)]
        self.rp_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, right_pos_name)]
        
        # Look up flat memory array slice addresses for velocity sensors
        self.lv_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, left_vel_name)]
        self.rv_adr = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, right_vel_name)]
        
        self.radius = radius
        self.prev_l = 0.0
        self.prev_r = 0.0
        self.initialized = False

    def step(self, data):
        """
        Reads position deltas and current linear wheel velocities.
        
        Returns:
            d_left, d_right: Float distance transitions (meters)
            v_left, v_right: Float instantaneous linear wheel speeds (meters/second)
        """
        # 1. Compute position changes
        curr_l = data.sensordata[self.lp_adr] * self.radius
        curr_r = data.sensordata[self.rp_adr] * self.radius
        
        if not self.initialized:
            self.prev_l = curr_l
            self.prev_r = curr_r
            self.initialized = True
            d_left, d_right = 0.0, 0.0
        else:
            d_left = curr_l - self.prev_l
            d_right = curr_r - self.prev_r
            self.prev_l, self.prev_r = curr_l, curr_r

        # 2. Extract linear speeds directly (converting from rad/s to m/s)
        v_left = data.sensordata[self.lv_adr] * self.radius
        v_right = data.sensordata[self.rv_adr] * self.radius

        return d_left, d_right, v_left, v_right
