import mujoco
import numpy as np

class IMUSensor:
    def __init__(self, model, accel_name="imu_accel", gyro_name="imu_gyro"):
        # self.model = model
        self.accel_id = model.sensor(accel_name).id
        self.gyro_id = model.sensor(gyro_name).id

        if self.accel_id == -1 or self.gyro_id == -1:
            raise ValueError("IMU sensors not found in the model")

        self.accel_adr = model.sensor_adr[self.accel_id]
        self.gyro_adr = model.sensor_adr[self.gyro_id]

        
    def read(self, data):

        accel = data.sensordata[self.accel_adr:self.accel_adr + 3]
        gyro = data.sensordata[self.gyro_adr:self.gyro_adr + 3]

        return np.array(accel), np.array(gyro)