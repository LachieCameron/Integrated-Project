import math
import xml.etree.ElementTree as ET

path = "models/turtlebot3_waffle_pi.xml"
tree = ET.parse(path)
root = tree.getroot()

base = root.find(".//body[@name='base']")

if base is None:
    raise RuntimeError("Robot base not found")

# Create 13 sensor sites.
# Each site local +Z points horizontally along
# the requested direction in the robot XY plane.
for angle in range(-90, 91, 15):
    name = f"lidar_{angle}"

    if base.find(f"site[@name='{name}']") is not None:
        continue

    theta = math.radians(angle)

    # Rotation Rz(theta) * Ry(pi/2)
    # Quaternion in MuJoCo order: w, x, y, z
    a = theta / 2
    s = math.sqrt(0.5)

    quat = [
        s * math.cos(a),
        -s * math.sin(a),
        s * math.cos(a),
        s * math.sin(a)
    ]

    ET.SubElement(
        base,
        "site",
        name=name,
        pos="-0.049 0 0.1255",
        quat=" ".join(f"{v:.8f}" for v in quat),
        size="0.005",
        rgba="0 1 0 0.4"
    )

sensor = root.find("sensor")
if sensor is None:
    sensor = ET.SubElement(root, "sensor")

for angle in range(-90, 91, 15):
    name = f"lidar_{angle}"

    if sensor.find(f"rangefinder[@name='{name}']") is None:
        ET.SubElement(
            sensor,
            "rangefinder",
            name=name,
            site=name,
            cutoff="3.0"
        )

tree.write(path, encoding="unicode")
print("13 LiDAR rangefinders added.")