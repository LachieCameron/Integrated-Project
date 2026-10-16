import math
import numpy as np
import xml.etree.ElementTree as ET

path = "assets/turtlebot3_waffle_pi.xml"
tree = ET.parse(path)
root = tree.getroot()

base = root.find(".//body[@name='base']")

if base is None:
    raise RuntimeError("Robot base not found")

