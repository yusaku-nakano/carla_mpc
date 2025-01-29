import carla
import random
import math
import numpy as np
from queue import Queue
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import xml.etree.ElementTree as ET
from transforms3d.euler import euler2quat

from rotations import Quaternion, skew_symmetric, angle_normalize, omega

np.set_printoptions(precision = 5, suppress = True)

# Sensor sampling frequency
IMU_FREQ = 200
GNSS_FREQ = 50

# Sensor noise variances
VAR_IMU_ACC = 0.001
VAR_IMU_GYRO = 0.001
VAR_GNSS = np.eye(3) * 100

# Sensor noise profile
NOISE_STDDEV = 5e-6
NOISE_BIAS = 1e-6

# Initial state covariance
POS_VAR = 1
ORIENTATION_VAR = 1000
VELOCITY_VAR = 1000

# Earth radius
EARTH_RADIUS_EQUA = 6378137.0

# Number of GNSS measurements to use for initialization
NUM_MEASUREMENTS = 10

# Initialize world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

#Spectator Camera
spectator = world.get_spectator()

def update_spectator(spectator):
    transform = vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))
    )

# Initialize ego vehicle and sensors
spawn_point = random.choice(world.get_map().get_spawn_points())
bp = world.get_blueprint_library()
vehicle_bp = bp.filter('model3')[0]
imu_bp = bp.filter('sensor.other.imu')[0]
gnss_bp = bp.filter('sensor.other.gnss')[0]

# Set sensor noise
imu_bp.set_attribute('noise_accel_stddev_x', str(NOISE_STDDEV))
imu_bp.set_attribute('noise_accel_stddev_y', str(NOISE_STDDEV))
imu_bp.set_attribute('noise_accel_stddev_z', str(NOISE_STDDEV))
imu_bp.set_attribute('noise_gyro_stddev_x', str(NOISE_STDDEV))
imu_bp.set_attribute('noise_gyro_stddev_y', str(NOISE_STDDEV))
imu_bp.set_attribute('noise_gyro_stddev_z', str(NOISE_STDDEV))
imu_bp.set_attribute('noise_gyro_bias_x', str(NOISE_BIAS))
imu_bp.set_attribute('noise_gyro_bias_y', str(NOISE_BIAS))
imu_bp.set_attribute('noise_gyro_bias_z', str(NOISE_BIAS))
gnss_bp.set_attribute('noise_alt_stddev', str(NOISE_STDDEV))
gnss_bp.set_attribute('noise_lat_stddev', str(NOISE_STDDEV))
gnss_bp.set_attribute('noise_lon_stddev', str(NOISE_STDDEV))
gnss_bp.set_attribute('noise_alt_bias', str(NOISE_BIAS))
gnss_bp.set_attribute('noise_lat_bias', str(NOISE_BIAS))
gnss_bp.set_attribute('noise_lon_bias', str(NOISE_BIAS))

# Set sensor sampling frequency
imu_bp.set_attribute('sensor_tick', str(1.0/IMU_FREQ))
gnss_bp.set_attribute('sensor_tick', str(1.0/GNSS_FREQ))

# Spawn vehicle and sensors
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

imu = world.spawn_actor(
    blueprint = imu_bp,
    transform = carla.Transform(carla.Location(x = 0, z = 0)),
    attach_to = vehicle
)
gnss = world.spawn_actor(
    blueprint = gnss_bp,
    transform = carla.Transform(carla.Location(x = 0, z = 0)),
    attach_to = vehicle
)

# Keep the sensor readings from the callback to queues
imu_queue = Queue()
gnss_queue = Queue()

# Hook sensor readings to callback methods
imu.listen(lambda data: imu_queue.put(data))
gnss.listen(lambda data: gnss_queue.put(data))

# Initializing vehicle states
p_est = np.zeros([3,1])
v_est = np.zeros([3,1])
q_est = np.zeros([4,1]) # Quaternion

# Initializing state covariance
p_cov_est = np.zeros([9,9])

# Gravity
g = np.array([0, 0, -9.81]).reshape(3,1) # gravity

# Motion model noise Jacobian
L_jac = np.zeros([9,6]) 
L_jac[3:, :] = np.eye(6)

# Measurement model Jacobian
H_jac = np.zeros([3,9]) 
H_jac[:, :3] = np.eye(3) 

# Initialize lists for plotting
x_gt_data = []
y_gt_data = []
x_est_data = []
y_est_data = []

# Initialize plot
plt.ion()
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.set_title("Ego Vehicle Position")
p_gt_line, = ax1.plot([], [], label = "Ground Truth", color = "r")
p_est_line, = ax1.plot([], [], label = "Estimated", color = "g")
ax1.legend()
plt.grid()

# Initialize the estimated coordinates using GNSS measurements
gnss_init_xyz = None
num_init_measurements = 0
ekf_initialized = False

def init_state_from_gnss(gnss_data, gnss_init_xyz, num_init_measurements):
    """
    Initialize the vehicle state by taking the average of NUM_MEASUREMENTS readings from GNSS
    to get the abosolute position of the vehicle
    """
    if gnss_init_xyz is None:
        # If this is the first GNSS reading, initialize the state to the GNSS reading
        gnss_init_xyz = np.array([gnss_data[0],
                                  gnss_data[1],
                                  gnss_data[2]])
    else:
        # Otherwise, add the GNSS reading to the array
        gnss_init_xyz[0] += gnss_data[0]
        gnss_init_xyz[1] += gnss_data[1]
        gnss_init_xyz[2] += gnss_data[2]
    # Increment the number of GNSS measurements taken
    num_init_measurements += 1
    # print("Number of GNSS measurements taken: ", num_init_measurements)	
    # print("gnss_init_xyz: ", gnss_init_xyz)
    return gnss_init_xyz, ekf_initialized, num_init_measurements

def _get_latlon_ref():
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref
gnss_lat_ref, gnss_lon_ref = _get_latlon_ref()

def gnss_to_xyz(latitude, longitude, altitude):
    """Creates Location from GPS (latitude, longitude, altitude).
    This is the inverse of the _location_to_gps method found in
    https://github.com/carla-simulator/scenario_runner/blob/master/srunner/tools/route_manipulation.py
    
    Modified from:
    https://github.com/erdos-project/pylot/blob/master/pylot/utils.py
    """
    scale = math.cos(gnss_lat_ref * math.pi / 180.0)
    basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * gnss_lon_ref
    basey = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + gnss_lat_ref) * math.pi / 360.0))

    x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
    y = scale * EARTH_RADIUS_EQUA * math.log(
        math.tan((90.0 + latitude) * math.pi / 360.0)) - basey

    # This wasn't in the original method, but seems to be necessary.
    y *= -1

    return np.array([x, y, altitude]).reshape(3, 1)

def get_sensor_readings(frame):
    """
    This function returns the sensor readings at a particular frame
    """
    sensors = {'imu': None, 
               'gnss': None}
    while not imu_queue.empty():
        imu_data = imu_queue.get()
        if imu_data.frame == frame:
            sensors['imu'] = imu_data
            imu_queue.task_done()
            break
        imu_queue.task_done()
    while not gnss_queue.empty():
        gnss_data = gnss_queue.get()
        if gnss_data.frame == frame:
            alt = gnss_data.altitude
            lat = gnss_data.latitude
            lon = gnss_data.longitude
            gps_xyz = gnss_to_xyz(lat, lon, alt)
            sensors['gnss'] = gps_xyz
            gnss_queue.task_done()
            break
        gnss_queue.task_done()
    return sensors

def carla_rotation_to_RPY(carla_rotation):
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)
    return (roll, pitch, yaw)

def carla_rotation_to_ros_quaternion(carla_rotation):
    roll, pitch, yaw = carla_rotation_to_RPY(carla_rotation)
    quat = euler2quat(roll, pitch, yaw)
    return quat

def predict_state(imu_f, imu_w, delta_t, p_est, v_est, q_est, p_cov_est):
    # vfa = VAR_IMU_ACC**2
    # vfw = VAR_IMU_GYRO**2
    # Q_km = delta_t**2 * np.diag([vfa,vfa,vfa,vfw,vfw,vfw])

    C_ns = Quaternion(*q_est).to_mat()
    p_est = p_est + delta_t * v_est + 0.5 * delta_t * delta_t * (C_ns @ imu_f + g)
    v_est = v_est + delta_t * (C_ns @ imu_f + g)
    q_est = Quaternion(axis_angle = imu_w * delta_t).quat_mult_right(q_est)
    # q_est = omega(imu_w, delta_t) @ q_est

    # Update covariance
    F_k = np.eye(9)
    F_k[:3,3:6] = np.eye(3) * delta_t
    # F_k[3:6,6:9] = skew_symmetric(np.dot(C_ns, imu_f.reshape(3,1)))
    F_k[3:6,6:] = -skew_symmetric(C_ns @ imu_f.reshape(3,1)) * delta_t

    Q_km = np.eye(6)
    Q_km[3:,:3] *= delta_t * delta_t * VAR_IMU_ACC
    Q_km[:3,3:] *= delta_t * delta_t * VAR_IMU_GYRO

    L_k = np.zeros((9,6))
    # L_k[3:6,0:3] = np.eye(3)
    # L_k[6:9,3:6] = np.eye(3)
    L_k[3:,:] = np.eye(6)

    # Propagate uncertainty
    p_cov_est = F_k @ p_cov_est @ F_k.T + L_k @ Q_km @ L_k.T
    return p_est, v_est, q_est, p_cov_est

def correct_state(gnss_data, p_est, v_est, q_est, p_cov_est):
    # Global position
    gnss_x = gnss_data[0]
    gnss_y = gnss_data[1]
    gnss_z = gnss_data[2]
    y_k = np.array([gnss_x, gnss_y, gnss_z])

    # Compute Kalman Gain
    K_k = p_cov_est @ H_jac.T @ np.linalg.inv(H_jac @ p_cov_est @ H_jac.T + VAR_GNSS)

    # Compute error state
    delta_x = K_k @ (y_k - p_est)

    # Correct predicted state
    delta_p = delta_x[:3]
    delta_v = delta_x[3:6]
    delta_phi_normalized = angle_normalize(delta_x[6:])

    p_corr = p_est + delta_p
    v_corr = v_est + delta_v
    q_corr = Quaternion(axis_angle = delta_phi_normalized).quat_mult_left(q_est)

    # Compute corrected covariance
    p_cov_corr = (np.eye(9) - K_k @ H_jac) @ p_cov_est

    return p_corr, v_corr, q_corr, p_cov_corr

def destroy():
    imu.destroy()
    gnss.destroy()
    carla.command.DestroyActor(vehicle)

timestep_initialized = False
initialization_complete = False
try:
    while True:
        world.tick()
        frame = world.get_snapshot().frame

        # Update spectator camera
        update_spectator(spectator)

        # Get sensor readings
        sensors = get_sensor_readings(frame)

        # EKF Initialization using GNSS
        if not ekf_initialized:
            if sensors['gnss'] is not None:
                gnss_init_xyz, ekf_initialized, num_init_measurements = init_state_from_gnss(sensors['gnss'], gnss_init_xyz, num_init_measurements)
                if num_init_measurements == NUM_MEASUREMENTS:
                    gnss_init_xyz /= NUM_MEASUREMENTS
                    # Initial position is the average of the first NUM_MEASUREMENTS GNSS measurements
                    p_est = gnss_init_xyz

                    # r_init = vehicle.get_transform().rotation
                    # euler_init = np.array([r_init.roll, r_init.pitch, r_init.yaw])
                    # q_est = Quaternion(euler = euler_init).to_numpy()
                    # v_init = vehicle.get_velocity()
                    # v_est = np.array([v_init.x, v_init.y, v_init.z]).reshape(3,1)

                    q_est = Quaternion().to_numpy()

                    p_cov_est[:3, :3] = np.eye(3) * POS_VAR
                    p_cov_est[3:6, 3:6] = np.eye(3) * VELOCITY_VAR
                    p_cov_est[6:, 6:] = np.eye(3) * ORIENTATION_VAR
                    ekf_initialized = True

                    print("Estimated initial position:", p_est)
                    print("Ground truth initial position:", vehicle.get_location())

                    # print("Estimated initial quaternion:", q_est)
                    # print("Ground truth initial quaternion:", carla_rotation_to_ros_quaternion(vehicle.get_transform().rotation))
            continue

        if sensors['imu'] is not None and not timestep_initialized:
            imu_data = sensors['imu']	
            last_timestamp = imu_data.timestamp
            # print("First timestamp: ", last_timestamp)
            timestep_initialized = True
            continue
        # If IMU measurement is available, predict the state
        if sensors['imu'] is not None and timestep_initialized:
            imu_data = sensors['imu']
            imu_f = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z]).reshape(3,1)
            imu_w = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z]).reshape(3,1)
            print("IMU Gyro Data: ", imu_w)
            
            delta_t = imu_data.timestamp - last_timestamp
            last_timestamp = imu_data.timestamp

            p_est, v_est, q_est, p_cov_est = predict_state(imu_f, imu_w, delta_t, p_est, v_est, q_est, p_cov_est)

            print("Predicted position:", p_est)
            print("Ground truth position:", vehicle.get_location())

            # print("Predicted quaternion:", q_est)
            # print("Ground truth quaternion:", carla_rotation_to_ros_quaternion(vehicle.get_transform().rotation))

            # Append estimated vehicle position for plotting
            x_est_data.append(p_est[0])
            y_est_data.append(p_est[1])

            # Retrieve ground truth vehicle state
            gt_location = vehicle.get_location()
            x_gt_data.append(gt_location.x)
            y_gt_data.append(gt_location.y)

            p_gt_line.set_data(x_gt_data, y_gt_data)
            p_est_line.set_data(x_est_data, y_est_data)

            ax1.relim()
            ax1.autoscale_view()
            plt.pause(0.01)

        # If GNSS measurement is available, correct the state
        if sensors['gnss'] is not None:
            gnss_data = sensors['gnss']
            p_corr, v_corr, q_corr, p_cov_corr = correct_state(gnss_data, p_est, v_est, q_est, p_cov_est)
            p_est, v_est, q_est, p_cov_est = p_corr, v_corr, q_corr, p_cov_corr
            
            print("Corrected position:", p_corr)
            print("Ground truth position:", vehicle.get_location())

            # print("Corrected quaternion:", q_est)
            # print("Ground truth quaternion:", carla_rotation_to_ros_quaternion(vehicle.get_transform().rotation))

            # Append estimated vehicle position for plotting
            x_est_data.append(p_corr[0])
            y_est_data.append(p_corr[1])

            # Retrieve ground truth vehicle state
            gt_location = vehicle.get_location()
            x_gt_data.append(gt_location.x)
            y_gt_data.append(gt_location.y)

            p_gt_line.set_data(x_gt_data, y_gt_data)
            p_est_line.set_data(x_est_data, y_est_data)

            ax1.relim()
            ax1.autoscale_view()
            plt.pause(0.01)
finally:
    destroy()