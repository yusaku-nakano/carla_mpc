import os
import decouple
import yaml
import carla
import time
import numpy as np
import casadi as ca
from typing import Tuple

def connect(config):
    # Connect to the CARLA server
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])  # seconds
    world = client.get_world()

    return client, world


def set_synchronous_mode(world, config, return_settings=False, no_rendering=False):
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable synchronous mode
    settings.fixed_delta_seconds = 1.0 / config['fps']  # Set the time step for each simulation tick
    settings.substepping = True
    settings.max_substeps = 10

    if no_rendering:
        settings.no_rendering_mode = True
    world.apply_settings(settings)
    # world.tick() #
    if return_settings:
        return settings

def load_map(client, map_name, minimal_layers):
    if minimal_layers:
        print("Loading with minimal layers")
        world = client.load_world(map_name, map_layers=carla.MapLayer.NONE)
    else:
        world = client.load_world(map_name)

    return world

def load_config():
    config_path = os.path.join(decouple.config("PROJECT_DIR"), decouple.config("MAIN_CONFIG_FILE"))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        # add relevant environment variables
        config["logging"]["directory"] = os.path.join(decouple.config("PROJECT_DIR"), decouple.config("LOG_DIR"))

        return config 
    
def update_fps(frame_count, start_time, last_fps):
    # global frame_count, start_time
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 1:  # Update every second
        fps = frame_count / elapsed_time
        # print(f"FPS: {fps:.2f}")  # Display FPS in the terminal
        # Reset counters
        frame_count = 0
        start_time = current_time
        return fps, frame_count, start_time
    return last_fps, frame_count, start_time

def get_vehicle_dimensions(vehicle):
    # Get the bounding box of the vehicle
    bounding_box = vehicle.bounding_box
    extent = bounding_box.extent

    # The extent gives half the length, width, and height
    vehicle_length = 2 * extent.x
    vehicle_width = 2 * extent.y
    vehicle_height = 2 * extent.z

    return vehicle_length, vehicle_width, vehicle_height

def get_vehicle_blueprint(vehicle, world):
    # Get the type ID of the vehicle
    type_id = vehicle.type_id

    # Retrieve the blueprint from the blueprint library
    blueprint = world.get_blueprint_library().find(type_id)

    return blueprint

def get_vehicle_max_velocity(vehicle, world):
    # Access the vehicle's blueprint
    blueprint = get_vehicle_blueprint(vehicle, world)
    print(blueprint)

    # Check for the 'max_speed' attribute
    if blueprint.has_attribute('max_speed'):
        max_speed = blueprint.get_attribute('max_speed').as_float()
        return max_speed
    else:
        return None


def get_vehicle_max_steering_angle(vehicle):
    # Access the vehicle's physics control
    physics_control = vehicle.get_physics_control()

    # The maximum steering angle is usually in degrees
    max_steer_angle = physics_control.wheels[0].max_steer_angle  # Assuming front left wheel

    return max_steer_angle * np.pi / 180  # Convert to radians

def compute_road_yaw(waypoints_list):
    road_yaw = np.array([np.deg2rad(p.transform.rotation.yaw) for p in waypoints_list])
    road_yaw[road_yaw > np.pi] -= 2 * np.pi
    road_yaw[road_yaw < -np.pi] += 2 * np.pi
    return road_yaw

def compute_arc_length(waypoint_list, x=None, y=None):
    """
    Computes the arc length given x and y coordinates of waypoints
    
    Returns:
    np.ndarray: arc length at each waypoint.
    """
    if x is None:
        x = np.array([point.transform.location.x for point in waypoint_list])
    if y is None:
        y = np.array([point.transform.location.y for point in waypoint_list])

    dx = np.gradient(x)
    dy = np.gradient(y)

    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.cumsum(ds)
    s = np.insert(s[:-1], 0, 0)  # insert 0 at the beginning for the initial point\

    # Ensure arc length is strictly increasing by removing duplicates
    _, unique_indices = np.unique(s, return_index=True)
    s = s[unique_indices]

    return s, unique_indices

def compute_curvature(waypoint_list, unique_indices, x=None, y=None,smoothing=False):
    # Extract global x-y points of waypoints

    if x is None:
        x = np.array([point.transform.location.x for point in waypoint_list])
    if y is None:
        y = np.array([point.transform.location.y for point in waypoint_list])

    # Compute curvature
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    denominator = (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    denominator[denominator == 0] = np.inf

    # todo: review. Setting np.inf makes curvature 0. Perhaps adding small epsilon(eps = 1e-12) would be better?
    curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / denominator

    curvature = np.nan_to_num(curvature)

    curvature = curvature[unique_indices]
    # if smoothing:
    #     curvature = smooth_signal_savgol(curvature, poly_order=3)

    return curvature

def compute_curvature_from_yaw(heading, s):
    """Compute curvature kappa from heading array and distance array."""
    # Ensure arrays are numpy for easier differencing
    heading_np = np.asarray(heading)
    s_np = np.asarray(s)

    # Numerical derivative of heading w.r.t. distance
    dtheta = np.diff(heading_np)
    ds = np.diff(s_np)

    # Avoid divide-by-zero if any ds is zero
    ds[ds == 0] = np.finfo(float).eps

    # curvature[i] is approximate for segment between i and i+1
    kappa = -dtheta / ds

    # Optionally, align length of kappa with heading by, e.g., centering or ignoring last point
    # For demonstration, just return the difference-based array
    return kappa
    

def curvature_function(waypoints_list, x=None, y=None, smoothing=False):
    """
    Creates a CasADi function for curvature as a function of arc length.
    
    Returns:
    CasADi.Function: CasADi function of curvature as a function of arc length.
    """
    arc_length, u_indices = compute_arc_length(waypoints_list, x, y)  # s
    curvature = compute_curvature(waypoints_list, u_indices, x, y, smoothing=smoothing)  # K
    road_yaw = compute_road_yaw(waypoints_list)
    # road_yaw_old = compute_road_yaw_old(waypoints_list)
    curvature_from_road_yaw = compute_curvature_from_yaw(road_yaw, arc_length)

    # Check for non-increasing arc lengths and filter them
    diff_s = np.diff(arc_length)
    if np.any(diff_s <= 0):
        print("Warning: Non-increasing arc length detected. Removing duplicates.")
        arc_length, unique_indices = np.unique(arc_length, return_index=True)
        curvature = curvature[unique_indices]

    # Create CasADi interpolation function
    s = ca.SX.sym('s')
    curvature_interpolant = ca.interpolant('curvature', 'linear', [arc_length], curvature)  # K(s)
    
    # return ca.Function('curvature', [s], [ca.pw_const(s, ca.DM(arc_length), np.concatenate([curvature, [curvature[-1]]]))]), arc_length, curvature_from_road_yaw
    return ca.Function('curvature', [s], [ca.pw_const(s, ca.DM(arc_length), np.concatenate([curvature, [curvature[-1]]]))]), arc_length, road_yaw

def global2frenet(xy,heading,route):
    '''
    xy: Tuple of current x and y positions in global coordinate system
    heading: Float. Current yaw angle of the vehicle in global coordinate system
    route: VehicleReference object

    returns:
    s, ey, epsi
    '''
    s_arr = route.s
    x_arr = route.x
    y_arr = route.y
    psi = route.heading

    norm_array = (x_arr - xy[0])**2 + (y_arr - xy[1])**2
    idx_min = np.argmin(norm_array)

    s_cur = s_arr.item(idx_min)

    #unsigned ey
    ey_cur = np.sqrt(norm_array.item(idx_min))

    #ey sign
    delta_x = x_arr.item(idx_min) - xy[0]
    delta_y = y_arr.item(idx_min) - xy[1]
    delta_vec = np.array([[delta_x],[delta_y]])

    # R: Rotation matrix, C:
    R = np.array([[np.cos(psi.item(idx_min)), np.sin(psi.item(idx_min))], [- np.sin(psi.item(idx_min)), np.cos(psi.item(idx_min))]])
    C = np.array([0,1]).reshape((1,-1)) #Positive y dir
    ey_dir = - np.sign(C @ R @ (delta_vec))
    ey_cur *= ey_dir
    epsi_cur = heading - psi.item(idx_min)

    if epsi_cur>= np.pi:
        epsi_cur -= 2*np.pi
    elif epsi_cur <= -np.pi:
        epsi_cur += 2*np.pi

    return s_cur, ey_cur, epsi_cur, idx_min

class VehicleAction:
    def __init__(self,action_dict: dict):
        self.a = action_dict['a']
        self.df = action_dict['df']

    def update(self,action_dict: dict):
        self.a = action_dict['a']
        self.df = action_dict['df']

class VehicleState:
    def __init__(self, state_dict: dict):
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.heading = state_dict['heading']
        self.v = state_dict['v']
        self.s = state_dict['s']
        self.ey = state_dict['ey']
        self.epsi = state_dict['epsi']
    def update(self, state_dict: dict):
        self.x = state_dict['x']
        self.y = state_dict['y']
        self.heading = state_dict['heading']
        self.v = state_dict['v']

        self.s = state_dict['s']
        self.ey = state_dict['ey']
        self.epsi = state_dict['epsi']

class VehicleReference(VehicleState):
    def __init__(self, state_dict: dict):
        super().__init__(state_dict)
        self.K = state_dict['K']

    def update(self, state_dict: dict):
        super().update(state_dict)
        self.K = state_dict['K']
        
class PlannedTrajectory:
    def __init__(self, timesteps, n_dims: Tuple):
        (nx, nu) = n_dims
        #Frent frame states
        self._frenet_traj = np.zeros((nx,timesteps+1))
        #input trajectory
        self._input_traj = np.zeros((nu,timesteps))
        self._planned_traj = [self._frenet_traj,self._input_traj]

    def __getitem__(self, item):
        return self._planned_traj[item]

    def __setitem__(self, value, value2):
        self._frenet_traj = value
        self._input_traj = value2

    @property
    def x(self):
        return self._frenet_traj[0,:]

    @property
    def y(self):
        return self._frenet_traj[1,:]
    
    @property
    def s(self):
        return self._frenet_traj[2,:]
    
    @property
    def ey(self):
        return self._frenet_traj[3,:]

    @property
    def epsi(self):
        return self._frenet_traj[4,:]
    
    @property
    def v(self):
        return self._frenet_traj[5,:]
    
    @property
    def heading(self):
        return self._frenet_traj[6,:]

    @property
    def a(self):
        return self._input_traj[0,:]

    @property
    def df(self):
        return self._input_traj[1,:]
    
    def set_frenet_traj(self, value):
        self._frenet_traj = value
    
    def set_input_traj(self, value):
        self._input_traj = value
    
    def get_frenet_traj(self):
        return self._frenet_traj
    
    def get_input_traj(self):
        return self._input_traj

class ReferenceTrajectory(PlannedTrajectory):
    def __init__(self, timesteps, n_dims):
        super().__init__(timesteps, n_dims)

    def compute_curvature(self):
        # Computes curvature from the planned trajectory
        return NotImplementedError
    
    @property
    def curvature(self):
        return self.compute_curvature()
    
