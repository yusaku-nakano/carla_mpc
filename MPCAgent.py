import copy
import pdb
from warnings import WarningMessage

import carla
import sys
import decouple
import os
import importlib
import numpy as np
import time
import matplotlib.pyplot as plt

from utils import curvature_function, get_vehicle_dimensions, get_vehicle_max_velocity, \
    get_vehicle_max_steering_angle, global2frenet, VehicleState, VehicleAction, VehicleReference
from prediction_utils import ActorState

CARLA_PATH = os.path.join(decouple.config("CARLA_DIR"), "PythonAPI/carla")

class BaseAgent:
    def __init__(self, ego, config, world_model, route=None):
        self.ego = ego
        self.config = config
        self.name = self.config["agent"]["name"]
        self.route = route
        self.world_model = world_model
        self.predictor = self.load_predictor()
    
    def load_predictor(self):
        """
        Dynamically loads a prediction model based on the configuration provided in config.yaml
        Crucial for agent's ability to predict the future states of other actors in the environment
        """
        try:
            # Imports the prediction.models module, which contains the implementation of various prediction models
            module = importlib.import_module("prediction_models")
            # Retrieves the specific class corresponding to the prediction model name specified in the config.yaml
            class_ = getattr(module, self.config["prediction"]["model"])
        except AttributeError as e:
            # Handles the case where the specified prediction model class is not found in the prediction.models module
            print(
                f"[Error] The provided prediction model {self.config['prediction']['model']} in config.yaml was not found. Make sure there is a class implementation matching the same name. Full trace: "
            )
            raise e
        # Instantiates the predictor model using the class retrieved in the previous step
        predictor = class_(self.config["prediction"])
        return predictor
    
    def perceive(self):
        """
        Simulates the perception process and updates the world model with perceived data
        """
        # If the agent is NOT a custom agent...
        if self.name != "Custom":
            # Update the actors of the world model with the ground truth actors
            self.world_model.actors = self.world_model.actors_gt
            # Update the map of world model with the ground truth map
            self.world_model.map = self.world_model.map_gt
            # Exit early since predefined CARLA agents don't perform any additional perception tasks
            return
        # If the agent IS a custom agent, call the "perceive" method of the custom agent
        self.agent.perceive()
    
    def predict_carla_agent(self, carla_actor):
        """
        Predicts the future state of a specific CARLA actor (e.g. vehicle, pedestrian) using the loaded predictor model
        """
        # Start time of prediction process for given actor
        t_start_actor = time.time()
        # Creates ActorState object for given CARLA actor
        actor_state = ActorState(carla_actor)

        # Arguments required by predictor model to make prediction
        predictor_args = {
            "actor_state": actor_state,
            "actor_id": carla_actor.id
        }

        # Calls "predict" method of predictor model to generate predictions for the actor
        actor_predictions = self.predictor.predict(predictor_args)
        # Updates the world model with the predictions generated for the actor
        self.world_model.predictions[carla_actor.id] = actor_predictions
    
    def predict(self):
        """
        Runs prediction process for all detected actors in the environment and
        updates the world model with the predictions
        """
        # seld.world_model.predictions: dictionary that stores predictions for all actors in environment,
        # indexed by their unique IDs
        self.world_model.predictions = {}
        self.predictor.prepare_data(self.world_model)

        # Filters list of actors in world model to include only vehicles and pedestrians
        carla_actors_filtered = [actor for actor in self.world_model.actors if ("vehicle" in actor.type_id or "walker" in actor.type_id)]

        # For each detected actor...
        for carla_actor in carla_actors_filtered:
            # Skip prediction process for ego vehicle if configured to do so
            if not self.config["prediction"]["predict_ego"] and carla_actor.id == self.ego.id:
                continue 
            # Otherwise, predict the future state of the current actor using the predictor model
            self.predict_carla_agent(carla_actor)
    
    def plan(self):
        raise NotImplementedError
    
    def control(self):
        raise NotImplementedError
    
    def set_destination(self, destination):
        raise NotImplementedError
    
    def done(self):
        raise NotImplementedError
    
    def get_ego_velocity(self):
        return np.sqrt((self.ego.get_velocity().x ** 2) + (self.ego.get_velocity().y ** 2))
    
    def get_ego_acceleration(self):
        return np.sqrt((self.ego.get_acceleration().x ** 2) + (self.ego.get_acceleration().y ** 2))
    
    def get_ego_heading(self, rad=False):
        yaw = self.ego.get_transform().rotation.yaw
        if rad:
            yaw = np.deg2rad(yaw)
        return yaw

class MPCAgent(BaseAgent):
    def __init__(self, ego, config, world_model, route, destination):
        # Ensure any initialization logic in parent class is executed first
        super().__init__(ego, config, world_model, route)

        # Capture initial position (x,y) of ego vehicle; useful for tracking starting point
        self.xy0 = (self.ego.get_location().x, self.ego.get_location().y)
        self.name = self.config["agent"]["name"] # name of agent
        self.route = route # set of waypoints to follow
        self.dt = self.config['planner']['dt'] # timestep
        # Counter to keep track of how many consecutive times planner failed to find feasible solution
        self.infeas_tick = 0 
        # Initialize throttle and brake
        self.throttle = 0
        self.brake = 0
        # Initialize previous control inputs
        self.u_prev = [0,0]
        self.u_prev_prev = [0,0]

        # Initialize variables to store the last planned trajectory and last cost
        # Used to detect errors in QP solver
        self.last_planned_x = None
        self.last_planned_y = None
        self.last_cost = None

        # Configure how obstacles are filtered based on predictor being used
        self.set_obstacle_filtering()
        # Initialize a flag to indicate whether planned trajectory exists
        self.planned_traj_exists = False

        # Dynamically load the planner class specified in configuration file
        # If specified planner class is not found, raise error
        try:
            planner_module = importlib.import_module("MPCPlanner")
            planner_class = getattr(planner_module, self.config["planner"]["model"])
            print("Planner being used: ", planner_class)
        except AttributeError as e:
            print(
                f"[Error] The provided planner model {self.config['planner']['model']} in config.yaml was not found. Make sure there is a class implementation matching the same name. Full trace: "
            )
            raise e
        
        # Compute the global curvature function: K(s)
        curvature, s, road_yaw = curvature_function([point[0] for point in self.route], smoothing=False)

        # Global reference trajectory from origin and destination
        self.ref = VehicleReference({'K': curvature,
                                     'x': np.array([point[0].transform.location.x for point in self.route]),
                                     'y': np.array([point[0].transform.location.y for point in self.route]),
                                     's': s,
                                     'ey': np.zeros((len(self.route),)),
                                     'epsi': np.zeros((len(self.route),)),
                                     'v': self.config["planner"]["v_max"] * np.ones((len(self.route),)),
                                     'heading': road_yaw})
        # self.visualize_reconstructed_path()
        
        # Initialize time counter
        self.t = 0

        # Ego vehicle parameters (dimensions, max steering angle, max velocity)
        # Store parameters in self.ego_params an dupdate in planner configuration
        l,w,h = get_vehicle_dimensions(self.ego)
        sa_max = get_vehicle_max_steering_angle(self.ego)
        v_max = get_vehicle_max_velocity(self.ego, self.world_model.world)
        # print("v_max: ", v_max)
        self.ego_params = {'L': l,
                           'w': w,
                           'h': h,
                           'sa_max': sa_max,
                           'v_max': min(v_max,self.config['planner']['v_max']) if v_max else self.config["planner"]["v_max"]}
        self.config['planner'].update(self.ego_params)

        # Convert ego vehicle's position and heading from global -> frenet frame
        # Useful for path finding
        self.s, self.ey, self.epsi, idx_min = global2frenet(xy = (self.ego.get_location().x, self.ego.get_location().y),
                                                            heading = (self.ego.get_transform().rotation.yaw) * np.pi/180,
                                                            route = self.ref)
        # Construct initial state vector x0
        x0 = np.array([self.ego.get_location().x,
                       self.ego.get_location().y,
                       self.s,
                       float(self.ey),
                       self.epsi,
                       self.get_ego_velocity(),
                       self.ego.get_transform().rotation.yaw * np.pi/180,
                       0,
                       0])
        # Extract and store initial position, heading, and velocity from state vector
        self.x = x0[0]
        self.y = x0[1]
        self.heading = x0[6]
        self.v = x0[5]
        # Initialize planner with configuration, reference trajectory, and initial state vector
        self.planner = planner_class(self.config['planner'], self.ref, x0)
        # Apply Ackermann controller settings
        self.ego.apply_ackermann_controller_settings(carla.AckermannControllerSettings(speed_kp = .4,
                                                                                       speed_ki = .3,
                                                                                       speed_kd = .3,
                                                                                       accel_kp = 0.02,
                                                                                       accel_ki = 0.3,
                                                                                       accel_kd = 0.03))
    def visualize_reconstructed_path(self):
        """
        Recursively visualize both original and reconstructed paths during each planning step
        """
        # Store reconstructed points
        x_reconstructed = []
        y_reconstructed = []
        
        # Start from the first point of reference trajectory
        x_reconstructed.append(self.ref.x[0])
        y_reconstructed.append(self.ref.y[0])
        
        # Reconstruct points using forward integration
        ds = 0.1  # Small step size for integration
        s_current = 0
        
        while s_current < self.ref.s[-1]:
            # Find current index in the reference path
            idx = np.searchsorted(self.ref.s, s_current) - 1
            idx = max(0, min(idx, len(self.ref.s) - 1))
            
            # Get current position
            x_current = x_reconstructed[-1]
            y_current = y_reconstructed[-1]
            
            # Get current heading and curvature
            heading = self.ref.heading[idx]
            k = self.ref.K(s_current)
            
            # Update heading based on curvature
            heading_new = heading + k * ds
            
            # Update position
            x_new = x_current + ds * np.cos(heading)
            y_new = y_current + ds * np.sin(heading)
            
            x_reconstructed.append(x_new)
            y_reconstructed.append(y_new)
            
            s_current += ds
        
        # Convert to numpy arrays
        x_reconstructed = np.array(x_reconstructed)
        y_reconstructed = np.array(y_reconstructed)
        
        # Clear previous visualizations (optional, depending on your preference)
        self.world_model.world.debug.draw_point(
            carla.Location(x=0, y=0, z=0),
            size=0.1,
            color=carla.Color(r=0, g=0, b=0),
            life_time=0.01
        )
        
        # Visualize both paths
        # Original reference path (green)
        for i in range(len(self.ref.x)-1):
            self.world_model.world.debug.draw_line(
                carla.Location(x=self.ref.x[i], y=self.ref.y[i], z=1.0),
                carla.Location(x=self.ref.x[i+1], y=self.ref.y[i+1], z=1.0),
                thickness=0.1,
                color=carla.Color(r=0, g=255, b=0),  # Green
                life_time=0.1  # Short lifetime for recursive updates
            )
        
        # Reconstructed path (red)
        for i in range(len(x_reconstructed)-1):
            self.world_model.world.debug.draw_line(
                carla.Location(x=x_reconstructed[i], y=y_reconstructed[i], z=1.2),
                carla.Location(x=x_reconstructed[i+1], y=y_reconstructed[i+1], z=1.2),
                thickness=0.1,
                color=carla.Color(r=255, g=0, b=0),  # Red
                life_time=0.1  # Short lifetime for recursive updates
            )
        
        # Draw current vehicle position and heading
        ego_x = self.ego.get_location().x
        ego_y = self.ego.get_location().y
        ego_heading = (self.ego.get_transform().rotation.yaw) * np.pi/180
        
        # Vehicle position (blue dot)
        self.world_model.world.debug.draw_point(
            carla.Location(x=ego_x, y=ego_y, z=1.5),
            size=0.1,
            color=carla.Color(r=0, g=0, b=255),  # Blue
            life_time=0.1
        )
        
        # Vehicle heading (blue arrow)
        arrow_length = 2.0
        self.world_model.world.debug.draw_arrow(
            carla.Location(x=ego_x, y=ego_y, z=1.5),
            carla.Location(
                x=ego_x + arrow_length * np.cos(ego_heading),
                y=ego_y + arrow_length * np.sin(ego_heading),
                z=1.5
            ),
            thickness=0.1,
            arrow_size=0.2,
            color=carla.Color(r=0, g=0, b=255),  # Blue
            life_time=0.1
        )
        
        # Calculate and print error statistics
        errors = []
        for i in range(len(self.ref.x)):
            idx = np.argmin((x_reconstructed - self.ref.x[i])**2 + (y_reconstructed - self.ref.y[i])**2)
            error = np.sqrt((x_reconstructed[idx] - self.ref.x[i])**2 + (y_reconstructed[idx] - self.ref.y[i])**2)
            errors.append(error)
        
        print("\r" + f"Mean reconstruction error: {np.mean(errors):.3f} m, Max error: {np.max(errors):.3f} m", end="")
    
    def set_obstacle_filtering(self):
        """
        Configure how obstacles are filtered based on the predictor being used
        """
        # Retrieves the "predictor" object associated with agent
        predictor = self.predictor

        # Define default configuration parameters that can be overriden based on the predictor's behavior
        overridables = {
            'num_modes_per_vehicle': 1, # Only one predicted trajectory per vehicle will be considered
            'obstacle_filtering': 'distance' # Obstacles will be filtered based on distance from ego vehicle
        }
        # If the predictor is a Constant Velocity model, override obstacle filtering
        if predictor.name == "ConstantVelocity" or predictor.alias == "CV":
            print("[AVAgent] Overriding obstacle filtering")
            self.override_planner_config(overridables)
            return
        # If the predictor generates fewer than 2 trajectories, override obstacle filtering
        if hasattr(predictor, "n_trajectories") and predictor.n_trajectories < 2:
            print("[AVAgent] Overriding obstacle filtering")
            self.override_planner_config(overridables)
            return
    
    def override_planner_config(self, overridables):
        """
        Helper function used to update the planner's configuration based on the overridables dictionary
        Ensures that specific parameters in the planner's configuration are overriden with new values
        """
        # Iterate over overridable parameters
        for param_name, param_value in overridables.items():
            # Update planner configuration
            # Example: 
            # If param_name = 'num_modes_per_vehicle' and param_value = 1:
            #  Planner will be configured to consider only one predicted trajectory per vehicle
            # If param_name = 'obstacle_filtering' and param_value = 'distance':
            #  Planner will filter obstacles based on distance from ego vehicle
            self.config["planner"][param_name] = param_value
    
    def perceive(self):
        """
        Run (or simulate) perception and update world model
        Should be easy to override with custom perception functionality
        For now, updates with ground truth
        """
        self.world_model.actors = self.world_model.actors_gt
        self.world_model.map = self.world_model.map_gt
    
    def check_undetected_error_3(self, planned_traj):
        """
        Detects a specific type of error in the planner's optimization process 
        "QP solver error status 3"; not explicitly caught by planner itself
        """
        # Boolean flag that detects conditions that suggests presence of undetected error
        possible_error_3 = False
        # Retrieves latest cost from planner; measure of how well planned trajectory aligns with desired objectives
        latest_cost = self.planner.last_cost
        # Create deep copies of planned trajectory's x and y coordinates
        # Deep copy: ensures values are independent of original trajectory object, preventing unintended modifications
        latest_planned_x = copy.deepcopy(planned_traj.x)
        latest_planned_y = copy.deepcopy(planned_traj.y)

        # Conditions that suggest presence of undetected error:
        # 1. If the planner found a feasible solution
        # 2. Cost of latest planned trajectory is the same as the cost of the previously planned trajectory
        # 3. x-coordinates of latest planned trajectory are identical to those of previously planned trajectory
        # 4. y-coordinates of latest planned trajectory are identical to those of previously planned trajectory
        if (self.opt_flag
                and latest_cost == self.last_cost
                and np.all(latest_planned_x == self.last_planned_x)
                and np.all(latest_planned_y == self.last_planned_y)):
            possible_error_3 = True
        
        # Updates stored values of last planned trajectory (x,y) and last cost with latest values
        self.last_planned_x = latest_planned_x
        self.last_planned_y = latest_planned_y
        self.last_cost = latest_cost

        return possible_error_3
    
    def plan(self):
        """
        Core Planning Function of the MPCAgent Class
        Responsible for generating a trajectory for the ego vehicle to follow, based on the 
        current state of the world, the ego vehicle's position, and the reference trajectory
        """
        # Initialize a list to store the dimensions (length, width) of other vehicles in the environment
        self.tv_params = []
        # For each predicted vehicle (key in self.world_model.predictions)...
        for key in self.world_model.predictions.keys():
            # Retrieve bounding box extend of the vehicle
            extent = self.world_model.actors_carla.find(key).bounding_box.extent
            # Append length and width of vehicle to self.tv_params
            self.tv_params.append([extent.x*2, extent.y*2])
        # Retrieve dimensions of ego vehicle from its bounding box and store them in self.ego_dim
        self.ego_dim = [self.world_model.ego.bounding_box.extent.x*2, 
                        self.world_model.ego.bounding_box.extent.y*2]
        
        # print("SKIBIDIIIII TOILETTTT")
        # self.visualize_reconstructed_path()
        # If this is the initial timestep...
        if self.t == 0:
            # Ego vehicle's position and heading are converted from Global -> Frenet
            self.s, self.ey, self.epsi, idx_min = global2frenet(
                                                                xy = (self.ego.get_location().x, self.ego.get_location().y),
                                                                heading = (self.ego.get_transform().rotation.yaw) * np.pi/180,
                                                                route = self.ref
            )
            # Lateral error (self.ey) is extracted and stored
            self.ey = self.ey[0][0]
        # For subsequent timesteps...
        else:
            # Ego vehicle's position and heading are converted from Global -> Frenet    
            _, self.ey, _, idx_min = global2frenet(
                                                    xy = (self.ego.get_location().x, self.ego.get_location().y),
                                                    heading = (self.ego.get_transform().rotation.yaw) * np.pi/180,
                                                    route = self.ref
            )
        
        # Create a VehicleReference object (curr_ref) to represent current reference trajectory
        curr_ref = VehicleReference({
                                    'K': self.ref.K,
                                    'x': 0,
                                    'y': 0,
                                    's': 0,
                                    'ey': 0,
                                    'epsi': 0,
                                    'v': 0,
                                    'heading': 0
        })

        # If we're using Ackermann control...
        if self.config['planner']['control'] == 'Ackermann':
            # The ego vehicle's state (position, velocity, heading, etc) and previous control inputs (acc, steering angle) are included
            planner_args = {'ref': curr_ref,
                            'actor_state': VehicleState({
                                                        'x': self.ego.get_location().x,
                                                        'y': self.ego.get_location().y,
                                                        's': self.s,
                                                        'ey': -self.ey,
                                                        'epsi': self.epsi,
                                                        'v': self.ego.get_velocity(),
                                                        'heading': (self.ego.get_transform().rotation.yaw) * np.pi/180}),
                            'prev_action': VehicleAction({
                                                        'a': self.get_ego_acceleration,
                                                        'df': self.ego.get_control().steer}),
                            'obstacle_preds': self.world_model.predictions,
                            'ego_route': self.route,
                            'tv_params': self.tv_params,
                            'ego_dim': self.ego_dim}
        # If we're using Perfect control...
        else:
            # The ego vehicle's state and previous control inputs are included, 
            # but the values are taken from the agent's internal state state variables (self.x, self.y, etc)
            planner_args = {'ref': curr_ref,
                            'actor_state': VehicleState({
                                                        'x': self.x,
                                                        'y': self.y,
                                                        's': self.s,
                                                        'ey': float(self.ey),
                                                        'epsi': self.epsi,
                                                        'v': self.v,
                                                        'heading': self.heading}),
                            'prev_action': VehicleAction({
                                                        'a': self.u_prev[0],
                                                        'df': self.u_prev[1]}),
                            'obstacle_preds': self.world_model.predictions,
                            'ego_route': self.route,
                            'tv_params': self.tv_params,
                            'ego_dim': self.ego_dim}
        
        # Call the planner
        planned_traj, self.opt_flag = self.planner.plan(planner_args)
        print("[navigation_agents.py, plan()] self.opt_flag:", self.opt_flag)

        # Check for undetected errors (QP solver error status 3)
        self.possible_error_3 = self.check_undetected_error_3(planned_traj)
        if self.possible_error_3:
            print("[navigation_agents] detected possible uncaught error 3")
            self.opt_flag = False
        
        # Convert planned trajectory to waypoints
        # If the planner found a feasible solution...
        if self.opt_flag:
            self.planned_traj_exists = True
            # Store planned trajectory in self.planned_traj
            self.planned_traj = planned_traj
            planned_points_list = []
            for i in range(self.config["planner"]["N"]+1):
                x = self.planned_traj.x[i]
                y = self.planned_traj.y[i]
                heading = self.planned_traj.heading[i]
                v = self.planned_traj.v[i]
                a = self.planned_traj.a[i] if i < self.config["planner"]["N"] else np.nan
                ey = self.planned_traj.ey[i]
                epsi = self.planned_traj.epsi[i]
                s = self.planned_traj.s[i]
                # Planned trajectory converted into list of waypoints (planned_points_list)
                planned_points_list.append((x, y, heading, v, a, ey, epsi, s))
            # List of waypoints stored in self.world_model.planned_route
            self.world_model.planned_route = planned_points_list
        # If the planner did NOT find a feasible solution...
        else:
            # and if this is the first timestep...
            if self.t < 1:
                # Store the infeasible trajectory in self.planned_traj
                self.planned_traj = planned_traj
            # Increment the infeasibility counter
            self.infeas_tick += 1
            planned_points_list = []
            # List of NaN values to represent invalid planned trajectory
            for i in range(self.config["planner"]["N"]+1):
                planned_points_list.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            # Invalid trajectory stored in self.world_model.planned_route
            self.world_model.planned_route = planned_points_list
        # Increment timestep
        self.t += 1

    def control(self):
        """
        Executes the control actions (steering, acc) based on planned trajectory generated by "plan" method
        Ensures ego vehicle follows planned trajectory or handles infeasible plans
        """
        # If we're using Ackermann control...
        if self.config['planner']['control'] == 'Ackermann':
            # and if the planner found a feasible solution...
            if self.opt_flag:
                # Extract df, a, v from planned trajectory
                df = self.planned_traj.df[0]
                a = self.planned_traj.a[0] if self.t > 1 else 2
                v = self.planned_traj.v[1]
                print(f'\n Steer: {df}, Accel: {a}')

                # VehicleAckermannControl object created with computed steering angle, acc, max speed, and jerk values
                self.next_control = carla.VehicleAckermannControl(steer = df,
                                                                  acceleration = a,
                                                                  speed = self.config["planner"]["v_max"],
                                                                  jerk = 0.9)
                # Store control input
                self.world_model.control = self.next_control
                # Apply the control input
                self.ego.apply_ackermann_control(self.next_control)
            else:
                # If infeasible, apply maximum braking (acc = -4) with desired speed = 0
                self.next_control = carla.VehicleAckermannControl(steer = 0,
                                                                  acceleration = -4,
                                                                  speed = 0)
                # Store and apply control input
                self.world_model.control = self.next_control
                self.ego.apply_ackermann_control(self.next_control)
        # If we're using Perfect control...
        elif self.config['planner']['control'] == 'Perfect':
            # and if the planner found a feasible solution...
            if self.opt_flag:
                # Reset infeasibility counter to 0
                self.infeas_tick = 0
                # Ego vehicle's position, heading, and velocity are updated to match the planned trajectory
                self.ego.set_transform(carla.Transform(
                                                       location = carla.Location(x = self.planned_traj.x[1],
                                                                                 y = self.planned_traj.y[1],
                                                                                 z = self.ego.get_location().z),
                                                       rotation=carla.Rotation(yaw=np.degrees(self.planned_traj.heading[1])
                                                       )
                ))
                self.ego.set_target_velocity(carla.Vector3D(
                                                        x = self.planned_traj.v[1] * np.cos(self.planned_traj.heading[1]),
                                                        y = self.planned_traj.v[1] * np.sin(self.planned_traj.heading[1]),
                                                        z = 0
                ))
                # Previous control inputs, vel, pos, and other state variables updated based on planned trajectory
                self.u_prev = [self.planned_traj.a[1], self.planned_traj.df[1]]
                self.v = self.planned_traj.v[1]
                self.x = self.planned_traj.x[1]
                self.y = self.planned_traj.y[1]
                self.epsi = self.planned_traj.epsi[1]
                self.s = self.planned_traj.s[1]
                self.ey = self.planned_traj.ey[1]
                self.heading = self.planned_traj.heading[1]
                self.heading_prev = self.planned_traj.heading[1]
                self.v_prev = self.v
            # If infeasible...
            else:
                # AND if the number of consecutive infeasible plans is LESS than the planner's horizon...
                if self.infeas_tick < self.planner.planner.N-1:
                    if not self.planned_traj_exists:
                        return
                    print("[navigation_agents.py, control()] self.infeas_tick: ", self.infeas_tick)
                    print("[navigation_agents.py, control()] Optimization Problem Infeasible. Setting ego to ", self.planned_traj.x[1 + self.infeas_tick], self.planned_traj.y[1 + self.infeas_tick])
                    # Ego vehicle's position, heading, and velocity are updated to follow the previously planned trajectory
                    self.ego.set_transform(carla.Transform(
                                                           location = carla.Location(x = self.planned_traj[1 + self.infeas_tick],
                                                                                     y = self.planned_traj[1 + self.infeas_tick],
                                                                                     z = self.ego.get_location().z),
                                                           rotation = carla.Rotation(yaw = np.degrees(self.planned_traj.heading[1 + self.infeas_tick]))
                    ))
                    self.ego.set_target_velocity(carla.Vector3D(
                                                            x = self.planned_traj.v[1] * np.cos(self.planned_traj.heading[1 + self.infeas_tick]),
                                                            y = self.planned_traj.v[1] * np.sin(self.planned_traj.heading[1 + self.infeas_tick]),
                                                            z = 0
                    ))
                    # Previous control inputs and other state variables are updated based on the planned trajectory
                    self.u_prev = [self.planned_traj.a[1 + self.infeas_tick],
                                   self.planned_traj.df[1 + self.infeas_tick]]
                    self.v = self.planned_traj.v[1 + self.infeas_tick]
                    self.x = self.planned_traj.x[1 + self.infeas_tick]
                    self.y = self.planned_traj.y[1 + self.infeas_tick]
                    self.s = self.planned_traj.s[1 + self.infeas_tick]
                    self.ey = self.planned_traj.ey[1 + self.infeas_tick]
                    self.epsi = self.planned_traj.epsi[1 + self.infeas_tick]
                    self.heading = self.planned_traj.heading[1 + self.infeas_tick]
                    self.heading_prev = self.planned_traj.heading_prev[1 + self.infeas_tick]
                    self.v_prev = self.v
                # If the number of consecutive infeasible plans exceeds the planner's horizon...
                else:
                    print("[navigation_agents.py, control()] self.infeas_tick: ", self.infeas_tick)
                    print("[navigation_agents.py, control()] Optimization Problem Infeasible, but tick greater than plan. Applying full break. ", self.infeas_tick)
                    # Ego vehicle's position and heading are set to the last point in the previously planned trajectory
                    # Bring ego vehicle to full stop by setting its velocity to 0
                    self.ego.set_transform(carla.Transform(
                                                           location = carla.Location(x = self.planned_traj[-1],
                                                                                     y = self.planned_traj[-1],
                                                                                     z = self.ego.get_location().z),
                                                           rotation = carla.Rotation(yaw = np.degrees(self.planned_traj.heading[-1]))
                    ))
                    self.ego.set_target_velocity(carla.Vector3D(
                                                            x = 0,
                                                            y = 0,
                                                            z = 0
                    ))
                    # Previous control inputs and state variables are updated accordingly
                    self.u_prev = [0, 0]
                    self.v = self.planned_traj.v[-1]
                    self.x = self.planned_traj.x[-1]
                    self.y = self.planned_traj.y[-1]
                    self.s = self.planned_traj.s[-1]
                    self.ey = self.planned_traj.ey[-1]
                    self.epsi = self.planned_traj.epsi[-1]
                    self.heading = self.planned_traj.heading[-1]
                    self.heading_prev = self.planned_traj.heading_prev[-1]
                    self.v_prev = self.v
        else:
            raise ValueError
    
    def set_destination(self, destination):
        self.destination = destination
    
    def get_planned_route(self, max_waypoints):
        """
        Returns a list of waypoints with the currently planned route
        """
        return self.world_model.planned_route[:max_waypoints]
    
    def done(self, threshold = 1):
        """
        Returns true if distance of agent to last waypoint of route is less than threshold
        """
        end_waypoint = self.route[-1][0]
        end_xy = np.array((end_waypoint.transform.location.x, 
                           end_waypoint.transform.location.y))
        current_xy = np.array((self.ego.get_location().x,
                               self.ego.get_location().y))
        diff = end_xy - current_xy
        dist = np.linalg.norm(diff)
        if dist < threshold: 
            return True
        return False
    
    def is_feasible(self):
        return self.opt_flag