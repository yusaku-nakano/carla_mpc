import casadi as ca
import numpy as np
import yaml
import pdb
import time
from typing import List

from kinematic_bicycle_model_frenet import KinematicBicycleModelFrenet
from utils import VehicleAction, VehicleReference, ReferenceTrajectory, PlannedTrajectory
from MPCSolver_old import MPCSolver

class BasePlanner:
    def __init__(self, planner_config: dict):
        self.name = planner_config['model']
        self.planner_config = planner_config
    def check_inputs(self, args: dict):
        raise NotImplementedError
    def plan(self, args: dict) -> List[PlannedTrajectory]:
        raise NotImplementedError

class MPCPlanner(BasePlanner):
    def __init__(self, planner_config: dict, ref: dict, x0: np.ndarray):
        super().__init__(planner_config)
        self.planner_config = planner_config
        self.planner = MPCSolver(planner_config, obstacles=None, ref=ref)
        # self.planner = MPCSolver(planner_config, obstacles=None, ref=ref, x0=x0)
        self.planned_traj = PlannedTrajectory(timesteps=self.planner_config['N'], n_dims=(self.planner_config['nx'], self.planner_config['nu']))
    
    def check_inputs(self, args):
        if "actor_state" not in args:
            raise ValueError("actor_state is required")
        if "prev_action" not in args:
            raise ValueError("prev_action is required")
        if "ref" not in args:
            raise ValueError("ref is required")
        if "obstacle_preds" not in args:
            raise ValueError("obstacle_preds is required")
    
    def get_filtered_obstacles_preds(self, ego_pos, obstacle_preds, radius=50, N=10):
        """
        Get the obstacle predictions in the format required by the planner within a 
        predefined radius of the ego vehicle's initial position
        Sorted in the order of closest to the ego vehicle

        Inputs:
        - ego_pos: 2D numpy array representing ego position [x,y]
        - obstacle_preds: dictionary where each key corresponds to an obtacle ID
            and the value is a list of predicted trajectories for that obstacle
        - radius: maximum distance from the ego vehicle within which obstacles will be considered
        - N: time horizon for obstacle predictions

        Output:
        - A numpy array of filtered obstacle predictions in a format compatible with the planner
        """
        dist = {}
        # M: large number to fill in the empty slots for dummy obstacles really far away from ego
        # Ensures the planner always receives a fixed-sized array
        M = max(ego_pos) + 10000 
        # preds_arr: 2D numpy array to hold filtered predictions
        preds_arr = M*np.ones((2*self.planner_config['num_obstacles']*self.planner_config['num_modes_per_vehicle'], N+1))

        # If the filtering mode is "distance"
        if self.planner_config['obstacle_filtering'] == 'distance':
            # For each obstacle in obstacle_preds...
            for _, (ind, pred) in enumerate(obstacle_preds.items()):
                ind = int(ind)
                # Euclidean distance from ego vehicle to obstacle's predicted position
                distance = np.linalg.norm(ego_pos - np.array([pred[0].x[0], pred[0].y[0]]))
                # Only include obstacles with distances between 2*ca_radius (safety buffer) and radius
                if 2*self.planner_config['ca_radius'] < distance < radius:
                    dist.update({str(ind): distance})
            # Sort the filtered obstacles by distance in ascending order
            sorted_dist = dict(sorted(dist.items(), key=lambda x: x[1]))
            # Loop through sorted obstacles
            m = 0
            for key in sorted_dist.keys():
                # Extract predictions for the first mode of the obstacle (pred[0])
                # and then combine the x,y predictions into a 2-row matrix
                # "mode" refers to one of the posttible predicted behaviors/trajectories of the obstacle
                preds_arr[2*m:2*(m+1), :] = np.vstack([obstacle_preds[int(key)][0].x[:N+1][np.newaxis,:],
                                                       obstacle_preds[int(key)][0].y[:N+1][np.newaxis,:]])
                m += 1
                # Stop if the array is filled to its capacity
                if m == self.planner_config['num_obstacles']*self.planner_config['num_modes_per_vehicle']:
                    break
        elif self.planner_config['obstacle_filtering'] == 'probability':
            # Iterate through each obstacle in the predictions dictionary
            for _, (ind, pred) in enumerate(obstacle_preds.items()):
                # Verify that we have enough predicted modes for each vehicle
                assert self.planner_config['num_modes_per_vehicle'] <= len(pred)
                ind = int(ind)
                # Calculate Euclidean distance from ego to obstacle's initial position
                distance = np.linalg.norm(ego_pos - np.array([pred[0].x[0], pred[0].y[0]]))
                # Only consider obstacles within specified distance range
                if 2*self.planner_config['ca_radius'] < distance < radius:
                    # Add to distance dictionary
                    dist.update({str(ind): distance})
            # Sort obstacles by distance
            sorted_dist = dict(sorted(dist.items(), key=lambda x: x[1]))
            m = 0 # Counter for filling prediction array
            # Process each obstacle in order of distance
            for key in sorted_dist.keys():
                # For each obstacle, consider multiple predicted trajectories (modes)
                for mode_ind in range(self.planner_config['num_modes_per_vehicle']):
                    obstacle_preds[int(key)]
                    mode_ind = 0

                    # Fill prediction array with xy-coordinates
                    # Remember: preds_arr is arranged as:
                    # EXAMPLE:
                    # preds_arr = np.array([
                    #     # Obstacle 1, Mode 1
                    #     [10.0,  11.2,  12.5,  13.8],  # x-coordinates over time
                    #     [5.0,   5.1,   5.3,   5.6],   # y-coordinates over time
                        
                    #     # Obstacle 1, Mode 2 (e.g., turning behavior)
                    #     [10.0,  10.8,  11.2,  11.5],  # x-coordinates over time
                    #     [5.0,   5.5,   6.2,   7.0],   # y-coordinates over time
                        
                    #     # Obstacle 2, Mode 1
                    #     [15.0,  14.8,  14.5,  14.1],  # x-coordinates over time
                    #     [8.0,   8.2,   8.5,   8.9],   # y-coordinates over time
                        
                    #     # Obstacle 2, Mode 2 (e.g., stopping behavior)
                    #     [15.0,  14.9,  14.8,  14.8],  # x-coordinates over time
                    #     [8.0,   8.1,   8.1,   8.1]    # y-coordinates over time
                    # ])
                    # Extract x,y coordinates for the current obstacle and mode
                    preds_arr[2*m:2*(m+1), :] = np.vstack([obstacle_preds[int(key)][mode_ind].x[:N+1][np.newaxis,:],
                                                           obstacle_preds[int(key)][mode_ind].y[:N+1][np.newaxis,:]])
                    m += 1
                    # Stop if we've filled all available slots
                    if m == self.planner_config['num_obstacles']*self.planner_config['num_modes_per_vehicle']:
                        break
        else:
            raise ValueError('Invalid obstacle_filtering in config')
        print("\n[filtered predictions] Sorted dist:", sorted_dist)
        return preds_arr
    
    def plan(self, args: dict) -> List[PlannedTrajectory]:
        self.check_inputs(args)
        self.obs_preds = self.get_filtered_obstacles_preds(ego_pos = np.array([args['actor_state'].x, args['actor_state'].y]),
                                                           obstacle_preds = args['obstacle_preds'],
                                                           radius = self.planner_config['detection_radius'],
                                                           N = self.planner_config['N'])
        # Solve an optimization problem
        try:
            t_start = time.time()
            x_sol, u_sol, J_opt, is_opt = self.planner.solve(curr_state = args['actor_state'],
                                                             u_prev = args['prev_action'],
                                                             obs_preds = self.obs_preds,
                                                             v_des = self.planner_config['v_max'])
            self.last_cost = J_opt
            self.last_optimization_time = time.time()-t_start
            print("\n Optimization Time: ", self.last_optimization_time)
            print(f'\n J_Opt:  {J_opt}')
            # Set the planned trajectory with the solution
            if is_opt:
                self.planned_traj.set_frenet_traj(x_sol)
                self.planned_traj.set_input_traj(u_sol)
            return self.planned_traj, is_opt
        except:
            is_opt = False
            self.last_optimization_time = np.nan 
            self.last_cost = None
            print("\n Optimization Problem is Infeasible")
            planned_traj = PlannedTrajectory(timesteps = self.planner_config['N'],
                                             n_dims = (self.planner_config['nx'], self.planner_config['nu']))
            planned_traj.set_frenet_traj(x_sol)
            planned_traj.set_input_traj(u_sol)
            return planned_traj, is_opt


