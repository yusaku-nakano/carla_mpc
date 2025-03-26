import numpy as np
from collections import deque 
import queue
import math

class CUtils(object):
    def __init__(self):
        pass

    def create_var(self, var_name, value):
        if not var_name in self.__dict__:
            self.__dict__[var_name] = value

class VehicleController:
    def __init__(self,config):
        self.config = config
        # self.vars                = CUtils()
        #Define variables
        self.accumulated_speed_error = 0
        self.previous_speed_error = 0
        ## Longitudinal controller parameters
        self._longitudinal_controller = config['planner']['longitudinal_control']
        #PID
        self._kP_a = 1.1
        self._kI_a = 0.35
        self._kD_a = 0.27
        self._err_hist_a = queue.Queue(500)

        # ALC
        self._k_t                = 0.5 # Agressiveness coefficient
        self._vel_limit          = 69.4444 # Speed limit (250 kmph)
        self._steer_limit        = 1.22 # Steering limit (70 deg)

        ## LATERAL CONTROLLER PARAMETERS
        self._lateral_controller = config['planner']['lateral_control']
        self._cte_ref_dist       = 1.5 # Distance from vehicle centre to front axle (m) (2 for Bang-Bang, 1.5 for PID and Stanley)
        self._eps_lookahead      = 10e-3 # Epsilon distance approximation threshold (m)
        self._closest_distance   = 0 # Distance to closest waypoint (m)
        # PID
        self._kP_s               = 0.12
        self._kI_s               = 0.01
        self._kD_s               = 0.48
        self._err_hist_s         = queue.Queue(500) # Limited error buffer
        # Pure-Pursuit
        self._Kpp                = 0.9
        self._min_lookahead_dis  = 10
        self._wheelbase          = 3.0
        # Stanley
        self._Kcte               = 1.5
        self._Ksoft              = 1e-5
        self._Kvel               = 1.3
        # POP
        self._min_lookahead_dist = 6
        self._Kv                 = 0.2
        self._steering_list      = np.linspace(-3, 3, 21, endpoint = True)

    def calculate_steering(self, t, x, y, yaw, waypoints, v):
        if  self._lateral_controller == 'BangBang':
            # time_stamp=time.time()
            crosstrack_error = self.get_crosstrack_error(x, y, waypoints)
            if crosstrack_error > 0:
                steering = 1.22*0.1
            elif crosstrack_error < 0:
                steering = -1.22*0.1
            else:
                steering = 0
            # self._latency = (time.time()-time_stamp)*1000
            #print('Latency: ' + str(self._latency) + ' ms')
            return steering
        elif  self._lateral_controller == 'PID':
            # time_stamp=time.time()
            time_step = self.config['planner']['dt']
            current_crosstrack_error = self.get_crosstrack_error(x, y, waypoints)
            self._err_hist_s.put(current_crosstrack_error)
            accumulated_crosstrack_error = (self.accumulated_crosstrack_error + current_crosstrack_error)
            if self._err_hist_s.full():
                accumulated_crosstrack_error -= self._err_hist_s.get()
            crosstrack_error_change = (current_crosstrack_error - self.previous_crosstrack_error)
            p_term = self._kP_s * current_crosstrack_error
            i_term = self._kI_s * accumulated_crosstrack_error * time_step
            d_term = self._kD_s * crosstrack_error_change / time_step
            steering = p_term + i_term + d_term
            self.accumulated_crosstrack_error = accumulated_crosstrack_error
            self.previous_crosstrack_error = current_crosstrack_error
            # self._latency = (time.time()-time_stamp)*1000
            #print('Latency: ' + str(self._latency) + ' ms')
            return steering
        elif self._lateral_controller == 'PurePursuit':
            # time_stamp=time.time()
            lookahead_dis = self.get_lookahead_dis(v)
            idx = self.get_lookahead_point_index(x, y, waypoints, lookahead_dis)
            v1 = [waypoints[idx][0] - x, waypoints[idx][1] - y]
            v2 = [np.cos(yaw), np.sin(yaw)]
            alpha = self.get_alpha(v1, v2, lookahead_dis)
            if math.isnan(alpha):
                alpha = self.vars.alpha_previous
            if not math.isnan(alpha):
                self.vars.alpha_previous = alpha
            steering = self.get_steering_direction(v1, v2)*np.arctan((2*self._wheelbase*np.sin(alpha))/(lookahead_dis)) # Pure pursuit law
            if math.isnan(steering):
                steering = self.vars.steering_previous
            if not math.isnan(steering):
                self.vars.steering_previous = steering
            # self._latency = (time.time()-time_stamp)*1000
            #print('Latency: ' + str(self._latency) + ' ms')
            return steering
        elif self._lateral_controller == 'Stanley':
            # time_stamp=time.time()
            v1 = [waypoints[0][0] - x, waypoints[0][1] - y]
            v2 = [np.cos(yaw), np.sin(yaw)]
            heading_error = self.get_heading_error(waypoints, yaw)
            cte_term = self._Kcte * self.get_crosstrack_error(x, y, waypoints)
            cte_term = np.arctan(cte_term/(self._Ksoft+self._Kvel*v))
            cte_term = divmod(cte_term, np.pi)[1]
            if cte_term > np.pi/2 and cte_term < np.pi:
                cte_term -= np.pi
            steering =  (heading_error + cte_term) # Stanley control law
            # self._latency = (time.time()-time_stamp)*1000
            #print('Latency: ' + str(self._latency) + ' ms')
            return steering
        elif self._lateral_controller == 'POP':
            # time_stamp=time.time()
            steering_list = self.vars.steering_previous + self._steering_list * self._pi/180 # List of steering angles in the neighbourhood (left and right) of previous steering angle
            lookahead_dist = self._min_lookahead_dist + self._Kv*v
            lp_idx = self.get_lookahead_point_index(x, y, waypoints, lookahead_dist)
            lookahead_point = [waypoints[lp_idx][0], waypoints[lp_idx][1]] # Select a lookahead point from the dynamically updated list of proximal waypoints
            min_dist = float("inf") # Initialize minimum distance value to infinity
            steering = self.vars.steering_previous # Set steering angle to previous value if the following optimization problem yields no acceptable solution (sanity check)
            for i in range(len(steering_list)):
                predicted_vehicle_location = self.get_predicted_vehicle_location(x, y, steering_list[i], yaw, v) # Get predicted vehicle location based on its current state and control input (i-th steering angle from the list)
                dist_to_lookahead_point = self.get_distance(predicted_vehicle_location[0], predicted_vehicle_location[1], lookahead_point[0], lookahead_point[1]) # Compute the distance between predicted vehicle location and lookahead point
                if dist_to_lookahead_point < min_dist: # Optimization problem (Minimize distance between predicted vehicle location and lookahead point to ensure effective path-tracking)
                    steering = steering_list[i] # Select the steering angle that minimizes distance between predicted vehicle location and lookahead point
                    min_dist = dist_to_lookahead_point # Update the minimum distance value
            self.vars.steering_previous = steering # Update previous steering angle value
            # self._latency = (time.time()-time_stamp)*1000
            #print('Latency: ' + str(self._latency) + ' ms')
            return steering
        else:
            return 0
        
    def calculate_acceleration(self, v, v_desired, steer_output):
        if self._longitudinal_controller == 'PID':
            time_step = self.config['planner']['dt']
            current_speed_error = v_desired - v
            self._err_hist_a.put(current_speed_error)
            accumulated_speed_error = (self.accumulated_speed_error + current_speed_error)
            if self._err_hist_a.full():
                accumulated_speed_error -= self._err_hist_a.get()
            speed_error_change = (current_speed_error - self.previous_speed_error)
            p_term = self._kP_a * current_speed_error
            i_term = self._kI_a * accumulated_speed_error * time_step
            d_term = self._kD_a * speed_error_change / time_step
            acceleration = p_term + i_term + d_term
            self.accumulated_speed_error = accumulated_speed_error
            self.previous_speed_error = current_speed_error
            return acceleration
        elif self._longitudinal_controller == 'ALC':
            return self._k_t+((((self.config['planner']['v_max'] - v)/self.config['planner']['v_max']) - (abs(steer_output)/self.config['planner']['max_steering']))*(1-self._k_t))
        else:
            return 0