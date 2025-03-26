import casadi as ca
import numpy as np
import yaml
import pdb
import time
from typing import List

from kinematic_bicycle_model_frenet import KinematicBicycleModelFrenet
from utils import VehicleAction, VehicleReference, ReferenceTrajectory, PlannedTrajectory

class MPCSolver():
    def __init__(self, planner_config: dict, obstacles: List = None, ref: ReferenceTrajectory = None):
        # MPC Parameters
        self.N = planner_config['N'] # Time horizon
        self.dt = planner_config['dt']
        self.d_min = 2 * planner_config['ca_radius']
        self.ref = ref
        self.prev_sol = None
        self.nx = planner_config['nx']
        self.nu = planner_config['nu']

        # Vehicle Parameters
        self.L = planner_config['L']
        self.L_f = self.L/2
        self.L_r = self.L/2
        self.width = planner_config['w']
        self.steering_rate_limit = planner_config['steering_rate_limit'] # rad/s
        self.jerk_limit = planner_config['jerk_limit'] # m/s^3
        self.v_min = planner_config['v_min']
        self.v_max = planner_config['v_max']
        self.v_des = planner_config['v_des']
        self.a_min = planner_config['a_min']
        self.a_max = planner_config['a_max']
        self.ey_lim = planner_config['ey_lim']
        self.max_steering = planner_config['max_steering']

        # Initialize Dynamics Model
        self.model = KinematicBicycleModelFrenet(self.L_r, self.L_f, self.width, self.dt, discretization='euler', mode='casadi', num_rk4_steps=1)

        # Initialize Obstacles
        self.num_obstacles = len(obstacles) if obstacles else 5

        # Initialize the Optimization Problem
        self.add_decision_variables()
        self.add_initial_constraints()
        self.add_input_rate_constraints()
        self.add_ey_constraints()
        self.add_state_and_input_constraints()
        self.add_dynamic_constraints()
        # if self.num_obstacles > 0:
        #     self.add_collision_avoidance_constraints()

        nlp = {'x': ca.vertcat(ca.vec(self.x), ca.vec(self.u)), 
               'p': self.params,
               'f': self.cost_function(),
               'g': self.constraints}

        plugin_opts = {'expand': True,
                       'verbose': False,
                       'verbose_init': False,
                       'print_time': False,
                       'ipopt': {"print_level": 0,
                                 'acceptable_tol': 1e-4,
                                 'acceptable_obj_change_tol': 1e-4,
                                 'warm_start_init_point': 'yes',
                                 'warm_start_bound_push': 1e-9,
                                 'warm_start_bound_frac': 1e-9,
                                 'warm_start_slack_bound_frac': 1e-9,
                                 'warm_start_slack_bound_push': 1e-9,
                                 'warm_start_mult_bound_push': 1e-9,
                                 "max_iter": self.N*200,
                                 "max_wall_time": self.N}}
        self.S = ca.nlpsol('S', 'ipopt', nlp, plugin_opts)
        self.S.generate('mpc_solve.c')
        self.S.generate_dependencies('mpc_solve.c')

    
    def add_decision_variables(self):
        # Create state vector x, where each column represents state vector at specific timestep
        # x = [x, y, s, ey, epsi, v]
        self.x = ca.MX.sym('x', self.nx, self.N+1) 
        # Create input vector u, where each column represents input vector at specific timestep
        # u = [delta, a] ??
        self.u = ca.MX.sym('u', self.nu, self.N)
        # Create parameter vector for initial state and previous input
        self.params = ca.vertcat(ca.MX.sym('x0', self.nx, 1), ca.MX.sym('u_prev', self.nu, 1))
    
    def add_dynamic_constraints(self):
        """
        Enforces the vehicle's dynamic constraints across the prediction horizon

        Ensures that the vehicle's state evolution follows the physical dynamics
        governed by the kinematic bicycle model in the Frenet frame
        
        Constraint Formation:
        Creates constraints ensuring that the next state variables in the optimization
        match those predicted by the dynamics model
        Constraints take the form: x[:,k+1] - state_kp1 == 0
        Constrains 7 state variables:
        - x position
        - y position
        - path progress (s)
        - lateral error (ey)
        - heading error (epsi)
        - velocity (v)
        - heading angle
        """
        for k in range(self.N):
            state_kp1 = self.model(VehicleReference({'s': self.x[2,k],
                                                    'ey': self.x[3,k],
                                                    'epsi': self.x[4,k],
                                                    'v': self.x[5,k],
                                                    'K': self.ref.K,
                                                    'x': self.x[0,k],
                                                    'y': self.x[1,k],
                                                    'heading': self.x[6,k]}),
                                    VehicleAction({'a': self.u[0,k],
                                                    'df': self.u[1,k]}))
            if k==0:
                dynamic_constraints = ca.vertcat(self.x[0,k+1] - state_kp1.x,
                                                 self.x[1,k+1] - state_kp1.y,
                                                 self.x[2,k+1] - state_kp1.s,
                                                 self.x[3,k+1] - state_kp1.ey,
                                                 self.x[4,k+1] - state_kp1.epsi,
                                                 self.x[5,k+1] - state_kp1.v,
                                                 self.x[6,k+1] - state_kp1.heading)
            else:
                dynamic_constraints = ca.vertcat(dynamic_constraints,
                                                self.x[0,k+1] - state_kp1.x,
                                                self.x[1,k+1] - state_kp1.y,
                                                self.x[2,k+1] - state_kp1.s,
                                                self.x[3,k+1] - state_kp1.ey,
                                                self.x[4,k+1] - state_kp1.epsi,
                                                self.x[5,k+1] - state_kp1.v,
                                                self.x[6,k+1] - state_kp1.heading)
        self.constraints = ca.vertcat(self.constraints, dynamic_constraints)
        self.constraints_lower_bounds = ca.vertcat(self.constraints_lower_bounds, np.zeros(dynamic_constraints.shape))
        self.constraints_upper_bounds = ca.vertcat(self.constraints_upper_bounds, np.zeros(dynamic_constraints.shape))
    
    def add_collision_avoidance_constraints(self):
        """
        Creates safety constraints that prevent the vehicle from colliding with obstacles throughout its 
        planned trajectory
        Implements a basic collision avoidance mechanism using a circular representation for both the
        vehicle and obstacles

        Mathematical Formulation:
        - Both the vehicle and obstacles are represented as circles
        - A collision is avoided when the distance between the centers exceeds the sum of radii
        - Constraint uses squared distances to avoid computationally expensive square root operations
        """
        # Iterate through every timestep in the prediction horizon (plus the final state)
        for k in range(self.N+1):
            # For each timestep, iterate through every obstacle
            for i in range(self.num_obstacles):
                # Initialize with the first constraint and vertically concatenate all subsequent constraints
                if k == 0 and i == 0:
                    # self.params[...]: positions of all obstacles at all prediction timesteps
                    # self.x[0:2,k] - self.params[...]: vector from obstacle to the vehicle
                    # ca.bilin(ca.DM.eye(2), vector_diff, vector_diff): 
                    #   - ca.DM.eye(2): 2x2 identity matrix
                    #   - vector_diff: vector from obstacle to the vehicle
                    #   - with the identity matrix as the middle term, this calculates v^T * I * v
                    #   - Equivalent to v^T * v = v[0]^2 + v[1]^2 (the squared Euclidean norm)
                    #   - Gives the squared distance between vehicle and obstacle centers
                    # self.d_min**2 - squared_distance: ensures minimum safety distance is maintained between vehicle and obstacle
                    collision_avoidance_constraints = self.d_min**2 - ca.bilin(ca.DM.eye(2), self.x[0:2,k] - self.params[self.nx+self.nu+self.num_obstacles*2*k+2*i: self.nx+self.nu+self.num_obstacles*2*k+2*(i+1)],
                                                                                             self.x[0:2,k] - self.params[self.nx+self.nu+self.num_obstacles*2*k+2*i: self.nx+self.nu+self.num_obstacles*2*k+2*(i+1)])
                else:
                    collision_avoidance_constraints = ca.vertcat(collision_avoidance_constraints,
                                                                self.d_min**2 - ca.bilin(ca.DM.eye(2), self.x[0:2,k] - self.params[self.nx+self.nu+self.num_obstacles*2*k+2*i: self.nx+self.nu+self.num_obstacles*2*k+2*(i+1)],
                                                                                                       self.x[0:2,k] - self.params[self.nx+self.nu+self.num_obstacles*2*k+2*i: self.nx+self.nu+self.num_obstacles*2*k+2*(i+1)]))
        self.constraints = ca.vertcat(self.constraints, collision_avoidance_constraints)
        # Lower bound = 0: ensures that d_min^2 - distance^2 >= 0
        self.constraints_upper_bounds = ca.vertcat(self.constraints_upper_bounds, np.inf * np.ones(collision_avoidance_constraints.shape))
        # Upper bound = inf: places no upper limit on this value
        self.constraints_lower_bounds = ca.vertcat(self.constraints_lower_bounds, np.zeros(collision_avoidance_constraints.shape))
        # Together, these enforce: distance^2 >= d_min^2
    
    def add_initial_constraints(self):
        """
        Enforces that the first state in the planned trajectory exactly matches the current state
        of the vehicle
        Anchors the optimization to start from the vehicle's actual position

        Mathematical Formulation:
        - For each state variable index i (from 0 ~ nx-1), the method enforces:
        x[i,0] - x0[i] = 0
        or simply,
        x[i,0] = x0[i]
        """
        # The first self.nx elements in self.params represent the current state of the vehicle
        x0 = self.params[:self.nx]
        # self.x[:,0] is the first state in the planned trajectory
        # x0 is the current vehicle state
        # This creates a vector of differences that must equal zero
        initial_constraints = ca.vertcat(self.x[:,0] - x0)
        # Creates a vector of zeros with the same shape as the constraint vector
        initial_bounds = np.zeros(initial_constraints.shape)

        self.constraints = initial_constraints
        self.constraints_upper_bounds = initial_bounds
        self.constraints_lower_bounds = initial_bounds
    
    def add_ey_constraints(self):
        """
        Implements lateral position constraints to keep the vehicle within acceptable boundaries
        of the reference path
        Limits how far the vehicle can deviate laterally from the reference path
        """
        # Loop through prediction horizon
        for k in range(self.N+1):
            if k == 0:
                # First constraint:  ey <= ey_lim
                # Second constraint: ey >= -ey_lim
                ey_constraints = ca.vertcat(self.ey_lim - self.x[3,k],
                                            self.ey_lim + self.x[3,k])
            else:
                ey_constraints = ca.vertcat(ey_constraints,
                                            self.ey_lim - self.x[3,k],
                                            self.ey_lim + self.x[3,k])
        self.constraints = ca.vertcat(self.constraints, ey_constraints)
        self.constraints_upper_bounds = ca.vertcat(self.constraints_upper_bounds, np.inf*np.ones(ey_constraints.shape))
        # First constraint:   ey_lim - ey >= 0    ->    ey <=  ey_lim
        # Second constraint: -ey_lim + ey >= 0    ->    ey >= -ey_lim
        self.constraints_lower_bounds = ca.vertcat(self.constraints_lower_bounds, np.zeros(ey_constraints.shape))

    def add_input_rate_constraints(self):
        """
        Enforces constraints on two control inputs:
        - Acceleration rate changes (jerk limits)
        - Steering angle rate changes (steering rate limits)

        Prevents controller from generating solutions with abrupt changes that would be:
        - Uncomfortable for passengers
        - Physically impossible for vehicle to achieve

        Mathematical Formulation:
        Jerk limit:
        -jerk_lim <= (a[k] - a[k-1])/dt <= jerk_lim
        Steering rate limit:
        -steer_rate_lim <= (df[k] - df[k-1])/dt <= steer_rate_lim
        """
        u_prev_a = self.params[self.nx]
        u_prev_df = self.params[self.nx+1]
        input_rate_constraints = ca.vertcat(self.u[0,0] - u_prev_a + self.dt*self.jerk_limit,
                                           -self.u[0,0] + u_prev_a + self.dt*self.jerk_limit,
                                            self.u[1,0] - u_prev_df + self.dt*self.steering_rate_limit,
                                           -self.u[1,0] + u_prev_df + self.dt*self.steering_rate_limit)
        self.constraints = ca.vertcat(self.constraints, input_rate_constraints)
        self.constraints_upper_bounds = ca.vertcat(self.constraints_upper_bounds, np.inf*np.ones(input_rate_constraints.shape))
        self.constraints_lower_bounds = ca.vertcat(self.constraints_lower_bounds, np.zeros(input_rate_constraints.shape))

        for k in range(self.N):
            if k == 0:
                input_rate_constraints = ca.vertcat(self.u[0,k] - self.u[0,k-1] + self.dt*self.jerk_limit,
                                                   -self.u[0,k] + self.u[0,k-1] + self.dt*self.jerk_limit,
                                                    self.u[1,k] - self.u[1,k-1] + self.dt*self.steering_rate_limit,
                                                   -self.u[1,k] + self.u[1,k-1] + self.dt*self.steering_rate_limit)
            else:
                input_rate_constraints = ca.vertcat(input_rate_constraints,
                                                    self.u[0,k] - self.u[0,k-1] + self.dt*self.jerk_limit,
                                                   -self.u[0,k] + self.u[0,k-1] + self.dt*self.jerk_limit,
                                                    self.u[1,k] - self.u[1,k-1] + self.dt*self.steering_rate_limit,
                                                   -self.u[1,k] + self.u[1,k-1] + self.dt*self.steering_rate_limit)
        self.constraints = ca.vertcat(self.constraints, input_rate_constraints)
        self.constraints_upper_bounds = ca.vertcat(self.constraints_upper_bounds, np.inf*np.ones(input_rate_constraints.shape))
        self.constraints_lower_bounds = ca.vertcat(self.constraints_lower_bounds, np.zeros(input_rate_constraints.shape))
    
    def add_state_and_input_constraints(self):
        """
        Enforces constraints on:
        - Vehicle state variables (velocity)
        - Control input variables (acceleration, steering angle)

        Mathematical Formulation:
        Velocity constraints
        v_min <= x[5,k] <= v_max
        Acceleration constraints
        a_min <= u[0,k] <= a_max
        Steering angle constraints
        -max_steering <= u[1,k] <= max_steering
        """
        for k in range(self.N):
            if k == 0:
                state_input_constraints = ca.vertcat(self.x[5,k] - self.v_min,
                                                    -self.x[5,k] + self.v_max,
                                                     self.u[0,k] - self.a_min,
                                                    -self.u[0,k] + self.a_max,
                                                     self.u[1,k] + self.max_steering,
                                                    -self.u[1,k] + self.max_steering)
            else:
                state_input_constraints = ca.vertcat(state_input_constraints,
                                                     self.x[5,k] - self.v_min,
                                                    -self.x[5,k] + self.v_max,
                                                     self.u[0,k] - self.a_min,
                                                    -self.u[0,k] + self.a_max,
                                                     self.u[1,k] + self.max_steering,
                                                    -self.u[1,k] + self.max_steering)
        self.constraints = ca.vertcat(self.constraints, state_input_constraints)
        self.constraints_upper_bounds = ca.vertcat(self.constraints_upper_bounds, np.inf*np.ones(state_input_constraints.shape))
        self.constraints_lower_bounds = ca.vertcat(self.constraints_lower_bounds, np.zeros(state_input_constraints.shape))
    
    def cost_function(self):
        """
        Returns Objective Function

        Mathematical Formulation:
        -s_N + sum_0_N((v_k-v_des)^2 + 10(epsi^2 + ey^2)) + sum_0_Nm1(0.1(a_k^2 + df^2))
        
        Maximize progress along trajectory (s)
        - Weight: 1
        Minimize velocity deviation (v)
        - Weight: 1
        Minimize lateral deviation and heading error (epsi, ey)
        - Weight: 10
        Minimize control effort (a, df)
        - Weight: 0.1
        """
        sum_cost = 0
        for k in range(self.N+1):
            # Minimize v (velocity deviation)
            sum_cost += (self.x[5,k] - self.v_des)**2
            if k < self.N:
                # Minimize a, df (control effort)
                sum_cost += 0.1 * (self.u[0,k]**2 + self.u[1,k]**2)
            # Minimize epsi, ey (lateral deviation and heading error)
            sum_cost += 10 * (self.x[4,k]**2 + self.x[3,k]**2)
        # Maximize s (progress along trajectory)
        sum_cost += -self.x[2,self.N]
        return sum_cost
    
    def solve(self, curr_state=None, u_prev=None, obs_preds=None, v_des=None):
        """
        Solves the optimization problem defined in self.S

        Background on self.S: self.S is an nlpsol object created using CasADi's nonlinear programming interface
        - x0: initial guess for decision variables
        - p: parameters
        - lbg, ubg: constraint bounds


        """
        # Debug logging of current state
        print(f'\n s: {curr_state.s}, v: {curr_state.v}')

        # State Initialization: initializes the state trajectory with the current state
        self.x_init = np.hstack([np.array([[curr_state.x],
                                           [curr_state.y],
                                           [curr_state.s],
                                           [curr_state.ey],
                                           [curr_state.epsi],
                                           [curr_state.v],
                                           [curr_state.heading]]) for _ in range(self.x.shape[1])])
        
        # Parameter setup
        # Parameters for the solver include initial state (x_init[:,0]) and previous control inputs (u_prev.a, u_prev.df)
        print("obs_preds: ", obs_preds)
        if obs_preds is None or np.all(obs_preds == None):
            params = ca.vertcat(self.x_init[:,0], u_prev.a, u_prev.df)
        else:
            # If collision avoidance is enabled
            params = ca.vertcat(self.x_init[:,0], u_prev.a, u_prev.df, ca.vec(obs_preds))
        
        # Update desired velocity
        if v_des:
            self.v_des = v_des
        
        # Warm start: initializing an optimization solver with a guess that's closer to the optimal solution
        # rather than starting from scratch (cold start)

        # If a previous solution exists, use the previous solution as a starting point
        if self.prev_sol:
            self.sol = self.S(x0=self.prev_sol['x'], p=params, lbg=self.constraints_lower_bounds, ubg=self.constraints_upper_bounds)
        # Otherwise, start from default initial values for states and controls
        else:
            self.sol = self.S(x0=ca.vertcat(ca.DM(ca.vec(self.x_init)), ca.DM.zeros(self.u.shape[0] * self.u.shape[1])),
                              p=params, lbg=self.constraints_lower_bounds, ubg=self.constraints_upper_bounds)
        
        # Save the current solution for future warm starts
        self.prev_sol = self.sol

        x_opt = self.sol['x']
        J_opt = self.sol['f']

        # Extract solver stats and check for success
        self.solver_stats = self.S.stats()

        solver_flag = self.solver_stats['success']
        if not solver_flag:
            print(self.solver_stats['return_status'])
        
        # State reconstruction: rebuild the state trajectory from the solution vector x_opt
        state = np.array(x_opt[:self.nx])
        for i in range(1, self.N+1):
            state = np.hstack((state, np.array(x_opt[self.nx * i:self.nx * (i+1)])))
        
        control_start_ind = self.nx * (self.N+1)

        # Control input reconstruction: reconstruct the control input sequence for the entire horizon
        for i in range(self.N):
            if i < 1:
                control = np.array(x_opt[control_start_ind:control_start_ind+self.nu])
            else:
                control = np.hstack((control, np.array(x_opt[control_start_ind + self.nu * i:control_start_ind + self.nu * (i+1)])))
        
        # Return results:
        # - state: optimal state trajectory
        # - control: optimal control inputs
        # - J_opt: optimal cost
        # - solver_flag: whether the solver succeeded
        return (state, control, J_opt, solver_flag)