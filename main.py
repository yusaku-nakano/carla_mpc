import carla
import random
from random import choices
import numpy as np
import cv2
import sys
import os
import copy
import pickle
import decouple
import subprocess
import pdb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import argparse

sys.path.append(r'C:\Users\yusak\Downloads\CARLA_0.9.14\WindowsNoEditor\PythonAPI')
sys.path.append(r'C:\Users\yusak\Downloads\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla')

from agents.navigation.global_route_planner import GlobalRoutePlanner

# from MPCAgent import MPCAgent
from MPCAgent import MPCAgent
from world_model import WorldModel
from utils import load_config, connect, set_synchronous_mode, update_fps
from server import spawn_vehicle
from route import generate_route
from log import SimulationLog
from visualization import keep_latest_image, setup_birds_eye_camera, draw_world, draw_points, draw_waypoints

CARLA_PATH = os.path.join(decouple.config("CARLA_DIR"), "PythonAPI/carla")
PROJECT_DIR = decouple.config("PROJECT_DIR")


class Simulation:

    def __init__(self, args):
        self.args = args
        self.config = load_config()

        # Connect to CARLA server
        self.client, self.world = connect(self.config['server'])

        # If anything's remaining from previous run, destroy it
        self.destroy_all_vehicles() 

        # Loads map (Town05_Opt)
        self.check_and_load_map()

        # Addressing delay before car starts accelerating
        # github.com/carla-simulator/carla/issues/6477
        settings = self.world.get_settings()
        settings.substepping = True
        settings.max_substep_delta_time = 0.001
        settings.max_substeps = 100
        settings.fixed_delta_seconds = 1.0/self.config['server']['fps']
        self.world.apply_settings(settings)

        # Initialize world model
        self.world_model = WorldModel(self.world)

        # Initialize spectator
        self.spectator = self.world.get_spectator()

        # Set synchronous mode
        # When synchronous mode is enabled, the simulation is paused until the next tick
        # Makes sure simulation is synced with other processes (e.g. collecting sensor data, performing computations, etc.)
        self.settings = set_synchronous_mode(self.world, self.config['server'], return_settings=True)

        # For FPS Calculation and logging
        self.frame_count = 0 # frame count for fps calculation
        self.current_frame = 0 # current simulation frame
        self.start_time = time.time() # for fps calculation; updates every iteration
        self.simulation_start_time = time.time() # does NOT update every iteration
        self.fps = 0
        self.last_ego_position = None # keep track whether ego is stuck
        self.stuck_counter = 0
        self.wait_counter = 0

        # SimulationLog stores how simulation developed (e.g. actor positions, velocities, etc.)
        self.simulation_log = SimulationLog()

        # Whether to make predictions for the ego vehicle
        self.world_model.set_ego_predictions(self.config["prediction"]["predict_ego"])

        # Initialize route planner
        self.global_route_planner = GlobalRoutePlanner(self.world_model.map_gt, sampling_resolution=self.config["planner"]["dt"]*1) # 1 m/s (low velocity for fine samples of waypoints)

        # Setup spawn point, route, run name
        self.setup_run()

        self.counter = 0 # Frame counter that is never reset

        # Create logging directory if not exist
        self.check_logging_dir()

        # ???
        self.setup_recorder()

        # Set the agent destination
        self.setup_navigation()

        # Spawns vehicle with blueprint as tesla model 3, set ego for world model,
        # Setup MPC Agent, and collision detector
        self.spawn_vehicle_and_agent()

        # Setup birds eye camera
        self.setup_birds_eye_camera(self.config['visualization']['birds_eye_camera'])

        # store carla actor id of ego vehicle and other useful properties like global plan
        self.log_ego_attrs() 

        # set config of SimulationLog (???)
        self.log_config()

        if self.config["visualization"]["debug"]["live_ego_plots"]:
            self._setup_ego_live_plots()
        
    def _setup_ego_live_plots(self):  
        self.time_data = []
        self.vel_data = []
        self.p_vel_data = []
        self.h_data = []
        self.p_h_data = []
        self.acc_data = []
        self.p_acc_data = []
        self.throttle_data = []
        self.brake_data = []
        self.opt_flag_data = []

        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 8))

        # Configure velocity plot
        self.ax1.set_title("Ego Vehicle Velocity")
        self.ax1.set_ylabel("Velocity magnitude (m/s)")
        self.vel_line, = self.ax1.plot([], [], label="Velocity", color="r")
        self.p_vel_line, = self.ax1.plot([], [], label="Planned Velocity", color="g")
        self.vlimu_line, = self.ax1.plot([], [], label="Velocity Limit (max)", color="y")
        self.vliml_line, = self.ax1.plot([], [], label="Velocity Limit (min., neg)", color="y")
        self.ax1.legend()

        # Configure acceleration plot
        self.ax2.set_title("Ego Vehicle Acceleration")
        self.ax2.set_ylabel("Acceleration magnitude (m/s²)")
        self.acc_line, = self.ax2.plot([], [], label="Acceleration", color="r")
        self.p_acc_line, = self.ax2.plot([], [], label="Planned Acceleration", color="g")
        self.alimu_line, = self.ax2.plot([], [], label="Acceleration Limit (max)", color="y")
        self.aliml_line, = self.ax2.plot([], [], label="Acceleration Limit (min., neg)", color="y")
        self.ax2.legend()

        # Configure throttle/brake plot
        self.ax3.set_title("Ego Vehicle Brake/Throttle/opt-flag")
        self.ax3.set_ylabel("Input Magnitude")
        self.throttle_line, = self.ax3.plot([], [], label="Throttle", color="g")
        self.brake_line, = self.ax3.plot([], [], label="Brake", color="r")
        self.opt_flag_line, = self.ax3.plot([], [], label="Opt-flag", color="k")
        self.ax3.legend()

        # Configure heading plot
        self.ax4.set_title("Ego Vehicle Heading")
        self.ax4.set_ylabel("Heading (rad)")
        # self.h_line, = self.ax4.plot([], [], label="Current heading", color="r")
        # self.h_line_traj, = self.ax4.plot([], [], label="planned_traj.heading[0]", color="r")
        self.p_h_line, = self.ax4.plot([], [], label="Planned heading", color="g")
        self.ax4.legend()

        plt.grid()
    
    def log_config(self):
        self.simulation_log.set_config(self.config)
        # if self.args.scenario is not None:
        #     self.simulation_log.set_scenario_config(self.scenario.config)
    
    def check_and_load_map(self):
        # TODO: Implement case for scenario
        current_map = self.world.get_map().name
        target_map = "Town05_Opt"

        # If the current map is not "Town05_Opt", load it
        if target_map not in current_map:
            print(f"Current map is {current_map} and scenarios require Town05_Opt. Loading {target_map}...")
            self.client.load_world(target_map)
            print(f"{target_map} loaded successfully.")
    
    def log_ego_attrs(self):
        self.simulation_log.set_ego_id(self.ego.id)
        self.simulation_log.set_global_plan(self.route)
    
    def setup_run(self):
        # TODO: Implement case for scenario
        self.setup_normal_run()
    
    def setup_normal_run(self):
        spawn_point = random.choice(self.world_model.map_gt.get_spawn_points())
        self.route, self.spawn_point = generate_route(self.global_route_planner, self.world_model.map_gt, spawn_point, spawn_point, 200, ignore_destination=True)
        self.run_name = "last"
    
    def spawn_vehicle_and_agent(self):
        self.ego = spawn_vehicle(self.world, self.spawn_point, ego_vehicle=True, bp='vehicle.tesla.model3')
        self.world_model.set_ego(self.ego)
        self.setup_agent(self.config)
        self.setup_collision_sensor()
    
    def setup_collision_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        self.collision_sensor = self.world.spawn_actor(blueprint_library.find('sensor.other.collision'), carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: self.simulation_log.store_collision(event, self.current_frame))

    def check_logging_dir(self):
        # if logging directoring does not exist, create it
        if not os.path.exists(self.config["logging"]["directory"]):
            os.makedirs(self.config["logging"]["directory"])

    def setup_recorder(self):
        self.client.start_recorder(
            os.path.join(self.config["logging"]["directory"], f"{self.run_name}.log"),
            self.config["logging"]["additional_data"]
        )

    # def setup_birds_eye_camera(self, config):
    #     self.birds_eye_camera = None
    #     self.latest_image = None  # Global variable to store the latest image
    #     if config['active']:
    #         self.birds_eye_camera = setup_birds_eye_camera(self.ego, config)
    #         self.birds_eye_camera.listen(lambda image: keep_latest_image(image, self, 'latest_image'))

    def setup_agent(self, config):
        # self.agent = AVAgent(self.ego, config, world_model=self.world_model)
        ##self.setup_navigation()

        # ToDo: Add MPCAgent instantiation within AVAgent, so CARLA and Custom agents are interfaced uniformly.
        if self.config["agent"]["name"] == "Custom":
            self.agent = MPCAgent(self.ego, config, world_model=self.world_model, route=self.route,
                                  destination=self.ego_destination)

    def setup_navigation(self):
        # Set the agent destination
        self.spawn_points = self.world_model.map_gt.get_spawn_points()
        self.ego_destination = random.choice(self.spawn_points).location

    def update_spectator(self):
        if self.config['misc']['spectator_follows_ego']:
            transform = self.ego.get_transform()
            # transform = self.scenario.others[0].get_transform()
            self.spectator.set_transform(
                carla.Transform(transform.location + carla.Location(z=self.config["misc"]["spectator_z"]),
                                carla.Rotation(pitch=-90)))

    def draw_planned_route(self, max_waypoints=1000):
        # waypoints_list = self.agent.get_planned_route(max_waypoints)
        # draw_waypoints_II(self.world, waypoints_list, life_time=self.config["visualization"]["debug"]["life_time"])
        points_list = self.agent.get_planned_route(max_waypoints)
        draw_points(self.world, points_list, life_time=self.config["visualization"]["debug"]["life_time"])

    def draw_global_route(self):
        route = [wp[0] for wp in self.route][::50]
        draw_waypoints(self.world, route, life_time=self.config["visualization"]["debug"]["life_time"],
                       color=carla.Color(0, 255, 0, 125))

    def draw_predictions(self, max_points=100, downsampler=5):
        color = carla.Color(0, 0, 255, 125)
        for actor_id, predictions in self.world_model.predictions.items():
            for prediction in predictions:
                prediction_points = prediction[:max_points:downsampler]
                prediction_headings = prediction.get_heading()[:max_points:downsampler]
                draw_points(self.world, prediction_points, life_time=self.config["visualization"]["debug"]["life_time"],
                            color=color, headings=prediction_headings)

    def misc_updates(self):
        self.update_spectator()
        self.fps, self.frame_count, self.start_time = update_fps(self.frame_count, self.start_time,
                                                                 last_fps=self.fps)  # Calculate FPS
        self.world_model.update()

    def visualize(self):
        self.draw_global_route()
        self.draw_predictions()

        if self.config["visualization"]["world"]["active"]:
            draw_world(self.world_model, self.ego.id, self.fps)  # Draw the world

        if self.config['visualization']['birds_eye_camera']['active']:
            # Display the image using OpenCV
            if self.latest_image is not None:
                # cv works on bgr(a) instead of rgb(a), so we swap axis 0 and 2
                img = self.latest_image[..., [2, 1, 0]]
                cv2.imshow('Bird\'s Eye View', img)
                cv2.waitKey(1)

        if self.config["visualization"]["debug"]["live_ego_plots"]:  # ToDo:  move to visualization.py
            velocity = self.agent.get_ego_velocity()
            heading = self.agent.get_ego_heading(rad=True)
            acceleration = self.agent.get_ego_acceleration()

            throttle = self.agent.throttle
            brake = self.agent.brake
            if self.config["agent"]["name"] == "Custom":
                planned_vel = self.agent.planned_traj.v[1]
                planned_heading = self.agent.planned_traj.heading[1]
                planned_acc = self.agent.planned_traj.a[1]

                if self.agent.opt_flag == True:
                    opt_flag = 1
                else:
                    opt_flag = 0

            # Append data for plotting
            self.time_data.append(self.idx)
            self.vel_data.append(velocity)
            self.h_data.append(heading)
            self.acc_data.append(acceleration)
            self.throttle_data.append(throttle)
            self.brake_data.append(brake)
            if self.config["agent"]["name"] == "Custom":
                self.p_vel_data.append(planned_vel)
                self.p_h_data.append(planned_heading)
                self.p_acc_data.append(planned_acc)
                self.opt_flag_data.append(opt_flag)

            # Update the plot data
            self.vel_line.set_data(self.time_data, self.vel_data)
            self.acc_line.set_data(self.time_data, self.acc_data)
            self.throttle_line.set_data(self.time_data, self.throttle_data)
            self.brake_line.set_data(self.time_data, self.brake_data)
            if self.config["agent"]["name"] == "Custom":
                self.p_vel_line.set_data(self.time_data, self.p_vel_data)
                self.p_h_line.set_data(self.time_data, self.p_h_data)
                self.p_acc_line.set_data(self.time_data, self.p_acc_data)
                self.opt_flag_line.set_data(self.time_data, self.opt_flag_data)

            self.vliml_line.set_data(self.time_data, np.abs(self.config["planner"]["v_min"]))
            self.aliml_line.set_data(self.time_data, np.abs(self.config["planner"]["a_min"]))
            self.alimu_line.set_data(self.time_data, np.abs(self.config["planner"]["a_max"]))

            # Adjust plot limits to fit new data
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.ax3.relim()
            self.ax3.autoscale_view()
            self.ax4.relim()
            self.ax4.autoscale_view()

            # Redraw the plot
            plt.pause(0.01)
            self.idx += 1

    def log_times(self):
        iterables = [
            ("Perception", self.dt_perceive),
            ("Prediction", self.dt_predict),
            ("Planning", self.dt_plan),
            ("Control", self.dt_control),
            ("Viz", self.dt_viz),
            ("Other", self.dt_other),
            ("Tick", self.dt_tick)
        ]

        msg = f'\rFPS: {1.0 / self.iteration_time:.2f}'
        for (name, t) in iterables:
            msg += f" | {name}: {100 * t / self.iteration_time:.1f}%"
        sys.stdout.write(msg)

    def log_simulation_step(self):
        # print("skipping logging for now")
        self.simulation_log.store_actors(self.world_model, self.current_frame)
        self.simulation_log.store_predictions(self.world_model, self.current_frame)
        self.simulation_log.store_plan(self.world_model, self.current_frame)
        self.simulation_log.store_optimization_time(self.agent, self.current_frame)
        self.simulation_log.store_control(self.world_model, self.current_frame)
        self.simulation_log.store_feasibility(self.agent, self.current_frame)

    def check_timeout(self):
        total_time = time.time() - self.simulation_start_time
        # should_raise_error = self.args.scenario is not None and total_time >= self.config["misc"]["scenario_timeout"]
        # if should_raise_error:
        #     raise TimeoutError("Scenario timed out.")

    def check_stuck(self):
        current_x = self.ego.get_transform().location.x
        current_y = self.ego.get_transform().location.y
        current_z = self.ego.get_transform().location.z

        if self.last_ego_position is None:
            self.last_ego_position = (current_x, current_y)
            return

        last_x, last_y = self.last_ego_position

        if self.wait_counter == 0 and current_x == last_x and current_y == last_y:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_ego_position = (current_x, current_y)
        if self.wait_counter == 0:
            print("Ego position: ", current_x, current_y, current_z)

        if np.abs(current_z) >= self.config["misc"]["z_lim"]:
            msg = "Ego seems to be flying or falling"
            print(msg)
            raise TimeoutError(msg)

        if self.stuck_counter >= self.config["misc"]["stuck_counter_threshold"]:
            msg = "TimeOut. Ego vehicle seems stuck."
            print(msg)
            raise TimeoutError(msg)
        
    def setup_birds_eye_camera(self, config):
        self.birds_eye_camera = None
        self.latest_image = None  # Global variable to store the latest image
        if config['active']:
            self.birds_eye_camera = setup_birds_eye_camera(self.ego, config)
            self.birds_eye_camera.listen(lambda image: keep_latest_image(image, self, 'latest_image'))

    def run(self):
        # Let the vehicles settle after spawning
        M = 20
        for _ in range(M):
            # make sure the spectator is where the ego is on the first frame
            self.update_spectator()  

            # Save all the actors, timestemp
            self.world_model.update()
            self.world.tick()

        dt0 = time.time()
        prev_a = 0
        self.idx = 0
        while True:
            # If we're simulating a scenario, check if the timeout has been reached
            self.check_timeout()  

            # Based on our self.stuck_counter and our configged stuck_counter_threshold...
            # Checks if we're stuck (distanced based)
            self.check_stuck()

            # Update spectator, FPS, world_model
            dt = time.time()
            self.misc_updates()
            self.dt_other = time.time() - dt  # time spent with other simulation updates

            # For now... update actors and map with ground truth
            dt = time.time()
            self.agent.perceive()
            self.dt_perceive = time.time() - dt  # time spent in perception

            # Loads a prediction model based on what's configged
            # 
            dt = time.time()
            self.agent.predict()
            self.dt_predict = time.time() - dt  # time spent in prediction
            # print(f"Total time spent predicting: {self.dt_predict:.5f} sec")

            if self.wait_counter == 0:  # otherwise currently waiting. Do not plan or control ego
                dt = time.time()
                # pdb.set_trace()
                self.agent.u_prev_prev = [np.copy(self.agent.u_prev[0]), np.copy(self.agent.u_prev[1])]
                self.agent.plan()
                if self.config['agent']['name'] == "Custom":  # Carla agents do not have this attribute
                    self.draw_planned_route()
                self.dt_plan = time.time() - dt  # time spent in planning

                dt = time.time()
                # pdb.set_trace()
                try:
                    self.agent.control()
                except IndexError:
                    print("Warning: Control failed due to infeasible trajectory. Stopping vehicle.")
                    self.ego.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
                    continue
            else:  # keep stopped
                current_transform = self.ego.get_transform()
                self.ego.set_transform(current_transform)
                self.ego.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))

            NUM = 10
            DEC = 4
            print("")
            print("[main.py] self.agent.planned_traj.x", np.round(self.agent.planned_traj.x[:NUM], decimals=DEC))
            print("[main.py] self.agent.planned_traj.y", np.round(self.agent.planned_traj.y[:NUM], decimals=DEC))
            print("[main.py] self.agent.planned_traj.a", np.round(self.agent.planned_traj.a[:NUM], decimals=DEC))
            print("[main.py] self.agent.planned_traj.df", np.round(self.agent.planned_traj.df[:NUM], decimals=DEC))
            print("[main.py] np.diff(self.agent.planned_traj.a)",
                  np.round(np.diff(self.agent.planned_traj.a)[:NUM], decimals=DEC))
            print("[main.py] np.diff(self.agent.planned_traj.df)",
                  np.round(np.diff(self.agent.planned_traj.df)[:NUM], decimals=DEC))
            print("[main.py] u_prev_prev[0] - u_prev[0]",
                  np.round(self.agent.u_prev_prev[0] - self.agent.u_prev[0], decimals=DEC))
            print("[main.py] u_prev_prev[1] - u_prev[1]",
                  np.round(self.agent.u_prev_prev[1] - self.agent.u_prev[1], decimals=DEC))

            # if (self.agent.done() or self.scenario_done) and self.scenario is not None:
            #     self.scenario_done = True
            #     if self.wait_counter == 0:
            #         print(
            #             f"Scenario finished successfully. Waiting for {self.config['misc']['wait_after_scenario']} frames and exiting..")

            #     if self.wait_counter >= self.config["misc"]["wait_after_scenario"]:
            #         break

            #     if self.wait_counter % 10 == 0:
            #         print(f"Waiting frames left: {self.config['misc']['wait_after_scenario'] - self.wait_counter}...")
            #     self.wait_counter += 1
            #     time.sleep(0.05)

            # elif self.agent.done() and self.config['agent']['loop']:  # roam around indefinitely
            #     print("reached destination. Setting new destination")
            #     self.agent.set_destination(random.choice(self.spawn_points).location)
            if self.agent.done() and self.config['agent']['loop']:  # roam around indefinitely
                print("reached destination. Setting new destination")
                self.agent.set_destination(random.choice(self.spawn_points).location)

            # Apply control to other agents in scenario
            # if self.scenario is not None:
            #     self.scenario.run_step()
            self.dt_control = time.time() - dt  # time spent in control

            dt = time.time()
            self.log_simulation_step()
            self.dt_other += time.time() - dt  # time spent with other simulation updates

            dt = time.time()  # ToDo: Add visualization options as optional in config
            self.visualize()
            self.dt_viz = time.time() - dt  # time spent visualizing stuff

            current_time = time.time() - dt0

            dt = time.time()
            self.world.tick()  # Advance the simulation to the next frame
            self.dt_tick = time.time() - dt

            self.iteration_time = time.time() - dt0
            if self.config["misc"]["show_time_on_console"]:
                self.log_times()
            dt0 = time.time()
            self.current_frame += 1
            self.counter += 1

            print("=============================================================")
            print("=============================================================")

        self.simulation_log.store_timeout(False)

    def _try_destroy_actor(self, actor):
        try:
            actor.destroy()
        except AttributeError:
            print("Tried to destroy 'other', but did not exist.")

    def cleanup(self):
        self.settings.synchronous_mode = False  # Disable synchronous mode before exiting
        self.world.apply_settings(self.settings)
        self.ego.destroy()

        # if self.scenario is not None:
        #     for other in self.scenario.others:
        #         self._try_destroy_actor(other)

        if hasattr(self, 'other'):
            self._try_destroy_actor(self.other)

        if self.config['visualization']['birds_eye_camera']['active'] and self.birds_eye_camera is not None:
            self.birds_eye_camera.stop()  # Stop the camera from listening to new images
            self.birds_eye_camera.destroy()  # Remove the camera from the simulation
        cv2.destroyAllWindows()

    def destroy_all_vehicles(self):
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                self.world.tick()

            # Get all actors in the world
            actors = self.world.get_actors()

            # Filter actors to get only vehicles
            vehicles = actors.filter('vehicle.*')

            if vehicles:
                print(f"Destroying {len(vehicles)} vehicles")
                for vehicle in vehicles:
                    vehicle.destroy()
            else:
                print("No vehicles found in the simulation.")

        except Exception as e:
            print(f"An error occurred: {e}")

def try_run(args):
    simulation = Simulation(args)
    should_hang = False
    try:
        simulation.run()
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt")
        simulation.simulation_log.store_timeout(False)
    except TimeoutError as e:
        print(f"Timed out")
        simulation.simulation_log.store_timeout(True)
        should_hang = True  # halt process after finally (to make sure we save logs and such), and then wait for parent process to kill this due to timeout
    finally:
        # save cached roadgraph idxs, if they exist
        if hasattr(simulation.agent.predictor, "renderer"):
            renderer = simulation.agent.predictor.renderer
            savefile = os.path.join(PROJECT_DIR, decouple.config("CACHE_DIR"), renderer.CACHED_RG_FILTER_NAME + ".pkl")
            with open(savefile, "wb") as f:
                pickle.dump(renderer.cached_rg_idxs, f)

        simulation_time = time.time() - simulation.simulation_start_time  # Real time in seconds that the simnulation was running
        simulation.simulation_log.store_simulation_time(simulation_time)
        simulation.cleanup()
        simulation.client.stop_recorder()
        simulation.simulation_log.save(simulation.config['logging']['directory'], simulation.run_name,
                                       simulation.world_model)
        if should_hang:
            print(
                f"Hanging for {simulation.config['misc']['scenario_timeout_slack']} seconds. Waiting to be killed, or finish hanging.")
            time.sleep(simulation.config['misc']['scenario_timeout_slack'])

def parse_args():
    parser = argparse.ArgumentParser()

    # scenario_options = []
    # scenario_config_files = os.listdir(os.path.join(PROJECT_DIR, "src/scenario/config"))
    # for scenario_config_file in scenario_config_files:
    #     scenario_options.append(scenario_config_file.split("_")[0])
    #     scenario_options.append(scenario_config_file)

    # parser.add_argument(
    #     '-sc', '--scenario',
    #     type=str,
    #     default=None,
    #     help="Scenario to be played. Options: scX where X=scenario number",
    #     choices=sorted(scenario_options)
    # )

    # parser.add_argument("-a", "--all", action="store_true", help="Evaluate all scenarios")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    try_run(args)

    # if args.all:
    #     raise NotImplementedError()  # Not functional yet
    #     all_scenarios = utils.get_all_scenario_filenames(remove_extension=False)
    #     print("Simulating all scenarios:", all_scenarios)

    #     for scenario in all_scenarios:
    #         print("Simulating", scenario)
    #         args.run_name = scenario

    #         try_run(args)

    # else:
    #     try_run(args)
