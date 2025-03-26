import carla
import os
import pickle
import numpy as np
import copy
import decouple


class SimulationLog:
    """
    In addition to CARLA's default recorder, which stores how the simulation developed (e.g. actor positions, velocities, etc...),
    log additional information for computation of submodule metrics. E.g: predictions of agents at each point, or planned path of
    the ego vehicle at each step
    ToDo: Efficient logging & writting to file. For now simply store basic info and save after script ends
    ToDo: Don't log anything that's not absolutely needed. Otherwise logs become massive
    """

    def __init__(self):
        # Dictionaries of {frame_number : [pred/plan...etc] info}
        self.actors = {}  # detected actors, not actual actors
        self.actors_gt = {}  # actual ground truth actors
        self.predictions = {}
        self.plans = {}
        self.optimization_times = {}
        self.controls = {}
        self.rg = None  # Waymo's roadgraph. Mainly for plotting purposes later
        self.ego_id = None
        self.global_plan = None
        self.scenario_config = None
        self.config = None
        self.timeout = None
        self.feasibility = {}
        self.collisions = {}

    @staticmethod
    def load_from_file(run_name):
        folder = os.path.join(decouple.config("PROJECT_DIR"), decouple.config("LOG_DIR"))
        path = os.path.join(folder, f"{run_name}.pkl")

        # allow file's shortcut name
        for filename in os.listdir(folder):
            if not filename.endswith(".pkl"):
                continue
            if run_name in filename:
                path = os.path.join(folder, filename)
                print(f"\tLoaded from {filename}")
                break

        with open(path, "rb") as f:
            log_data = pickle.load(f)

        log = SimulationLog()
        for key, data in log_data.items():
            setattr(log, key, data)

        return log

    def set_ego_id(self, ego_id):
        self.ego_id = ego_id

    def set_global_plan(self, global_plan):
        # global plan is a list of (waypoint, option) from carla
        xy = [(wp.transform.location.x, wp.transform.location.y) for (wp, opt) in global_plan]
        xy = np.array(xy)
        self.global_plan = xy

    def _store_actor_porperties_in_dict(self, d, actors_list, decimals=3):
        dec = decimals
        for actor in actors_list:
            location = actor.get_location()
            yaw = np.round(actor.get_transform().rotation.yaw, decimals=dec)
            velocity = actor.get_velocity()
            acceleration = actor.get_acceleration()
            speed = np.linalg.norm([velocity.x, velocity.y])
            acc = np.linalg.norm([acceleration.x, acceleration.y])
            speed = np.round(speed, decimals=dec)
            acc = np.round(acc, decimals=dec)
            xyz = np.round(location.x, decimals=dec), np.round(location.y, decimals=dec), np.round(location.z,
                                                                                                   decimals=dec)

            props = np.concatenate([xyz, yaw, speed, acc], axis=None)
            d[actor.id] = props

    def store_actors(self, wm, frame):
        # ToDo: only store road users. Right now, all carla actors are stored
        # ToDo: store relevant properties
        # ToDo: define storage format
        actors_to_log = {}
        self._store_actor_porperties_in_dict(actors_to_log, wm.actors)
        self.actors[frame] = actors_to_log

        actors_gt_to_log = {}
        self._store_actor_porperties_in_dict(actors_gt_to_log, wm.actors_gt)
        self.actors_gt[frame] = actors_gt_to_log

    def store_predictions(self, wm, frame):
        self.predictions[frame] = copy.deepcopy(wm.predictions)

    def store_plan(self, wm, frame):
        self.plans[frame] = wm.get_plan()

    def store_optimization_time(self, agent, frame):
        self.optimization_times[frame] = agent.planner.last_optimization_time

    def store_control(self, wm, frame):
        # manually copy to dict to avoid dependency on carla.VehicleControl class
        if isinstance(wm.control, carla.VehicleControl):
            self.controls[frame] = {
                "throttle": wm.control.throttle,
                "steer": wm.control.steer,
                "brake": wm.control.brake,
                "hand_brake": wm.control.hand_brake,
                "reverse": wm.control.reverse,
                "manual_gear_shift": wm.control.manual_gear_shift,
                "gear": wm.control.gear,
            }
        if isinstance(wm.control, carla.VehicleAckermannControl):
            self.controls[frame] = {
                "acceleration": wm.control.acceleration,
                "steer": wm.control.steer,
                "steer_speed": wm.control.steer_speed,
                "speed": wm.control.speed,
                "jerk": wm.control.jerk,
            }

    def store_simulation_time(self, t):
        self.total_time = t

    def store_timeout(self, value=True):
        self.timeout = value

    def store_feasibility(self, agent, frame):
        self.feasibility[frame] = agent.is_feasible()

    def store_collision(self, collision_event, frame):
        loc = collision_event.transform.location
        imp = collision_event.normal_impulse
        data = {
            'server_frame': collision_event.frame,
            'server_timestamp': collision_event.timestamp,
            'location': (loc.x, loc.y, loc.z),  # will correspond with ego ego location
            'heading': collision_event.transform.rotation.yaw,  # will correspond with ego heading
            'actor_id': collision_event.actor.id,  # ego
            'other_actor_id': collision_event.other_actor.id,
            'normal_impulse': (imp.x, imp.y, imp.z),
        }
        self.collisions[frame] = data

    def get_actors_gt(self, ids, is_ego=False):
        # Returns gt dictionary reformatted s.t.:
        # result = {
        #   'frames' : [all frames]
        #   'actor_id' : [all coords]
        # }
        #

        result = {
            'frames': []
        }

        for frame, actor_gts in self.actors_gt.items():
            result['frames'].append(int(frame))
            for actor_id in ids:
                key = "data" if is_ego else actor_id

                if key not in result:
                    result[key] = []
                result[key].append(actor_gts[actor_id])

        for k in result.keys():
            result[k] = np.array(result[k])

        return result

    def get_ego_gt(self):
        return self.get_actors_gt([self.ego_id], is_ego=True)

    def get_ego_xy(self, from_planned=True):
        if from_planned:
            return self.get_planned_xy()[:, 1]  # all frames, current timestep
        # data --> x,y,z, yaw, speed, acc
        return self.get_ego_gt()['data'][:, :2]

    def get_ego_heading(self, from_planned=True):
        h_from_carla = self.get_ego_gt()['data'][:, 3]
        if from_planned:
            h_from_plan = self.get_planned_heading()[:, 1]
            h_from_plan[np.isnan(h_from_plan)] = h_from_carla[np.isnan(h_from_plan)]
            return h_from_plan # all frames, current timestep
        # data --> x,y,z, yaw, speed, acc
        return h_from_carla

    def get_ego_speed(self, from_planned=True):
        v_from_carla = self.get_ego_gt()['data'][:, 4]
        if from_planned:
            v_from_plan = self.get_planned_v()[:, 1]
            v_from_plan[np.isnan(v_from_plan)] = v_from_carla[np.isnan(v_from_plan)]
            return  v_from_plan # all frames, current timestep
        # data --> x,y,z, yaw, speed, acc
        return v_from_carla

    def get_ego_acceleration(self, from_planned=True):
        a_from_carla = self.get_ego_gt()['data'][:, 5]
        if from_planned:
            a_from_plan = self.get_planned_a()[:, 1]
            a_from_plan[np.isnan(a_from_plan)] = a_from_carla[np.isnan(a_from_plan)]
            return a_from_plan # all frames, current timestep
        # data --> x,y,z, yaw, speed, acc
        return a_from_carla

    def get_ego_id(self):
        return self.ego_id

    def get_plans(self, return_frames=False):
        # each plan is a list of tuples with: (x, y, heading, v, a, ey, epsi, s)

        frames = []
        plans = []

        for frame, plan in self.plans.items():
            frames.append(int(frame))
            plans.append(plan.tolist())

        plans = np.array(plans)

        if return_frames:
            frames = np.array(frames)
            return frames, plans

        return plans

    def get_optimization_times(self, return_frames=False):
        frames = []
        times = []

        for frame, t in self.optimization_times.items():
            frames.append(int(frame))
            times.append(t)

        times = np.array(times)

        if return_frames:
            frames = np.array(frames)
            return frames, times

        return times

    def get_planned_xy(self):
        plans = self.get_plans()
        return plans[..., :2]

    def get_planned_heading(self):
        plans = self.get_plans()
        return plans[..., 2]

    def get_planned_v(self):
        plans = self.get_plans()
        return plans[..., 3]

    def get_planned_a(self):
        plans = self.get_plans()
        return plans[..., 4]

    def get_total_number_of_frames(self):
        ego_data = self.get_ego_gt()
        frames = ego_data["frames"]
        return len(frames)

    def get_total_number_of_seconds(self):
        # gets the total number of simulation time in seconds, based on the number of available frames and the FPS
        num_frames = self.get_total_number_of_frames()
        num_seconds = float(num_frames) / self.config["server"]["fps"]
        return num_seconds

    def get_frames_of_interest(self, return_frames=False, return_distances=False):
        # For scenarios with a point of interest specified, it returns booleans indicating whether the ego is within a
        # relevant distance (calculated from ego target speed and maximum horizon of 8s)
        ego_data = self.get_ego_gt()
        frames, ego_xy = ego_data["frames"], ego_data["data"][..., :2]

        poi = self.scenario_config["poi"]
        poi_xy = np.array([poi["x"], poi["y"]])

        ego_poi_diff = ego_xy - poi_xy
        distances = np.linalg.norm(ego_poi_diff, axis=1)

        max_h = 9.0  # max is 8, add a little slack
        v_max = self.config["planner"]["v_max"]
        threshold = v_max * max_h

        within_distance = distances <= threshold

        returnables = [within_distance]

        if return_frames:
            returnables = [frames, within_distance]

        if return_distances:
            returnables.append(distances)  # different appending for compatibility with older scripts

        return returnables

    def get_nonego_ids(self):
        # find all actors for which there are predictions, and remove ego_id (in case there are also predictions for it)
        first_frame = list(self.predictions.keys())[0]
        agent_ids = list(self.predictions[first_frame].keys())
        if self.ego_id in agent_ids:
            agent_ids.remove(self.ego_id)
        return agent_ids

    def get_control_signals(self):
        # Returns controls dictionary reformatted s.t.:
        # result = {
        #   'frames' : [all frames]
        #   'signal' : [all values of this signal]
        # }
        #

        result = {
            'frames': []
        }
        for frame, controls in self.controls.items():
            result['frames'].append(int(frame))
            for signal, value in controls.items():
                if signal not in result:
                    result[signal] = []
                result[signal].append(value)
        return result

    def get_feasibility(self):
        frames = []
        is_feasible = []
        for frame, is_f in self.feasibility.items():
            frames.append(int(frame))
            is_feasible.append(is_f)
        return frames, is_feasible

    def get_collisions(self):
        frames, _ = self.get_feasibility()
        is_collision = [False] * len(frames)

        for f, collision_event in self.collisions.items():
            is_collision[f] = True

        return frames, is_collision

    def set_config(self, config):
        self.config = config

    def set_scenario_config(self, config):
        self.scenario_config = config

    def save(self, log_dir, run_name, wm):
        data = {
            "actors": self.actors,
            "actors_gt": self.actors_gt,
            "predictions": self.predictions,
            "plans": self.plans,
            "controls": self.controls,
            "ego_id": self.ego_id,
            "global_plan": self.global_plan,
            "scenario_config": self.scenario_config,
            "config": self.config,
            "feasibility": self.feasibility,
            "collisions": self.collisions,
            "optimization_times": self.optimization_times,
            "timeout": self.timeout,
            "total_time": self.total_time,
        }
        # data = {
        #     "actors": self.actors,
        #     "actors_gt": self.actors_gt,
        #     "predictions": self.predictions,
        #     "plans": self.plans,
        #     "controls": self.controls,
        #     "rg": wm.to_waymo_format(only_rg=True),
        #     "ego_id": self.ego_id,
        #     "global_plan": self.global_plan,
        #     "scenario_config": self.scenario_config,
        #     "config": self.config,
        #     "feasibility": self.feasibility,
        #     "collisions": self.collisions,
        #     "optimization_times": self.optimization_times,
        #     "timeout": self.timeout,
        #     "total_time": self.total_time,
        # }

        with open(os.path.join(log_dir, f'{run_name}.pkl'), 'wb') as f:
            pickle.dump(data, f)