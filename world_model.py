import uuid
import numpy as np

class WorldModel:
    # Note: "_gt" stands for ground truth

    def __init__(self, world, ego=None, plot_every=None):
        self.STATE_KEYS = ["x", "y", "z", "bbox_yaw", "length", "width", "height", "speed", "timestamp_micros",
                           "vel_yaw", "velocity_x", "velocity_y", "valid"]
        self.TL_KEYS = ["state", "x", "y", "z", "id", "valid", "timestamp_micros"]

        self.world = world  # carla world
        self.map = None
        self.map_gt = world.get_map()  # ToDo: update to custom map format? [If time allows]
        self.map_carla = world.get_map()
        self.timestamp_seconds = world.get_snapshot().timestamp.elapsed_seconds
        self.timestamp_seconds_start = world.get_snapshot().timestamp.elapsed_seconds  # Start time of the world model
        self.actors = None  # most recent actors
        self.actors_gt = world.get_actors()  # ToDo: update to custom actor format? [If time allows]
        self.actors_carla = world.get_actors()
        self.predictions = None  # Most recent predictions
        self.predictions_gt = None  # ToDo: simulate ground truth trajectories [Likely won't do at the end]
        self.ego = None  # Assumed only 1
        self.planned_route = None  # Most recent ego plan
        self.control = None  # Most recent calculated control to be applied to ego
        self.plot_every = plot_every  # Plot the world model (from waymo format) every X frames.
        self._ego_predictions = False
    
    def set_ego(self, ego):
        self.ego = ego

    def update(self):
        self.actors_gt = self.world.get_actors()
        self.actors_carla = self.world.get_actors()
        self.timestamp_seconds = self.world.get_snapshot().timestamp.elapsed_seconds
        # self.update_waymo_format()
    
    def set_ego_predictions(self, value):
        self._ego_predictions = value  # value is boolean

    def get_plan(self):
        #includes other properties like planned heading, velocity, acceleration..etc
        return np.array(self.planned_route)

    def to_waymo_format(self, past_downsampler=1, only_rg=False):
        # converts the current world model into a similar format as waymo's motion tf.Example protos.
        # https://waymo.com/open/data/motion/tfexample/

        data = {
            "scenario/id": hex(int(uuid.uuid4())),  # [1], string, hexadecimal string
        }

        data.update(self.waymo_roadgraph)
        if only_rg:
            return data
        data.update(self.waymo_state_current)
        data.update(self.waymo_tl_current)
        data.update(self.waymo_state_misc)

        if past_downsampler is not None:
            waymo_state_past_downsampled = downsample_waymo_dict(self.waymo_state_past, past_downsampler)
            waymo_tl_past_downsampled = downsample_waymo_dict(self.waymo_tl_past, past_downsampler, is_tl=True)

            data.update(waymo_state_past_downsampled)
            data.update(waymo_tl_past_downsampled)

            return data

        data.update(self.waymo_state_past)
        data.update(self.waymo_tl_past)

        return data

def downsample_waymo_dict(d, downsampler, is_tl=False):
        d_downsampled = {}
        for k, v in d.items():
            if not is_tl:  # downsample of the other dimension
                d_downsampled[k] = v[:, ::downsampler]
            else:
                d_downsampled[k] = v[::downsampler, :]

        return d_downsampled