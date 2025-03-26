from typing import List
import numpy as np

class TrajectoryPrediction:
    def __init__(self, timesteps, n_dims):
        self.timesteps = timesteps
        self.n_dims = n_dims
        self._prediction = np.zeros((timesteps, n_dims))

    def __getitem__(self, index):
        return self._prediction[index]

    def __setitem__(self, index, value):
        self._prediction[index] = value

    def set_x(self, x):
        self._prediction[:, 0] = x

    def set_y(self, y):
        self._prediction[:, 1] = y

    def set_xy(self, xy):
        self._prediction[:, :2] = xy

    def set_z(self, z):
        self._prediction[:, 2] = z

    def shorten(self, new_length):
        self._prediction = self._prediction[:new_length]
        self.timesteps = new_length

    @property
    def x(self):
        return self._prediction[:, 0]

    @property
    def y(self):
        return self._prediction[:, 1]

    @property
    def xy(self):
        return self._prediction[:, :2]

    @property
    def xyz(self):
        if self.n_dims <= 2:
            raise AttributeError('n_dims must be greater than 2')

    @property
    def xyzh(self):
        if self.n_dims <= 3:
            raise AttributeError('n_dims must be greater than 3')

    def compute_heading(self):
        # Computes heading to the next xy point
        # ToDO: implement
        raise NotImplementedError

    def get_heading(self):
        #returns heading angle from predcted points
        heading_angles = []
        points = self.xy

        for i in range(len(points) - 1):
            # Get current and next point
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Compute difference in x and y
            dx = x2 - x1
            dy = y2 - y1

            # Compute the angle using atan2, which gives the angle in radians
            angle = np.arctan2(dy, dx)

            # Append the angle to the list
            heading_angles.append(angle)

        # make the heading angle of the final point same as second to last
        heading_angles.append(angle)

        return heading_angles

    def to_numpy(self):
        return self._prediction

    def to_dict(self):
        return {
            "coords": self._prediction
        }


class BasePredictor:
    def __init__(self, predictor_config: dict):
        self.name = predictor_config['model']
        self.alias = predictor_config['alias']
        self.predict_heading = predictor_config['predict_heading']
        self.predict_z = predictor_config['predict_z']
        self.prediction_horizon = predictor_config['prediction_horizon']
        self.prediction_step = predictor_config['prediction_step']

        if "n_trajectories" in predictor_config:
            self.n_trajectories = predictor_config['n_trajectories']

        self.n_dims = 2
        if self.predict_z:
            self.n_dims += 1
        if self.predict_heading:
            self.n_dims += 1

    def check_inputs(self, args: dict):
        raise NotImplementedError

    def prepare_data(self, data):
        # will be called once before making predictions for each agent. Particularly useful for data-driven models that
        # might ingest data in a very specific format. If not overriden it does nothing
        return data

    def predict(self, args: dict) -> List[TrajectoryPrediction]:
        # Takes variable number of inputs depending on the method: actor state, actor history, surrounding actors, environment (e.g. road/traffic lights...etc)
        # ToDo: establish format for this
        raise NotImplementedError

class ConstantVelocity(BasePredictor):
    def __init__(self, predictor_config):
        super().__init__(predictor_config)
        self.n_trajectories = 1

    def check_inputs(self, args):
        if "actor_state" not in args:
            raise ValueError("actor_state is required")

    def predict(self, args: dict) -> List[TrajectoryPrediction]:
        self.check_inputs(args)

        actor_state = args["actor_state"]
        horizons = np.arange(self.prediction_step, self.prediction_horizon + self.prediction_step,
                             self.prediction_step)  # ToDo: get prediction horizon and resolution from config
        timesteps = len(horizons)
        prediction = TrajectoryPrediction(timesteps, self.n_dims)
        prediction[:, 0] = actor_state.x + actor_state.vx * horizons
        prediction[:, 1] = actor_state.y + actor_state.vy * horizons

        if self.predict_z:
            prediction[:, 2] = actor_state.z + actor_state.vz * horizons

        if self.predict_heading:
            prediction[:, self.n_dims - 1] = prediction.compute_heading()

        return [prediction]