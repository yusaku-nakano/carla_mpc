class ActorState:
    "Gets state from carla actor. ToDo: merge with other 'state' classes"

    def __init__(self, carla_actor):
        current_loc = carla_actor.get_location()
        current_v = carla_actor.get_velocity()

        self.x = current_loc.x
        self.y = current_loc.y
        self.z = current_loc.z

        self.vx = current_v.x
        self.vy = current_v.y
        self.vz = current_v.z