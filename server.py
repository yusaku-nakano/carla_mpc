import random

# requests to server (e.g. spawning vehicles)
def spawn_vehicle(world, spawn_point, autopilot=False, ego_vehicle=False, tick_world=True,
                  bp=None):  # ToDo: allow blueprint specification in config
    blueprint_library = world.get_blueprint_library()

    if bp is None:
        # Filtering out bicycles, motorcycles, etc.
        vehicles = [x for x in blueprint_library.filter('vehicle.*') if
                    int(x.get_attribute('number_of_wheels')) == 4 and 'truck' not in x.id.lower()]
        vehicle_bp = random.choice(vehicles)
    else:
        vehicle_bp = blueprint_library.filter(bp)[0]
    if ego_vehicle:
        # set vehicle's role to 'hero' --> allows scenario_runner to identify this vehicle as ego. Also needed for waymo's data format
        vehicle_bp.set_attribute('role_name', 'hero')

    if spawn_point == 'random':
        spawn_point = random.choice(world.get_map().get_spawn_points())
        # spawn_point.location.z = 0.6 # offset z to avoid collision
    print(f"STARTING POINT: {spawn_point}.")

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    if tick_world:
        world.tick()

    if autopilot:
        """
        [!!] Note: calling the set_autopilot method instantiates a traffic manager (on default port 8000), even if
        autopilot is set to False. This can cause issues with scenario_runner if trying to spawn a traffic manager
        on the same port.
        """
        vehicle.set_autopilot(autopilot)
    return vehicle