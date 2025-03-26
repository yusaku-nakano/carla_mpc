import carla
import numpy as np
import math
import sys
import cv2

def draw_points(world, points, z=0.7, life_time=1.0, color=None, headings=None):
    """
    Draw a list of 2D points at a certain height given in z.

        :param world: carla.world object
        :param points: list or iterable container with the 2D points to draw
        :param z: height in meters
        :param color: carla.Color rgba
    """
    for i, p1 in enumerate(points[:-1]):
        p2 = points[i + 1]

        base_z1 = 0
        base_z2 = 0
        if len(p1) == 3:
            base_z1 = p1[2]
            base_z2 = p2[2]
        if headings is not None:
            h1 = headings[i]
            h2 = headings[i + 1]

        begin = carla.Location(x=p1[0], y=p1[1], z=base_z1 + z)

        end = carla.Location(x=p2[0], y=p2[1], z=base_z2 + z)
        if headings is not None:  # use provided headings
            end = begin + carla.Location(x=math.cos(h1), y=math.sin(h1))
        if color is not None:
            world.debug.draw_arrow(begin, end, arrow_size=0.3, color=color, life_time=life_time)
        else:
            world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=life_time)

def draw_waypoints(world, waypoints, z=0.5, life_time=1.0, color=None):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
        :param color: carla.Color rgba
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))

        if color is not None:
            world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=life_time, color=color)
        else:
            world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=life_time)

def draw_map(world_map, img):
    # map = world.get_map() # Avoid calling this every time, I think it's expensive
    waypoints = world_map.generate_waypoints(distance=20.0)  # Adjust distance as needed
    for waypoint in waypoints:
        # Transform waypoint location to pixel space for simplification
        x = int(waypoint.transform.location.x) + 600  # Adjust offset as needed
        y = int(waypoint.transform.location.y) + 400  # Adjust offset as needed
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1)  # Draw waypoints as small circles

        # Optionally, draw connections between waypoints for visualization of road links
        for next_wp in waypoint.next(20.0):  # Adjust distance as needed
            next_x = int(next_wp.transform.location.x) + 600  # Adjust offset as needed
            next_y = int(next_wp.transform.location.y) + 400  # Adjust offset as needed
            cv2.line(img, (x, y), (next_x, next_y), (255, 255, 255), 1)

def draw_traffic_lights(img, world):
    # Define colors for traffic light states
    colors = {
        carla.TrafficLightState.Red: (0, 0, 255),  # Red
        carla.TrafficLightState.Yellow: (0, 255, 255),  # Yellow
        carla.TrafficLightState.Green: (0, 255, 0),  # Green
        carla.TrafficLightState.Off: (255, 255, 255),  # White (Off)
        carla.TrafficLightState.Unknown: (128, 128, 128)  # Gray (Unknown)
    }

    for tl in world.get_actors().filter('traffic.traffic_light'):
        # Assume a simple representation of traffic lights as circles on the map
        location = tl.get_location()
        # Convert world location to image coordinates (adjust as needed)
        x = int(location.x) + 600  # Adjust offset as needed
        y = int(location.y) + 400  # Adjust offset as needed

        # Get the traffic light state and select the corresponding color
        state_color = colors.get(tl.state, (128, 128, 128))  # Default to gray if state is unknown

        # Draw the traffic light on the image
        cv2.circle(img, (x, y), 5, state_color, -1)  # Adjust the circle size as needed

def rotate_point_around_origin(px, py, theta):
    """Rotate a point around the origin (0, 0) by theta degrees."""
    theta_rad = math.radians(theta)
    rx = px * math.cos(theta_rad) - py * math.sin(theta_rad)
    ry = px * math.sin(theta_rad) + py * math.cos(theta_rad)
    return rx, ry

def draw_actor_bounding_box(img, actor, color=(0, 255, 0)):  # Default color is green
    """Draws the bounding box of a given actor on the image with the specified color."""
    location = actor.get_location()
    extent = actor.bounding_box.extent
    yaw = actor.get_transform().rotation.yaw

    # Corner points relative to the actor's location, considering the extent
    corners = [
        (extent.x, extent.y),
        (-extent.x, extent.y),
        (-extent.x, -extent.y),
        (extent.x, -extent.y)
    ]

    # Rotate corners and translate to the actor's location, ensuring integer coordinates
    world_corners = []
    for corner in corners:
        rotated_corner = rotate_point_around_origin(corner[0], corner[1], yaw)
        world_corner = (int(rotated_corner[0] + location.x + 600), int(rotated_corner[1] + location.y + 400))
        world_corners.append(world_corner)

    # Draw the bounding box on the image
    num_corners = len(world_corners)
    for i in range(num_corners):
        cv2.line(img, world_corners[i], world_corners[(i + 1) % num_corners], color, 2)
    
def draw_world(world_model, my_vehicle_id, fps=None):
    img = np.zeros((800, 1200, 3), dtype=np.uint8)

    # Draw road waypoints
    draw_map(world_model.map_gt, img)

    # Draw traffic lights and their status
    draw_traffic_lights(img, world_model.world)

    # Draw all actors
    for actor in world_model.world.get_actors():
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('walker.pedestrian'):
            color = (255, 0, 0)  # Default color for other actors (blue)
            if actor.id == my_vehicle_id:
                color = (0, 255, 0)  # Color for the spawned vehicle (green)
            draw_actor_bounding_box(img, actor, color)

    # Display FPS on the OpenCV window
    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("CARLA World", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("CARLA World", cv2.WND_PROP_VISIBLE) < 1:
        cv2.destroyAllWindows()
        sys.exit()

def setup_birds_eye_camera(vehicle, config):
    # Get the world the vehicle is in
    world = vehicle.get_world()

    # Define the blueprint for the camera
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(config['image_size_x']))
    camera_bp.set_attribute('image_size_y', str(config['image_size_y']))
    camera_bp.set_attribute('fov', str(config['fov']))  # Field of view

    # Adjust the camera location and rotation to get a bird's-eye view
    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=config['z']), carla.Rotation(pitch=-90))

    # Spawn the camera and attach to the vehicle
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    return camera

def keep_latest_image(image, simulation_obj, attribute_name):
    # stores image to passed object

    # Convert the image from CARLA's image format to an OpenCV format
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Drop the alpha channel
    array = array[:, :, ::-1]  # Convert from BGR to RGB

    setattr(simulation_obj, attribute_name, array)
    #
    # image_container = array  # Store the latest image