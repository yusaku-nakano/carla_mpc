import carla
import random
import numpy as np

from agents.navigation.local_planner import RoadOption

"""
PROBLEM: When GlobalRoutePlanner generates a path requiring a lane change, discontinuities occur in the curvature of the trajectory
ATTEMPTED SOLUTION: 
- Randomly generate a route
- Make sure that the Lane ID is consistent throughout the path (e.g. if on left lane, stay on left lane)
- If the lane changes, recursively generate new routes starting from the last valid waypoint before the lane change
- Check if the route length exceeds some minimum length. If the route is too short, continue generating from the last waypoint
"""

# Calculate the length of a route
def route_length(route):
    length = 0.0
    for i in range(1, len(route)):
        prev_wp = route[i-1][0].transform.location
        curr_wp = route[i][0].transform.location
        length += np.sqrt((prev_wp.x - curr_wp.x) ** 2 +
                          (prev_wp.y - curr_wp.y) ** 2)
    return length

# Function to check if lane change is allowed at a waypoint
def lane_change_allowed(waypoint):
    return waypoint.lane_change != carla.LaneChange.NONE

def validate_route(route, min_length, disconnect_threshold):
    if not route:
        raise ValueError("Route is empty!")
    
    total_length = 0.0
    for i in range(1, len(route)):
        prev_wp = route[i-1][0].transform.location
        curr_wp = route[i][0].transform.location

        # Calculate the distance between consecutive waypoints
        distance = np.sqrt((prev_wp.x - curr_wp.x) ** 2 +
                           (prev_wp.y - curr_wp.y) ** 2)
        
        # Check if the distance exceeds the disconnect threshold
        if distance > disconnect_threshold:
            raise Exception(f"Disconnected route segment detected between waypoint {i-1} and {i}. Distance: {distance:.2f} exceeds threshold: {disconnect_threshold:.2f}. Waypoint {i-1}: {prev_wp.x}, {prev_wp.y}, {prev_wp.z}. Waypoint {i}: {curr_wp.x}, {curr_wp.y}, {curr_wp.z}.")
        
        # Update total route length
        total_length += distance

        # Check for duplicate consecutive waypoints
        if prev_wp.x == curr_wp.x and prev_wp.y == curr_wp.y and prev_wp.z == curr_wp.z:
            raise Exception(f"Duplicate consecutive waypoints detected at index {i-1} and {i}. Waypoint: ({prev_wp.x}, {prev_wp.y}, {prev_wp.z})")

    # Check if the total route length is below the required minimum length
    if total_length < min_length:
        raise Exception(f"Total route length {total_length:.2f} is below the required minimum length {min_length:.2f}")

    print(f"Route validation successful. Total length: {total_length:.2f}")

def generate_route_simplified(carla_map, start_point, sampling_resolution, min_length):
    # Initialize the route and length
    route = []
    total_length = 0.0

    # Get the initial waypoint

    try:
        current_waypoint = carla_map.get_waypoint(start_point.location, lane_type=carla.LaneType.Driving)
    except AttributeError as e:
        current_waypoint = carla_map.get_waypoint(start_point, lane_type=carla.LaneType.Driving)

    # Generate waypoints along the route until the total length is reached
    while total_length < min_length:
        route.append((current_waypoint, RoadOption.LANEFOLLOW))

        # Sample the next waypoint based on sampling resolution
        next_waypoints = current_waypoint.next(sampling_resolution)
        if not next_waypoints:
            print("No more waypoints available; reached the end of the road.")
            break

        # Update to the next waypoint and accumulate the length
        next_waypoint = next_waypoints[0] # In case there are more than one, we choose the first one
        segment_length = current_waypoint.transform.location.distance(next_waypoint.transform.location)
        total_length += segment_length
        current_waypoint = next_waypoint

    print(f"Generated simplified route starting at {start_point}, consisting of {len(route)} waypoints and total distance of {total_length:.2f}m")
    return route, start_point

# Recursively generate a valid route (with no lane changes) that's at least "min_length" long
def generate_route(route_planner, map, start_point, branch_point, min_length, current_route=[], attempts=0, max_attempts=100, disconnect_threshold=5, ignore_destination=False):
    if ignore_destination:
        return generate_route_simplified(map, start_point, route_planner._sampling_resolution, min_length)

    # If we have over "max_attempts" attempts to generate a valid route, raise an exception
    if attempts >= max_attempts:
        raise RecursionError(f"Exceeded maximum attempts ({max_attempts}) while generating the route.")
    # Base case: if current_route is long enough, return the current_route
    total_length = route_length(current_route)

    
    # Generate route from current position and random endpoint
    end_point = random.choice(map.get_spawn_points())
    route_segment = route_planner.trace_route(branch_point.location, end_point.location)

    if not route_segment:
        print("Route couldn't be generated. Retrying...")
        new_point = random.choice(map.get_spawn_points())
        return generate_route(route_planner, map, new_point, new_point, min_length, current_route, attempts + 1)

    # The lane ID we want to maintain
    if len(current_route) == 0:
        lane_id = route_segment[0][0].lane_id
    else:
        lane_id = current_route[0][0].lane_id
    print(f"STARTING/BRANCH POINT: {route_segment[0][0].transform.location.x}, {route_segment[0][0].transform.location.y}, {route_segment[0][0].transform.location.z}")
    print(f"NEEDED LENGTH: {min_length}, LANE ID = {lane_id}")

    # Filter waypoints by lane and find where lane changes
    filtered_route = []
    for i, (waypoint, road_option) in enumerate(route_segment):
        curr_wp = waypoint.transform.location
        # Check for disconnections after a lane change
        if i == 0 and len(filtered_route) != len(current_route):
            prev_wp = current_route[-1][0].transform.location
            distance = np.sqrt((prev_wp.x - curr_wp.x)**2 + (prev_wp.y - curr_wp.y)**2)
            if distance > disconnect_threshold:
                print(f"Disconnected segment detected between waypoints. Distance: {distance}. Restarting...")
                remaining_length = min_length - route_length(current_route + filtered_route)
                return generate_route(route_planner, map, start_point, filtered_route[-1][0].transform, remaining_length, current_route + filtered_route, attempts + 1)
        # Check for disconnections with the previous waypoint (except for first waypoint)
        if len(filtered_route)>0:
            prev_wp = filtered_route[-1][0].transform.location
            distance = np.sqrt((prev_wp.x - curr_wp.x)**2 + (prev_wp.y - curr_wp.y)**2)
            if distance > disconnect_threshold:
                print(f"Disconnected segment detected between waypoints. Distance: {distance}. Restarting...")
                remaining_length = min_length - route_length(current_route + filtered_route)
                return generate_route(route_planner, map, start_point, filtered_route[-1][0].transform, remaining_length, current_route + filtered_route, attempts + 1)
        # Check for duplicate waypoints
        if len(filtered_route)>0:
            prev_wp = filtered_route[-1][0].transform.location
            if prev_wp == curr_wp:
                print(f"Duplicate waypoint detected at index {i}. Skipping waypoint: ({curr_wp.x}, {curr_wp.y}, {curr_wp.z})")
                continue  # Skip this waypoint as it's a duplicate
        if waypoint.is_intersection:
            # Allow lane ID = 0 in intersections
            print(f"Waypoint {i}: ({curr_wp.x}, {curr_wp.y}, {curr_wp.z}). Remaining length: {min_length - route_length(filtered_route)}. Lane ID: {waypoint.lane_id} (intersection_)")
            filtered_route.append((waypoint, road_option))
        elif waypoint.lane_id == lane_id:
            # If lane ID is the same, append that waypoint to the route
            print(f"Waypoint {i}: ({curr_wp.x}, {curr_wp.y}, {curr_wp.z}). Remaining length: {min_length - route_length(filtered_route)}. Lane ID: {waypoint.lane_id}")
            filtered_route.append((waypoint, road_option))
        else:
            print(f"Waypoint is intersection?: {waypoint.is_intersection}.")
            # If the current waypoint is NOT at an intersection or have the target lane_id...
            # AND if the the lane change happened at the first waypoint (after discarding the initial duplicate)
            remaining_length = min_length - route_length(filtered_route)
            if len(filtered_route) > 0 and waypoint.lane_id != lane_id:
                if remaining_length > 0:
                    # Generate a new route starting from the last valid waypoint
                    print(f"Lane change detected at waypoint {waypoint.transform.location}. Lane ID: {waypoint.lane_id}. Restarting from last valid waypoint with remaining length: {remaining_length}")
                    last_valid_wp = filtered_route[-1][0].transform
                    # Remove the last waypoint since we're starting there and we don't want duplicates
                    filtered_route.pop()
                    return generate_route(route_planner, map, start_point, last_valid_wp, remaining_length, current_route + filtered_route, attempts + 1)
                else:
                    # We already reached the desired length
                    print(f"Lane change detected at waypoint {waypoint.transform.location}. Lane ID: {waypoint.lane_id}. No recursion needed because the desired route length has been met. Remaining length: {remaining_length}")
                    break
            # Should not go through this...
            else:
                print(f"something went wrong. Waypoint {i}:{curr_wp.x}, {curr_wp.y}, {curr_wp.z}, with Lane ID: {waypoint.lane_id}")
    
    # Append filtered segment to current route
    current_route += filtered_route
    total_length = route_length(current_route)

    # Make sure route is long enough
    if total_length >= min_length:
        validate_route(current_route, min_length, disconnect_threshold)
        print(f"Route generation successful. Total route length: {total_length}")
        return current_route, start_point
    else: 
        # If route too short, generate more
        remaining_length = min_length - total_length
        print(f"Route too short. Current length: {total_length}. Remaining length: {remaining_length}. Generating more waypoints.")
        last_wp = current_route[-1][0].transform
        current_route.pop()
        return generate_route(route_planner, map, start_point, last_wp, remaining_length, current_route, attempts + 1)

def calculate_yaw(location1, location2):
    if isinstance(location1, carla.Waypoint):
        location1

    # Calculate the direction vector between two locations
    dx = location2.x - location1.x
    dy = location2.y - location1.y

    # Calculate the yaw angle in degrees (atan2 returns radians, convert to degrees)
    yaw = np.degrees(np.atan2(dy, dx))

    return yaw

def gen_route_manual(map, start_point):
    # Approach Junction
    route = []
    curr_wp = map.get_waypoint(start_point)
    while not curr_wp.is_junction:
        route.append(curr_wp)
        next_wps = curr_wp.next(2.0)
        if len(next_wps) == 0:
            break
        curr_wp = next_wps[0]
    
    if curr_wp.is_junction:
        route.append(curr_wp)
        possible_ways = curr_wp.next(10)
        if possible_ways:
            curr_wp = possible_ways[0]
            route.append(curr_wp)
    
    while curr_wp.is_junction:
        next_wps = curr_wp.next(2.0)
        if len(next_wps) == 0:
            break
        curr_wp = next_wps[0]
        route.append(curr_wp)
    
    for _ in range(5):
        next_wps = curr_wp.next(2.0)
        if len(next_wps) == 0:
            break
        curr_wp = next_wps[0]
        route.append(curr_wp)