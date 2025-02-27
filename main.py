import carla
import random
import math
import numpy as np
from queue import Queue
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import xml.etree.ElementTree as ET
import traceback

# Camera Settings
BASELINE = 0.4
FOV = 90
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

# Stereo Matching Algorithms
# Choose either between "BM" or "SGBM"
STEREO_MATCHER = "SGBM"

# Depth Map Settings
NDISP_FACTOR = 6 # Adjust this; disparity search range for computing depth map
MIN_DISPARITY = 0
NUM_DISPARITIES = 16*NDISP_FACTOR - MIN_DISPARITY
BLOCKSIZE = 11
WINDOWSIZE = 6

# Motion Estimation Settings
# For each pixel with a depth LESS than S_LIMIT, we will compute the coordinate of that point in
# the camera coordinate system, and use those points to estimate the motion.
S_LIMIT = 100

# Display settings
SHOW_PLOTS = True
SHOW_LR_CAMERAS = True
SHOW_DEPTHS = True
VISUALIZE_MATCHES = True

# Plot Settings
PLOT_3D = True
PLOT_XY = True
PLOT_XZ = True
PLOT_YZ = True
NUM_PLOTS = sum([PLOT_3D, PLOT_XY, PLOT_XZ, PLOT_YZ])

# Initialize world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

#Spectator Camera
spectator = world.get_spectator()

def update_spectator(spectator):
    transform = vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90))
    )

# Initialize ego vehicle and sensors
spawn_point = random.choice(world.get_map().get_spawn_points())
bp = world.get_blueprint_library()
vehicle_bp = bp.filter('model3')[0]

# Spawn vehicle and sensors
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
# vehicle.set_autopilot(True)

# Append vehicle to list of actors
actors = []
actors.append(vehicle)

# Add left and right cameras
camera_bp = bp.find('sensor.camera.rgb')

camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
camera_bp.set_attribute("fov", str(FOV))

# camera_bp.set_attribute('fov', str(FOV))
left_camera_transform = carla.Transform(carla.Location(x=2, y=-BASELINE/2, z=1.4))
right_camera_transform = carla.Transform(carla.Location(x=2, y=BASELINE/2, z=1.4))
left_camera = world.spawn_actor(camera_bp, left_camera_transform, attach_to=vehicle)
right_camera = world.spawn_actor(camera_bp, right_camera_transform, attach_to=vehicle)

# Append cameras to list of actors
actors.append(left_camera)
actors.append(right_camera)

# Build the K Projection Matrix
# K = [[Fx 0 image_width/2],
#      [0 Fy image_height/2],
#      [0 0 1]]
# Fx and Fy are the same because the pixel aspect ratio is 1

image_width = camera_bp.get_attribute("image_size_x").as_int()
image_height = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()
focal = image_width / (2.0 * np.tan(fov * np.pi / 360.0))

K = np.array([[focal, 0,     image_width / 2.0],
              [0,     -focal, image_height / 2.0],
              [0,     0,     1]])

# sensor_queue = Queue()
# def process_image(sensor_data, sensor_name, sensor_queue):
#     frame = np.array(sensor_data.raw_data, dtype=np.uint8)
#     frame = np.reshape(frame, (sensor_data.height, sensor_data.width, 4))[...,:3]
#     sensor_queue.put((frame, sensor_name))

camera_data = {'left': np.zeros((image_height, image_width, 3), dtype=np.uint8),
               'right': np.zeros((image_height, image_width, 3), dtype=np.uint8)}

def l_camera_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    data_dict['left'] = array[..., :3]  # Keep only RGB channels

def r_camera_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    data_dict['right'] = array[..., :3]  # Keep only RGB channels

left_camera.listen(lambda image: l_camera_callback(image, camera_data))
right_camera.listen(lambda image: r_camera_callback(image, camera_data))

def compute_disparity(img_left, img_right):
    """
    Compute disparity between left and right images

    numDisparities: number of disparities to search
        since numDisparities is set to 96, the maximum possible disparity is 96
    blockSize: size of the block used for matching.
        A larger block size can smooth out noise but may lose fine details
    P1 and P2: parameters used in the disparity transform
        P1: penalty for small disparity changes
        P2: penalty for large disparity changes
    mode: mode of the disparity transform
    """
    img_left_grayscale = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_grayscale = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if STEREO_MATCHER == "BM":
        stereo = cv2.StereoBM.create(numDisparities=16*NDISP_FACTOR, blockSize=15)
    elif STEREO_MATCHER == "SGBM":
        stereo = cv2.StereoSGBM_create(minDisparity = MIN_DISPARITY,
                                       numDisparities = NUM_DISPARITIES,
                                       blockSize = BLOCKSIZE,
                                       P1 = 8*3*WINDOWSIZE**2,
                                       P2 = 32*3*WINDOWSIZE**2,
                                       disp12MaxDiff = 1,
                                       uniquenessRatio = 15,
                                       speckleWindowSize = 0,
                                       speckleRange = 2,
                                       preFilterCap = 63,
                                       mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity = stereo.compute(img_left_grayscale, img_right_grayscale).astype(np.float32)/16
    # .compute method caculates the disparity map for the left image using the SGBM matcher
    # The result is divided by 16 to convert it from fixed-point (used by CV2) to floating point

    return disparity

def calculate_depth(img_left, img_right, K):
    """
    Computes a depth map from the disparity

    focal: focal length of the camera
    baseline: distance between the left and right cameras

    Depth formula: depth = focal_length * baseline / disparity
    """
    disparity = compute_disparity(img_left, img_right)
    focal = K[0,0]

    # Replace all instances of 0 and -1 with a small minimum value (to avoid div by 0 or negatives)
    disparity[disparity == 0] = 0.0001
    disparity[disparity == -1] = 0.0001

    # Initialize the depth map as an array of ones with the same shape as the disparity map
    # data type is np.single (32-bit floating point)
    depth = np.ones(disparity.shape, np.single)
    depth = (focal * BASELINE) / disparity
    return depth

def extract_features(image):
    """
    Extracts keypoints and descriptors from an image

    ORB: Oriented FAST and Rotated BRIEF feature detector and descriptor
    """
    # ORB detector initialized usinng ORB_create
    orb = cv2.ORB_create(nfeatures=1500)

    # Find the keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """
    Matches features between two images
    
    BFMatcher: Brute Force Matcher
    """
    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.knnMatch(descriptors1, descriptors2, k=2)
    return matches

def filter_matches_by_distance(matches, dist_threshold):
    """
    Filters matches by distance

    matches: list of matched features from two images
    dist_threshold: maximum allowed relative distance betwween the two matches: (0, 1)

    filtered_matches: list of good matches, satisfying the distance threshold
    """
    filtered_matches = [match1 for match1, match2 in matches if match1.distance < (dist_threshold * match2.distance)]
    # filtered_match = []
    # for i, (match1, match2) in enumerate(matches):
    #     if match1.distance < dist_threshold:
    #         filtered_match.append(match1)

    return filtered_matches

def estimate_motion(matches, keypoints1, keypoints2, K, depth=None):
    """
    Estimates the relative motion (rotation and translation) between two consecutive frames
    using matched keypoints and optionally depth information
    
    Inputs:
    - matches: a list of matches between keypoints in the two images
    - keypoints1: keypoints in the first image
    - keypoints2: keypoints in the second image
    - K: camera intrinsic matrix
    - depth: depth map for the first image (optional)
    
    Outputs:
    - R: rotation matrix
    - T: translation vector
    """
    # image_1_points and image_2_points are lists of 2D points in the first and second images, respectively
    # object_points is a list of 3D points in the world coordinate system
    image_1_points = []
    image_2_points = []
    object_points = []

    for m in matches:
        # Extract coordinates (u1, v1) of the keypoint in the first image
        # Extract coordinates (u2, v2) of the keypoint in the second image
        # .queryIdx: index of the keypoint in the first image
        # .trainIdx: index of the keypoint in the second image
        u1, v1 = keypoints1[m.queryIdx].pt
        u2, v2 = keypoints2[m.trainIdx].pt

        # Retrieve the depth value "s" at coordinates (u1, v1) from the depth map
        s = depth[int(v1), int(u1)]

        # If the depth value "s" is LESS than 1000:
        if s < S_LIMIT:
            # Compute the 3D point in the camera coordinate system using the formula:
            # p_c = K^{-1} * (s * [u1, v1, 1]^T)
            p_c = np.linalg.inv(K) @ (s * np.array([u1, v1, 1]))

            # Append the 2D points and the computed 3D point to their respective lists
            image_1_points.append(np.array([u1, v1]))
            image_2_points.append(np.array([u2, v2]))
            object_points.append(p_c)

    # Estimate the relative motion between the two images using RANSAC
    # Stack the 3D points into a single NumPy array
    object_points = np.vstack(object_points)
    # Convert the 2D points into a single NumPy array
    image_points = np.array(image_2_points)
    # Estimate the rotation (R_vec) and translation (T) between the two frames using 
    # the PnP algorithm with RANSAC
    _, R_vec, T, _ = cv2.solvePnPRansac(object_points, image_points, K, None)
    # Using Rodrigues' formula, convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(R_vec)

    return R, T

def update_trajectory(prev_trajectory, prev_RT, matches, keypoints1, keypoints2, K, depth=None):
    # Extract each column of prev_trajectory and store it as a list of 3D points
    # Example:
    # prev_trajectory = np.array([
    #                            [1, 2, 3, 4],   # x-coordinates
    #                            [5, 6, 7, 8],   # y-coordinates
    #                            [9, 10, 11, 12] # z-coordinates
    #                            ])
    # trajectory = np.array([
    #                       [1, 5, 9],  
    #                       [2, 6, 10],  
    #                       [3, 7, 11],  
    #                       [4, 8, 12] 
    #                       )
    trajectory = [prev_trajectory[:,i] for i in range(prev_trajectory.shape[-1])]

    # Estimate the relative motion (rotation and translation) between two frames
    R, T = estimate_motion(matches, keypoints1, keypoints2, K, depth=depth)
    # R = 3x3, T = 3x1

    # Extrinsic matrix from previous frame (camera) to world frame
    RT_prevframe_world = prev_RT

    # Combine the rotation matrix (R) and translation vector (T) into a 4x4 transformation matrix
    # Extrinsic matrix from previous frame (camera) to current frame (camera)
    RT_prevframe_currframe = np.eye(4)
    RT_prevframe_currframe = np.hstack([R, T])
    RT_prevframe_currframe = np.vstack([RT_prevframe_currframe, np.array([0, 0, 0, 1])])

    # Extrinsic matrix from current frame (camera) to world frame
    RT_currframe_world = RT_prevframe_world @ np.linalg.inv(RT_prevframe_currframe)

    # Calculate current camera position from origin
    new_position = RT_currframe_world[:3,3]
    
    new_trajectory = np.array([
        new_position[2], 
        new_position[0],
        new_position[1]  
    ])

    # Append the new 3D point to the trajectory
    trajectory.append(new_trajectory[0:3])
    # print("Trajectory: ", trajectory)

    # Convert the trajectory from a list of 3D points to a NumPy array
    trajectory = np.array(trajectory).T

    return trajectory, RT_currframe_world

def plot_trajectory(ax1, ax2, ax3, ax4, fig, trajectory_gt, trajectory_est):
    plt.cla()
    
    if PLOT_3D:
        ax1.clear()
        ax1.plot3D(trajectory_est[0, :], trajectory_est[1, :], trajectory_est[2, :], 'r', label = "Estimated")
        ax1.scatter3D(trajectory_est[0, :], trajectory_est[1, :], trajectory_est[2, :], color = "r")
        ax1.plot3D(trajectory_gt[0, :], trajectory_gt[1, :], trajectory_gt[2, :], 'b', label = "Ground Truth")
        ax1.scatter3D(trajectory_gt[0, :], trajectory_gt[1, :], trajectory_gt[2, :], color = "b")
        ax1.set_title("3D")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()
    if PLOT_XY:
        ax2.clear()
        ax2.plot(trajectory_est[0, :], trajectory_est[1, :], 'r', label = "Estimated")
        ax2.scatter(trajectory_est[0, :], trajectory_est[1, :], color = "r")
        ax2.plot(trajectory_gt[0, :], trajectory_gt[1, :], 'b', label = "Ground Truth")
        ax2.scatter(trajectory_gt[0, :], trajectory_gt[1, :], color = "b")
        ax2.set_title("XY")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.legend()
    if PLOT_XZ:
        ax3.clear()
        ax3.plot(trajectory_est[0, :], trajectory_est[2, :], 'r', label = "Estimated")
        ax3.scatter(trajectory_est[0, :], trajectory_est[2, :], color = "r")
        ax3.plot(trajectory_gt[0, :], trajectory_gt[2, :], 'b', label = "Ground Truth")
        ax3.scatter(trajectory_gt[0, :], trajectory_gt[2, :], color = "b")
        ax3.set_title("XZ")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Z")
        ax3.legend()
    if PLOT_YZ:
        ax4.clear()
        ax4.plot(trajectory_est[1, :], trajectory_est[2, :], 'r', label = "Estimated")
        ax4.scatter(trajectory_est[1, :], trajectory_est[2, :], color = "r")
        ax4.plot(trajectory_gt[1, :], trajectory_gt[2, :], 'b', label = "Ground Truth")
        ax4.scatter(trajectory_gt[1, :], trajectory_gt[2, :], color = "b")
        ax4.set_title("YZ")
        ax4.set_xlabel("Y")
        ax4.set_ylabel("Z")
        ax4.legend()

    fig.canvas.draw()
    fig_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow("Figure", cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def main():
    try:
        waypoint = world.get_map().get_waypoint(spawn_point.location)

        # Transformation matrix from world coordinates to camera coordinates
        world2camera = np.array(left_camera.get_transform().get_inverse_matrix())

        # Ground truth (actual) trajectory
        trajectory_gt = left_camera.get_location()
        trajectory_gt = np.array([trajectory_gt.x, trajectory_gt.y, trajectory_gt.z]).reshape((3,1))

        # Estimated trajectory from visual odometry
        trajectory_est = np.array([0, 0, 0]).reshape((3,1))
        R = np.diag([1, 1, 1])
        T = trajectory_est

        # RT: 4x4 transformaton matrix combining rotation (R) and translation (T)
        RT = np.hstack([R, T])
        RT = np.vstack([RT, np.array([0, 0, 0, 1])])

        # Initialize old_frame to store the previous frame for feature matching
        old_frame = None

        # Initialize a matplotlib figure with four subplots
        # ax1: 3D plot for visualizing the trajectory
        # ax2: 2D plot for the XY-plane
        # ax3: 2D plot for the XZ-plane
        # ax4: 2D plot for the YZ-plane
        if NUM_PLOTS > 0:
            fig = plt.figure(figsize=(15,4))
        else:
            fig = plt.figure()
        
        ax1, ax2, ax3, ax4 = None, None, None, None
        plot_counter = 1
        if PLOT_3D:
            ax1 = fig.add_subplot(1, NUM_PLOTS, plot_counter, projection='3d')
            plot_counter += 1
        if PLOT_XY:
            ax2 = fig.add_subplot(1, NUM_PLOTS, plot_counter)
            ax2.set_aspect('equal', adjustable='box')
            plot_counter += 1
        if PLOT_XZ:
            ax3 = fig.add_subplot(1, NUM_PLOTS, plot_counter)
            ax3.set_aspect('equal', adjustable='box')
            plot_counter += 1
        if PLOT_YZ:
            ax4 = fig.add_subplot(1, NUM_PLOTS, plot_counter)
            ax4.set_aspect('equal', adjustable='box')

        left_frame = None
        right_frame = None
        curr_frame = None

        # Main loop
        while True:
            world.tick()
            waypoint = np.random.choice(waypoint.next(1))
            vehicle.set_transform(waypoint.transform)
            # Current ground truth location
            trajectory_gt_curr = left_camera.get_location()
            trajectory_gt = np.hstack([trajectory_gt, 
                                       np.array([trajectory_gt_curr.x, 
                                                 trajectory_gt_curr.y, 
                                                 trajectory_gt_curr.z]).reshape((3,1))])

            # for _ in range(2):
            #     s_frame = sensor_queue.get(True, 1.0)
            #     if s_frame[1] == "left":
            #         left_frame = s_frame[0]
            #     elif s_frame[1] == "right":
            #         right_frame = s_frame[0]
            left_frame = camera_data['left'].copy()
            right_frame = camera_data['right'].copy()
            # If both frames are available:
            if (left_frame is not None) and (right_frame is not None):
                curr_frame = left_frame.copy()
                if SHOW_LR_CAMERAS:
                    cv2.imshow("Left Frame", left_frame)
                    cv2.imshow("Right Frame", right_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows() 
                        break
                depth = calculate_depth(left_frame, right_frame, K)
            if old_frame is not None:
                keypoints1, descriptors1 = extract_features(old_frame)
                keypoints2, descriptors2 = extract_features(curr_frame)
                matches = match_features(descriptors1, descriptors2)
                matches = filter_matches_by_distance(matches, dist_threshold=0.6)

                if VISUALIZE_MATCHES:
                    # Match features of current frame and previous frame of Left Camera
                    image_matches = cv2.drawMatches(old_frame, keypoints1, curr_frame, keypoints2, matches, None)

                    scale_percent = 70 # Resize to scale_percent% of original size
                    im_width = int(image_matches.shape[1] * scale_percent / 100)
                    im_height = int(image_matches.shape[0] * scale_percent / 100)

                    resized_image_matches = cv2.resize(image_matches, (im_width, im_height), interpolation=cv2.INTER_AREA)
                    cv2.imshow("Matches", resized_image_matches)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

                # Update trajectory
                trajectory_est, RT = update_trajectory(trajectory_est, RT, matches, keypoints1, keypoints2, K, depth=depth)

                # Map the ground truth trajectory to the camera coordinate system
                # 1. Append a row of 1s to the ground truth trajectory to make it homogeneous
                trajectory_gt_world = np.append(trajectory_gt, np.ones((1, trajectory_gt.shape[1])), axis=0)
                # 2. Multiply by world2camera to transform the ground truth trajectory to the camera coordinate system
                trajectory_gt_camera = world2camera @ trajectory_gt_world
                # 3. Rearrange the coordinates to match the camera's coordinate system
                trajectory_gt_camera = np.array([trajectory_gt_camera[0], 
                                                 trajectory_gt_camera[1], 
                                                 trajectory_gt_camera[2]])
                
                # Plot trajectory
                if SHOW_PLOTS:
                    plot_trajectory(ax1, ax2, ax3, ax4, fig, trajectory_gt_camera, trajectory_est)

                # Normalize the depth map for visualization and display it
                depth = ((depth - depth.mean()) / depth.std()) * 255
                depth = depth.astype(np.uint8)
                
                # Show depth map
                if SHOW_DEPTHS:
                    cv2.imshow("Depth", depth)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
            # Update old_frame with the current frame for the next iteration
            if curr_frame is not None:
                old_frame = curr_frame.copy()
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
    finally:
        # Destroy all actors
        for actor in actors:
            actor.destroy()
                
main()
