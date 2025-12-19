#!/usr/bin/env python

import math
import sys
import numpy as np
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf_transformations

# Constants
LEADER_POSE_TOPIC = 'leader_odom'
FREQUENCY = 10  # Hz

USE_SIM_TIME = True

# Leader behavior constants
LINEAR_VELOCITY = 0.5  # m/s
ANGULAR_VELOCITY = math.pi/8  # rad/s
MIN_THRESHOLD_DISTANCE = 1.0  # m
MIN_SCAN_ANGLE_RAD = -15.0 / 180 * math.pi
MAX_SCAN_ANGLE_RAD = +15.0 / 180 * math.pi

class Goal:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

class Pose:
    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading

class Swarm(Node):
    pose = Pose()

    def __init__(self, robot_name, is_leader=False, node_name="swarm", context=None, goal=Goal(0,0)):
        super().__init__(node_name, context=context)

        # Use simulation time
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])

        # Store leader flag
        self.is_leader = is_leader

        # dynamic topic usage
        DEFAULT_CMD_VEL_TOPIC = f'{robot_name}/cmd_vel'
        DEFAULT_SCAN_TOPIC = f'{robot_name}/base_scan'
        DEFAULT_POSE_TOPIC = f'{robot_name}/ground_truth'

        # Publishers/subscribers
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self._laser_sub = self.create_subscription(LaserScan, DEFAULT_SCAN_TOPIC, self._laser_callback, 1)
        self._pose_sub = self.create_subscription(Odometry, DEFAULT_POSE_TOPIC, self._pose_callback, 1)
        
        # Leader-specific: publish pose to leader_odom topic
        if self.is_leader:
            self._pose_pub = self.create_publisher(Odometry, LEADER_POSE_TOPIC, 1)
        else:
            # Follower-specific: subscribe to leader pose
            self._leader_sub = self.create_subscription(Odometry, LEADER_POSE_TOPIC, self._leader_pose_callback, 1)

        # Constants for potential field (follower behavior)
        self._gamma = 1.2     # attractive force scaling
        self._alpha = 0.9     # repulsive force scaling
        self._beta = 2.0      # max distance for repulsion
        self._epsilon = 0.03  # buffer to avoid singularity
        self._d_safe = 0.8    # safety distance
        self._k = 1.6         # angular velocity scaling
        self._burger_radius = 0.089
        self._d_far = 2.5     # min distance for attraction force

        # Force vectors (follower behavior)
        self.F_attractive = [0.0, 0.0]
        self.F_repulsive = [0.0, 0.0]
        self._random_Force = [0.0, 0.0]
        self.Force = [0.0, 0.0]

        # Random force logic (follower behavior)
        self._last_position = (self.pose.x, self.pose.y)
        self._previous_movement_time = None
        self._stuck_threshold = 0.2        # meters
        self._stuck_time_threshold = 3.0   # seconds
        self._random_force_multiplier = 0.0
        self._random_force_mag = 100.0
        
        self.goal = goal

        # Leader behavior variables
        self._close_obstacle = False
        self._is_rotating = False
        self._rotation_start_time = None
        self._rotation_duration = 0.0
        self._rotation_direction = 1
        self.linear_velocity = LINEAR_VELOCITY
        self.angular_velocity = ANGULAR_VELOCITY
        self.min_threshold_distance = MIN_THRESHOLD_DISTANCE
        self.scan_angle = [MIN_SCAN_ANGLE_RAD, MAX_SCAN_ANGLE_RAD]

    def move(self, linear_vel, angular_vel):
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)

    def stop(self):
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def _pose_callback(self, msg):
        self.pose.x = msg.pose.pose.position.x
        self.pose.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.pose.heading = tf_transformations.euler_from_quaternion(quaternion)

    def _laser_callback(self, msg):
        if self.is_leader:
            # Leader laser callback - detect obstacles for random walk
            if not self._close_obstacle:
                min_index = int((self.scan_angle[0] - msg.angle_min) / (msg.angle_increment))
                max_index = int((self.scan_angle[1] - msg.angle_min) / (msg.angle_increment))
                desired_scan_area = msg.ranges[min_index:max_index]

                # find minimum distance in scanning area
                min_distance = self.min_threshold_distance + 1.0
                for distance in desired_scan_area:
                    if distance != float('inf') and distance < min_distance:
                        min_distance = distance

                # check if minimum distance in scanning area is below threshold
                if min_distance < self.min_threshold_distance:
                    self._close_obstacle = True
                else:
                    self._close_obstacle = False
        else:
            # Follower laser callback - calculate repulsive forces
            min_index = int(((-135.0/180*math.pi) - msg.angle_min) / msg.angle_increment)
            max_index = int(((+135.0/180*math.pi) - msg.angle_min) / msg.angle_increment)
            desired_scan_area = msg.ranges[min_index:max_index]

            self.F_repulsive = [0.0, 0.0]
            for i, distance in enumerate(desired_scan_area):
                # filter out leader position
                angle = (min_index + i) * msg.angle_increment + msg.angle_min + self.pose.heading
                world_x = self.pose.x + distance * np.cos(angle)
                world_y = self.pose.y + distance * np.sin(angle)

                distance_to_leader = np.sqrt((world_x - self.goal.x)**2 + (world_y - self.goal.y)**2)
                if distance_to_leader < 0.2:  # tolerance radius around leader
                    continue

                if (self._d_safe + self._epsilon) < distance < self._beta:
                    F_repulsive_mag = self._alpha / ((distance - self._d_safe) ** 2)
                elif distance < (self._d_safe + self._epsilon):
                    F_repulsive_mag = self._alpha / (self._epsilon ** 2)
                else:
                    F_repulsive_mag = 0.0

                self.F_repulsive[0] -= F_repulsive_mag * np.cos(angle)
                self.F_repulsive[1] -= F_repulsive_mag * np.sin(angle)

    def _leader_pose_callback(self, msg):
        """Follower receives leader pose"""
        self.goal.x = msg.pose.pose.position.x
        self.goal.y = msg.pose.pose.position.y

    def random_movement(self):
        """Follower behavior - detect if stuck and apply random force"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        distance_moved = np.sqrt((self.pose.x - self._last_position[0])**2 + 
                                (self.pose.y - self._last_position[1])**2)
        
        if distance_moved > self._stuck_threshold:
            # we are moving
            self._previous_movement_time = current_time
            self._last_position = (self.pose.x, self.pose.y)
            self._random_force_multiplier = 0.0
        else:
            # not moving
            if self._previous_movement_time is None:
                self._previous_movement_time = current_time

            total_time_stuck = current_time - self._previous_movement_time

            if total_time_stuck > self._stuck_time_threshold:
                self._random_force_multiplier = min((total_time_stuck - self._stuck_time_threshold)/2.0, 5.0)
                print(f"Stuck for {total_time_stuck:.1f}s. Applying random force (multiplier: {self._random_force_multiplier:.2f})")
            else:
                self._random_force_multiplier = 0.0

        # Generate random force if stuck
        if self._random_force_multiplier > 0.0:
            random_angle = random.uniform(0, 2*np.pi)
            random_magnitude = self._random_force_mag * self._random_force_multiplier
            self._random_Force = [random_magnitude * np.cos(random_angle), 
                                 random_magnitude * np.sin(random_angle)]
        else:
            self._random_Force = [0.0, 0.0]

    def leader_behavior(self):
        """Execute random walk behavior for leader"""
        if not self._close_obstacle:
            self.move(self.linear_velocity, 0.0)  # move forward if no object is detected
        else:  # start rotating due to obstacle detected
            self.stop()
            
            # set how long to rotate for
            if not self._is_rotating:
                self._rotation_duration = random.uniform(0.5, 3.5)
                self._rotation_start_time = self.get_clock().now()
                self._rotation_direction = random.choice([-1, 1])
                self._is_rotating = True

            elapsed_time = (self.get_clock().now() - self._rotation_start_time).nanoseconds / 1e9
            if elapsed_time < self._rotation_duration:  # if hasn't rotated for random time then keep moving
                self.move(0.0, self.angular_velocity * self._rotation_direction)
            else:  # finished rotating so stop movement
                self._close_obstacle = False
                self._is_rotating = False
                self.move(0.0, 0.0)

        # Publish Pose to leader_odom topic
        leader_odom_msg = Odometry()
        leader_odom_msg.header.stamp = self.get_clock().now().to_msg()
        leader_odom_msg.header.frame_id = "odom"

        # position
        leader_odom_msg.pose.pose.position.x = self.pose.x
        leader_odom_msg.pose.pose.position.y = self.pose.y

        # orientation
        leader_odom_msg.pose.pose.orientation.x = 0.0
        leader_odom_msg.pose.pose.orientation.y = 0.0
        leader_odom_msg.pose.pose.orientation.z = 0.0
        leader_odom_msg.pose.pose.orientation.w = 1.0

        self._pose_pub.publish(leader_odom_msg)

    def follower_behavior(self):
        """Execute potential field behavior for follower"""
        # Attractive force toward goal
        dx = self.goal.x - self.pose.x
        dy = self.goal.y - self.pose.y
        theta = np.arctan2(dy, dx)
        d_j0 = np.sqrt(dx**2 + dy**2) - (2*self._burger_radius)

        if d_j0 < self._d_safe:
            F_attractive_mag = self._alpha/(d_j0**2)
        elif self._d_safe < d_j0 and d_j0 < self._d_far:
            F_attractive_mag = 0
        else:
            F_attractive_mag = self._gamma * (d_j0**2)

        self.F_attractive = [F_attractive_mag*np.cos(theta), F_attractive_mag*np.sin(theta)]

        # check if robot is stuck and apply random force
        self.random_movement()

        # Total force = attractive + repulsive
        self.Force = [
            self.F_attractive[0] + self.F_repulsive[0] + self._random_Force[0],
            self.F_attractive[1] + self.F_repulsive[1] + self._random_Force[1]
        ]

        # Orientation difference
        Force_orientation = np.arctan2(self.Force[1], self.Force[0])
        orientation = Force_orientation - self.pose.heading
        normalized_orientation = np.arctan2(np.sin(orientation), np.cos(orientation))
        angular_vel = self._k * normalized_orientation

        # Linear velocity projection
        heading_vector = [np.cos(self.pose.heading), np.sin(self.pose.heading)]
        linear_vel = self.Force[0]*heading_vector[0] + self.Force[1]*heading_vector[1]

        # Clamp linear velocity
        linear_vel = max(0.0, min(linear_vel, 0.52))

        print(f"Pose=({self.pose.x:.2f},{self.pose.y:.2f}), Goal=({self.goal.x},{self.goal.y}), "
              f"Linear={linear_vel:.2f}, Angular={angular_vel:.2f}")

        self.move(linear_vel, angular_vel)

    def spin(self):
        while rclpy.ok():
            # Switch behavior based on is_leader flag
            if self.is_leader:
                self.leader_behavior()
            else:
                self.follower_behavior()

            rclpy.spin_once(self)

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 3:
        print("Usage: ros2 run swarm swarm <robot_name> <is_leader>")
        print("Example: ros2 run swarm swarm robot_0 true")
        print("Example: ros2 run swarm swarm robot_1 false")
        return
    
    robot_name = sys.argv[1]
    is_leader = sys.argv[2].lower() in ['true', '1', 'yes']

    swarm = Swarm(robot_name, is_leader=is_leader)

    print(f"Starting {robot_name} as {'LEADER' if is_leader else 'FOLLOWER'}")

    try:
        swarm.spin()
    except KeyboardInterrupt:
        swarm.get_logger().error("ROS node interrupted.")
    finally:
        if rclpy.ok():
            swarm.stop()
            rclpy.try_shutdown()

if __name__ == "__main__":
    main()