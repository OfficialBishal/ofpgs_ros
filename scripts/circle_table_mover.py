#!/usr/bin/env python3
"""
Moves robot in circle around table for testing pose estimation.
"""

import math
import rospy
import sys
import tf

from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class CircleTableMover:
    """Moves the robot in a circle around the table using direct velocity commands."""
    
    def __init__(self, table_center_x=3.0, table_center_y=4.0, radius=1.5, num_waypoints=8, 
                 speed=0.3, angular_speed=1.0, distance_threshold=0.3, waypoint1_x=3.0, waypoint1_y=2.5):
        """
        Initialize the circle table mover.
        
        Args:
            table_center_x: X coordinate of table center (default: 3.0)
            table_center_y: Y coordinate of table center (default: 4.0)
            radius: Radius of the circle in meters (default: 1.5)
            num_waypoints: Number of waypoints around the circle (default: 8)
            speed: Linear speed in m/s (default: 0.3)
            angular_speed: Angular speed in rad/s (default: 1.0)
            distance_threshold: Distance threshold to consider waypoint reached (default: 0.3)
            waypoint1_x: X coordinate for waypoint 1 (default: 3.0)
            waypoint1_y: Y coordinate for waypoint 1 (default: 2.5)
        """
        self.table_center_x = table_center_x
        self.table_center_y = table_center_y
        self.radius = radius
        self.num_waypoints = num_waypoints
        self.speed = speed
        self.angular_speed = angular_speed
        self.distance_threshold = distance_threshold
        self.waypoint1_x = waypoint1_x
        self.waypoint1_y = waypoint1_y
        
        # Initialize velocity publisher (same as assignment 6)
        rospy.loginfo("Initializing velocity command publisher...")
        self.cmd_pub = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=10)
        
        # Initialize TF listener
        rospy.loginfo("Initializing TF listener...")
        self.tf_listener = tf.TransformListener()
        
        # Wait for TF to be ready
        rospy.loginfo("Waiting for TF transforms to be available...")
        try:
            self.tf_listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(10.0))
            rospy.loginfo("TF transforms available")
        except Exception as e:
            rospy.logwarn(f"TF transform wait timed out: {e}")
            rospy.logwarn("Continuing anyway...")
        
        # Calculate waypoints in a circle around the table, with waypoint 1 at specified position
        self.waypoints = self._calculate_circle_waypoints(waypoint1_x, waypoint1_y)
        rospy.loginfo(f"Created {len(self.waypoints)} waypoints around table at ({self.table_center_x}, {self.table_center_y})")
    
    def robot_position(self):
        """Get current robot position and orientation from TF."""
        try:
            self.tf_listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
            
            x = trans[0]
            y = trans[1]
            (roll, pitch, yaw) = euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
            
            return (x, y, yaw)
        except Exception as e:
            rospy.logwarn(f"Could not get robot position: {e}")
            return None
    
    def calculate_distance(self, waypoint):
        """Calculate distance from current position to a waypoint."""
        pos = self.robot_position()
        if pos is None:
            return float('inf')
        return math.sqrt((waypoint[0] - pos[0])**2 + (waypoint[1] - pos[1])**2)
    
    def calculate_angle(self, waypoint):
        """Calculate angle difference to face the waypoint."""
        pos = self.robot_position()
        if pos is None:
            return 0.0
        target_angle = math.atan2(waypoint[1] - pos[1], waypoint[0] - pos[0])
        angle_difference = target_angle - pos[2]
        # Normalize angle to [-pi, pi]
        return math.atan2(math.sin(angle_difference), math.cos(angle_difference))
    
    def calculate_angle_to_table(self):
        """Calculate angle difference to face the table center."""
        pos = self.robot_position()
        if pos is None:
            return 0.0
        target_angle = math.atan2(self.table_center_y - pos[1], self.table_center_x - pos[0])
        angle_difference = target_angle - pos[2]
        # Normalize angle to [-pi, pi]
        return math.atan2(math.sin(angle_difference), math.cos(angle_difference))
    
    def is_at_waypoint(self, waypoint):
        """Check if the robot is within threshold distance to the waypoint."""
        return self.calculate_distance(waypoint) < self.distance_threshold
    
    def stop_robot(self):
        """Stop the robot by publishing zero velocities."""
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.cmd_pub.publish(stop_twist)
    
    def _calculate_circle_waypoints(self, waypoint1_x=3.0, waypoint1_y=2.5):
        """
        Calculate waypoints in a circle around the table.
        Waypoint 1 is set to the specified position, then other waypoints are calculated
        in a circle around the table at equal intervals.
        Each waypoint faces the table center.
        
        Args:
            waypoint1_x: X coordinate for waypoint 1 (default: 3.0)
            waypoint1_y: Y coordinate for waypoint 1 (default: 2.5)
        
        Returns:
            List of tuples: [(x, y, yaw_deg), ...]
        """
        waypoints = []
        
        # Calculate the angle from table center to waypoint 1 position
        dx = waypoint1_x - self.table_center_x
        dy = waypoint1_y - self.table_center_y
        start_angle = math.atan2(dy, dx)
        
        # Calculate the actual distance from table center to waypoint 1
        actual_distance = math.sqrt(dx**2 + dy**2)
        
        # Adjust radius if waypoint 1 is not at the desired radius
        if abs(actual_distance - self.radius) > 0.1:
            rospy.logwarn(f"Waypoint 1 distance ({actual_distance:.2f}m) differs from desired radius ({self.radius:.2f}m)")
            rospy.logwarn(f"Using actual distance for waypoint 1, but {self.radius}m radius for other waypoints")
        
        # Waypoint 1: Use the specified position
        yaw_rad = math.atan2(self.table_center_y - waypoint1_y, self.table_center_x - waypoint1_x)
        yaw_deg = math.degrees(yaw_rad)
        waypoints.append((waypoint1_x, waypoint1_y, yaw_deg))
        rospy.loginfo(f"Waypoint 1: position=({waypoint1_x:.2f}, {waypoint1_y:.2f}), yaw={yaw_deg:.1f}°")
        
        # Calculate remaining waypoints in a circle
        for i in range(1, self.num_waypoints):
            # Calculate angle for this waypoint (in radians)
            # Start from start_angle and add equal intervals
            angle = start_angle + 2 * math.pi * i / self.num_waypoints
            
            # Calculate waypoint position
            x = self.table_center_x + self.radius * math.cos(angle)
            y = self.table_center_y + self.radius * math.sin(angle)
            
            # Calculate yaw angle to face the table center
            dx = self.table_center_x - x
            dy = self.table_center_y - y
            yaw_rad = math.atan2(dy, dx)
            yaw_deg = math.degrees(yaw_rad)
            
            waypoints.append((x, y, yaw_deg))
            
            rospy.loginfo(f"Waypoint {i+1}: position=({x:.2f}, {y:.2f}), yaw={yaw_deg:.1f}°")
        
        return waypoints
    
    def move_to_waypoint(self, x, y, timeout=60.0):
        """
        Move robot to a specific waypoint while always facing the table center.
        
        Args:
            x: X coordinate
            y: Y coordinate
            timeout: Maximum time to reach waypoint in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        rospy.loginfo(f"Moving to waypoint: ({x:.2f}, {y:.2f}) while facing table")
        
        waypoint = (x, y)
        start_time = rospy.get_time()
        rate = rospy.Rate(15)  # 15 Hz control loop
        
        while not rospy.is_shutdown():
            elapsed = rospy.get_time() - start_time
            if elapsed >= timeout:
                rospy.logwarn(f"Timeout waiting for waypoint ({x:.2f}, {y:.2f})")
                self.stop_robot()
                return False
            
            # Check if we've reached the waypoint
            if self.is_at_waypoint(waypoint):
                # Make sure we're facing the table before stopping
                table_ang_err = self.calculate_angle_to_table()
                if abs(table_ang_err) > 0.1:  # If not facing table, turn to face it
                    tw = Twist()
                    tw.angular.z = max(-self.angular_speed, min(self.angular_speed, 2.0 * table_ang_err))
                    tw.linear.x = 0.0
                    self.cmd_pub.publish(tw)
                    rate.sleep()
                    continue
                
                self.stop_robot()
                rospy.loginfo(f"Successfully reached waypoint: ({x:.2f}, {y:.2f})")
                return True
            
            # Calculate angle to waypoint (PRIMARY - must reach waypoint to follow circle)
            waypoint_ang_err = self.calculate_angle(waypoint)
            
            # Calculate angle to table center (SECONDARY - for viewing, but don't let it interfere)
            table_ang_err = self.calculate_angle_to_table()
            
            # Create velocity command
            tw = Twist()
            
            # Strategy: Move toward waypoint first, then adjust orientation to face table
            # Follow circular path, not straight to table
            
            # If angle error to waypoint is large, turn toward waypoint first
            if abs(waypoint_ang_err) > 0.25:  # Large angle error to waypoint
                # Turn toward waypoint to follow the circle
                tw.angular.z = max(-self.angular_speed, min(self.angular_speed, 2.0 * waypoint_ang_err))
                tw.linear.x = 0.0  # Don't move while turning toward waypoint
            else:
                # Angle to waypoint is acceptable, move forward toward waypoint
                tw.linear.x = self.speed
                
                # While moving, try to face table if it doesn't conflict too much
                # Only adjust for table if the waypoint angle error is small
                if abs(waypoint_ang_err) < 0.15:  # Close to waypoint direction
                    # Can afford to adjust orientation toward table
                    # Use mostly waypoint direction, but add some table-facing
                    combined_ang = 0.7 * waypoint_ang_err + 0.3 * table_ang_err
                else:
                    # Still need to correct toward waypoint
                    combined_ang = waypoint_ang_err
                
                tw.angular.z = max(-self.angular_speed, min(self.angular_speed, 1.5 * combined_ang))
            
            # Publish velocity command
            self.cmd_pub.publish(tw)
            
            rate.sleep()
        
        self.stop_robot()
        return False
    
    def find_closest_waypoint(self):
        """
        Find the waypoint closest to the robot's current position.
        
        Returns:
            tuple: (waypoint_index, waypoint) or (None, None) if no waypoints or position unavailable
        """
        if not self.waypoints:
            return None, None
        
        pos = self.robot_position()
        if pos is None:
            rospy.logwarn("Could not get robot position, using first waypoint")
            return 0, self.waypoints[0]
        
        min_distance = float('inf')
        closest_idx = 0
        closest_waypoint = self.waypoints[0]
        
        for i, (x, y, yaw_deg) in enumerate(self.waypoints):
            distance = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_waypoint = (x, y, yaw_deg)
        
        rospy.loginfo(f"Closest waypoint is #{closest_idx + 1} at ({closest_waypoint[0]:.2f}, {closest_waypoint[1]:.2f}), distance: {min_distance:.2f}m")
        return closest_idx, closest_waypoint
    
    def move_to_starting_position(self, timeout=60.0):
        """
        Move robot to the closest waypoint (starting position).
        
        Args:
            timeout: Maximum time to reach starting position in seconds
            
        Returns:
            tuple: (success: bool, starting_waypoint_index: int) or (False, None) if failed
        """
        if not self.waypoints:
            rospy.logerr("No waypoints available!")
            return False, None
        
        closest_idx, closest_waypoint = self.find_closest_waypoint()
        if closest_waypoint is None:
            rospy.logerr("Could not determine closest waypoint!")
            return False, None
        
        x, y, yaw_deg = closest_waypoint
        rospy.loginfo(f"Moving to starting position (waypoint #{closest_idx + 1}): ({x:.2f}, {y:.2f})")
        success = self.move_to_waypoint(x, y, timeout)
        return success, closest_idx
    
    def circle_table(self, num_loops=-1, wait_at_waypoint=5.0):
        """
        Move robot in a circle around the table.
        
        Args:
            num_loops: Number of times to complete the circle (-1 for infinite, default: -1)
            wait_at_waypoint: Time to wait at each waypoint in seconds (default: 5.0)
        """
        if num_loops == -1:
            rospy.loginfo(f"Starting to circle table: INFINITE loops, {len(self.waypoints)} waypoints per loop")
        else:
            rospy.loginfo(f"Starting to circle table: {num_loops} loop(s), {len(self.waypoints)} waypoints per loop")
        
        # First, move to starting position (closest waypoint)
        rospy.loginfo("=== Moving to starting position (closest waypoint) ===")
        success, start_idx = self.move_to_starting_position()
        if not success:
            rospy.logerr("Failed to reach starting position! Aborting.")
            return
        
        if start_idx is None:
            rospy.logerr("Could not determine starting waypoint index! Aborting.")
            return
        
        rospy.loginfo(f"Reached starting position (waypoint #{start_idx + 1})")
        rospy.sleep(1.0)  # Brief pause at starting position
        
        # Create waypoint sequence (only need to do this once, will repeat)
        if start_idx == 0:
            # If starting at waypoint 1, go through all waypoints and return to 1
            waypoint_indices = list(range(1, len(self.waypoints))) + [0]
        else:
            # Start from start_idx, go through all waypoints, and end at waypoint 1 (index 0)
            waypoint_indices = list(range(start_idx, len(self.waypoints))) + list(range(0, start_idx)) + [0]
        
        # Now circle the table continuously
        loop = 0
        first_loop = True
        
        while True:
            # Check if ROS is shutting down
            if rospy.is_shutdown():
                rospy.loginfo("Shutdown requested, stopping circle motion")
                break
            
            # Check if we've completed the requested number of loops
            if num_loops != -1 and loop >= num_loops:
                rospy.loginfo(f"Completed {num_loops} loop(s), stopping")
                break
            
            loop += 1
            if num_loops == -1:
                rospy.loginfo(f"=== Starting loop {loop} (continuous) ===")
            else:
                rospy.loginfo(f"=== Starting loop {loop}/{num_loops} ===")
            
            for waypoint_idx in waypoint_indices:
                # Check if ROS is shutting down
                if rospy.is_shutdown():
                    rospy.loginfo("Shutdown requested, stopping circle motion")
                    return
                
                x, y, yaw = self.waypoints[waypoint_idx]
                if num_loops == -1:
                    rospy.loginfo(f"--- Waypoint {waypoint_idx + 1}/{len(self.waypoints)} (Loop {loop}) ---")
                else:
                    rospy.loginfo(f"--- Waypoint {waypoint_idx + 1}/{len(self.waypoints)} (Loop {loop}/{num_loops}) ---")
                
                # Skip moving if this is the starting waypoint on the first loop (we're already there)
                if first_loop and waypoint_idx == start_idx and waypoint_idx == waypoint_indices[0]:
                    rospy.loginfo(f"Already at starting waypoint, waiting {wait_at_waypoint} seconds...")
                    rospy.sleep(wait_at_waypoint)
                    continue
                
                first_loop = False  # After first waypoint, we're no longer in first loop
                
                # Move to waypoint while always facing the table center
                success = self.move_to_waypoint(x, y)
                
                if success:
                    # Special handling for waypoint 1 (index 0) - ensure facing table
                    if waypoint_idx == 0:
                        rospy.loginfo("At waypoint 1, ensuring robot faces table center...")
                        # Turn to face table if not already facing it
                        table_ang_err = self.calculate_angle_to_table()
                        max_turn_time = 5.0  # Maximum time to turn
                        start_turn_time = rospy.get_time()
                        while abs(table_ang_err) > 0.1 and (rospy.get_time() - start_turn_time) < max_turn_time:
                            if rospy.is_shutdown():
                                return
                            tw = Twist()
                            tw.angular.z = max(-self.angular_speed, min(self.angular_speed, 2.0 * table_ang_err))
                            tw.linear.x = 0.0
                            self.cmd_pub.publish(tw)
                            rospy.sleep(0.1)
                            table_ang_err = self.calculate_angle_to_table()
                        self.stop_robot()
                        rospy.loginfo("Robot is facing table center")
                    
                    # Wait at waypoint to allow pose estimation to process
                    rospy.loginfo(f"Waiting {wait_at_waypoint} seconds at waypoint for pose estimation...")
                    rospy.sleep(wait_at_waypoint)
                else:
                    rospy.logwarn(f"Failed to reach waypoint {waypoint_idx + 1}, continuing to next waypoint...")
                    rospy.sleep(1.0)  # Brief pause before continuing
            
            if num_loops == -1:
                rospy.loginfo(f"=== Completed loop {loop} (continuing...) ===")
            else:
                rospy.loginfo(f"=== Completed loop {loop}/{num_loops} ===")
        
        rospy.loginfo("Finished circling table")


def main():
    """Main function."""
    rospy.init_node('circle_table_mover', anonymous=True)
    
    # Get parameters from ROS parameter server or use defaults
    table_center_x = rospy.get_param('~table_center_x', 3.0)
    table_center_y = rospy.get_param('~table_center_y', 4.0)
    radius = rospy.get_param('~radius', 1.5)
    num_waypoints = rospy.get_param('~num_waypoints', 8)
    num_loops = rospy.get_param('~num_loops', -1)  # -1 means infinite loops
    wait_at_waypoint = rospy.get_param('~wait_at_waypoint', 5.0)
    waypoint1_x = rospy.get_param('~waypoint1_x', 3.0)
    waypoint1_y = rospy.get_param('~waypoint1_y', 2.5)
    
    rospy.loginfo("Circle Table Mover Parameters:")
    rospy.loginfo(f"  Table center: ({table_center_x}, {table_center_y})")
    rospy.loginfo(f"  Circle radius: {radius} m")
    rospy.loginfo(f"  Number of waypoints: {num_waypoints}")
    rospy.loginfo(f"  Waypoint 1 position: ({waypoint1_x}, {waypoint1_y})")
    if num_loops == -1:
        rospy.loginfo(f"  Number of loops: INFINITE (continuous)")
    else:
        rospy.loginfo(f"  Number of loops: {num_loops}")
    rospy.loginfo(f"  Wait at waypoint: {wait_at_waypoint} s")
    
    # Wait for TF to be available
    rospy.loginfo("Waiting 3 seconds for system to initialize...")
    rospy.sleep(3.0)
    
    try:
        # Create mover and execute circle motion
        mover = CircleTableMover(
            table_center_x=table_center_x,
            table_center_y=table_center_y,
            radius=radius,
            num_waypoints=num_waypoints,
            waypoint1_x=waypoint1_x,
            waypoint1_y=waypoint1_y
        )
        
        mover.circle_table(num_loops=num_loops, wait_at_waypoint=wait_at_waypoint)
        
        rospy.loginfo("Circle table mover finished successfully")
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Circle table mover interrupted")
    except Exception as e:
        rospy.logerr(f"Error in circle table mover: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

