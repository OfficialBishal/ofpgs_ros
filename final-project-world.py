#!/usr/bin/env python3
"""
Final Project World - Isaac Sim World Setup

Creates an Isaac Sim world with:
- HSR robot positioned to face a table
- Coffee table with configurable objects on top
- Ground truth pose publisher for pose estimation evaluation
- Bounding box visualization for all objects
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import ast
import os
import sys
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from rospkg import RosPack
from scipy.spatial.transform import Rotation

# ROS package paths
rp = RosPack()
usd_repo_path = rp.get_path('usd')
template_repo_path = rp.get_path('hsr-omniverse')
template_world_repo_path = rp.get_path('rc_isaac_worlds')

# Add paths to sys.path (order matters)
sys.path.append(template_world_repo_path)
sys.path.append(template_repo_path)

# Import Isaac Sim world (must be first to initialize sim_app)
from robocanes_isaac_world import (
    robocanes_isaac_world, sim_app, Usd, Gf, omni, prims, 
    euler_angles_to_quat, RigidPrim
)

from pxr import UsdGeom

# Lazy import for physx_utils (may not be available immediately)
try:
    from omni.physx.scripts import utils as physx_utils
except (ImportError, ModuleNotFoundError):
    physx_utils = None

# Import HSR robot (after Isaac Sim is initialized)
import robocanes_hsr
from isaac_robot_behavior_start import isaac_robot_behavior_start
from isaac_robot_pose_pub import isaac_robot_pose_pub

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Final Project World Setup")
parser.add_argument(
    "--robot_spawn_pos_xyz", 
    type=str, 
    default='[3, 2.5, 0.01]', 
    # default='[3, 3.35, 0.01]', 
    help="Robot spawn position as xyz list."
)
parser.add_argument(
    "--robot_spawn_orient_xyz", 
    type=str, 
    default='[0, 0, 0]', 
    help="Robot spawn orientation as xyz list (will be overridden to face table)."
)
args, unknown_args = parser.parse_known_args()

# World constants
TABLE_POSITION = [3, 4]  # Grid coordinates (x, y)
TABLE_HEIGHT = 0.1  # Table base height in meters
OBJECT_HEIGHT_ABOVE_TABLE = 0.5  # Object height above table (when upright)
OBJECT_LYING_HEIGHT = 0.05  # Height when lying flat
# HEAD_TILT_ANGLE = -0.8  # Head tilt angle in radians (negative = down)
HEAD_TILT_ANGLE = -0.4  # Head tilt angle in radians (negative = down)

ARM_ROLL_JOINT_ANGLE = -1.57  # Arm roll joint angle in radians
WRIST_FLEX_JOINT_ANGLE = -1.57  # Wrist flex joint angle in radians

# Object configuration
OBJECT_NAME = "cracker_box"  # Primary object for pose estimation
ADDITIONAL_OBJECTS = [
    {"name": "mustard_bottle", "grid_x": 2.85, "grid_y": 3.85, 
     "orientation": [0, 0, 0], "lying": False}
]

# ============================================================================
# MAIN WORLD CLASS
# ============================================================================

class final_project_world_2(robocanes_isaac_world):
    """
    Final Project World
    
    Creates a world with HSR robot, table, and configurable objects.
    Robot is automatically oriented to face the table.
    """
    
    def __init__(self):
        """Initialize the world and add all objects."""
        print('>>> [WORLD] Initializing world...')
        super().__init__()
        
        # Store configuration
        self.object_name = OBJECT_NAME
        self.object_prims = {}  # Dictionary to store all object prims
        self.robot_spawn_position = None
        self._pending_head_tilt = None
        self._pending_arm_neutral = None
        
        # Setup ROS
        self._setup_ros_publisher()
        
        # Calculate robot spawn position and orientation
        robot_spawn_position = ast.literal_eval(args.robot_spawn_pos_xyz)
        self.robot_spawn_position = robot_spawn_position
        robot_spawn_orientation = self._calculate_robot_orientation_to_table(
            robot_spawn_position, TABLE_POSITION
        )
        
        # Add objects to world
        self._add_robot(robot_spawn_position, robot_spawn_orientation)
        self._add_table()
        self._add_main_object()
        self._add_additional_objects()
        
        print('>>> [WORLD] World initialization complete')
    
    # ========================================================================
    # ROS Setup
    # ========================================================================
    
    def _setup_ros_publisher(self):
        """Setup ROS publishers for ground truth pose and bounding boxes."""
        # Initialize ROS node if needed
        try:
            if not rospy.get_node_uri():
                rospy.init_node('final_project_world', anonymous=True, disable_signals=True)
        except Exception as e:
            print(f">>> [WORLD] ERROR initializing ROS node: {e}")
            return
        
        # Create ground truth pose publisher for main object
        gt_topic = f'/{self.object_name}/ground_truth_pose'
        self.object_gt_pub = rospy.Publisher(
            gt_topic, PoseStamped, queue_size=10, latch=True
        )
        print(f">>> [WORLD] Created ground truth publisher: {gt_topic}")
        
        # Initialize dictionary for ground truth pose publishers for additional objects
        self.object_gt_pubs = {}
    
    def _create_gt_publisher_for_object(self, object_name):
        """Create ground truth pose publisher for additional objects (like mustard)."""
        if object_name == self.object_name:
            return  # Main object already has publisher
        
        if object_name in self.object_gt_pubs:
            return  # Already exists
        
        try:
            gt_topic = f'/{object_name}/ground_truth_pose'
            self.object_gt_pubs[object_name] = rospy.Publisher(
                gt_topic, PoseStamped, queue_size=10, latch=True
            )
            print(f">>> [WORLD] Created ground truth publisher: {gt_topic}")
        except Exception as e:
            print(f">>> [WORLD] ERROR creating ground truth publisher for {object_name}: {e}")
    
    # ========================================================================
    # World Setup
    # ========================================================================
    
    def _add_robot(self, position, orientation):
        """Add HSR robot to the world."""
        print('>>> [WORLD] Adding HSR robot...')
        self.hsr_instance = robocanes_hsr.hsr(
            prefix='/hsrb',
            spawn_config={
                'translation': position,
                'orientation': orientation,
                'scale': [1, 1, 1]
            }
        )
        self.hsr_instance.onsimulationstart(self.sim_world)
        
        # Setup ROS publishers for robot
        self.isaac_robot_behavior_start = isaac_robot_behavior_start()
        self.isaac_robot_pose_pub = isaac_robot_pose_pub()
        
        # Set initial robot pose (will be set after simulation starts if articulation not available)
        print(f">>> [WORLD] Attempting to set head tilt to {HEAD_TILT_ANGLE:.3f} rad...")
        if not self.set_head_tilt_position(HEAD_TILT_ANGLE):
            print(f">>> [WORLD] Head tilt will be set after simulation starts (articulation not yet available)")
        print(f">>> [WORLD] Attempting to set arm neutral position...")
        if not self.set_arm_neutral_position():
            print(f">>> [WORLD] Arm neutral position will be set after simulation starts")
        print('>>> [WORLD] HSR robot added')
    
    def _add_table(self):
        """Add coffee table to the world."""
        print('>>> [WORLD] Adding table...')
        table_usd = os.path.join(usd_repo_path, 'robocanes_lab', 'robocanes_lab', 'coffeeTable.usd')
        
        table_prim = prims.create_prim(
            prim_path='/World/Props/coffee_table',
            usd_path=table_usd,
            translation=[TABLE_POSITION[0], TABLE_POSITION[1], TABLE_HEIGHT],
            orientation=euler_angles_to_quat([0.0, 0.0, 0.0]),
            scale=[1, 1, 1],
            semantic_label='coffee_table'
        )
        
        # Enable collision
        global physx_utils
        if physx_utils is None:
            from omni.physx.scripts import utils
            physx_utils = utils
        physx_utils.setCollider(prim=table_prim, approximationShape='sdfMesh')
        
        # Make table kinematic (non-movable)
        try:
            from pxr import UsdPhysics
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(table_prim)
            if rigid_body_api:
                rigid_body_api.CreateKinematicEnabledAttr(True)
        except Exception as e:
            print(f'>>> [WORLD] Warning: Could not set table as kinematic: {e}')
        
        self.table_prim = table_prim
        print('>>> [WORLD] Table added')
    
    def _add_main_object(self):
        """Add main object (primary object for pose estimation)."""
        print(f'>>> [WORLD] Adding {self.object_name}...')
        object_orientation = self._calculate_object_orientation_to_robot(
            TABLE_POSITION, self.robot_spawn_position
        )
        self.add_object_on_table(
            TABLE_POSITION[0], TABLE_POSITION[1]-0.2, 
            self.object_name, object_orientation
        )
        print(f'>>> [WORLD] {self.object_name} added')
    
    def _add_additional_objects(self):
        """Add additional objects for testing."""
        if not ADDITIONAL_OBJECTS:
            return
        
        for obj_config in ADDITIONAL_OBJECTS:
            obj_name = obj_config["name"]
            obj_grid_x = obj_config.get("grid_x", TABLE_POSITION[0] + 0.2)
            obj_grid_y = obj_config.get("grid_y", TABLE_POSITION[1])
            obj_orientation = obj_config.get("orientation", [0.0, 0.0, 0.0])
            
            print(f'>>> [WORLD] Adding {obj_name}...')
            self.add_object_on_table(obj_grid_x, obj_grid_y, obj_name, obj_orientation)
            print(f'>>> [WORLD] {obj_name} added')
    
    # ========================================================================
    # Object Placement
    # ========================================================================
    
    def add_object_on_table(self, grid_x, grid_y, object_name, orientation=None):
        """
        Add object on top of the table.
        
        Args:
            grid_x: Grid X coordinate
            grid_y: Grid Y coordinate
            object_name: Object name (e.g., "mustard_bottle", "cracker_box")
            orientation: Optional Euler angles [roll, pitch, yaw] in radians
        """
        # Map object names to YCB IDs
        ycb_id_map = {
            'mustard_bottle': '006_mustard_bottle',
            'cracker_box': '003_cracker_box',
        }
        
        ycb_id = ycb_id_map.get(object_name)
        if not ycb_id:
            print(f'>>> [WORLD] Warning: Unknown object "{object_name}"')
            return
        
        # Calculate object position
        object_pos = [grid_x, grid_y, TABLE_HEIGHT + OBJECT_HEIGHT_ABOVE_TABLE]
        
        # Get object USD path
        object_usd = os.path.join(
            usd_repo_path, 'ycb', ycb_id, 
            'google_16k_converted', 'textured_obj.usd'
        )
        
        if not os.path.exists(object_usd):
            print(f'>>> [WORLD] Warning: {object_name} USD not found at {object_usd}')
            return
        
        # Use provided orientation or default
        if orientation is None:
            orientation = [0.0, 0.0, 0.0]
        
        # Create object prim
        object_prim = prims.create_prim(
            prim_path=f'/World/Props/{object_name}',
            usd_path=object_usd,
            translation=object_pos,
            orientation=euler_angles_to_quat(orientation),
            scale=[1, 1, 1],
            semantic_label=ycb_id
        )
        
        # Enable collision and physics
        global physx_utils
        if physx_utils is None:
            from omni.physx.scripts import utils
            physx_utils = utils
        physx_utils.setCollider(prim=object_prim, approximationShape='sdfMesh')
        RigidPrim(prim_path=str(object_prim.GetPrimPath()), name=object_name)
        
        # Set object mass
        try:
            from pxr import UsdPhysics
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(object_prim)
            if rigid_body_api:
                mass_api = UsdPhysics.MassAPI.Apply(object_prim)
                if mass_api:
                    mass_api.CreateMassAttr(0.5)  # 0.5 kg
        except Exception as e:
            print(f'>>> [WORLD] Warning: Could not set {object_name} mass: {e}')
        
        # Store object prim
        self.object_prims[object_name] = object_prim
        if object_name == self.object_name:
            self.object_prim = object_prim
        
        # Create ground truth publisher for additional objects
        self._create_gt_publisher_for_object(object_name)
    
    # ========================================================================
    # Robot Control
    # ========================================================================
    
    def _calculate_robot_orientation_to_table(self, robot_pos, table_pos):
        """Calculate robot orientation to face the table."""
        dx = table_pos[0] - robot_pos[0]
        dy = table_pos[1] - robot_pos[1]
        yaw_angle = np.arctan2(dy, dx)
        return [0.0, 0.0, yaw_angle]
    
    def _calculate_object_orientation_to_robot(self, object_pos, robot_pos):
        """Calculate object orientation to face the robot."""
        dx = robot_pos[0] - object_pos[0]
        dy = robot_pos[1] - object_pos[1]
        yaw_angle = np.arctan2(dy, dx)
        return [0.0, 0.0, yaw_angle]
    
    def set_head_tilt_position(self, tilt_angle):
        """
        Set the head tilt joint position to look down at the table.
        
        Args:
            tilt_angle (float): Head tilt angle in radians. Negative values tilt down.
                                 Range: -1.57 to 0.52 radians (from URDF limits)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from omni.isaac.dynamic_control import _dynamic_control
            dc = _dynamic_control.acquire_dynamic_control_interface()
            art = dc.get_articulation('/World/hsrb')
            
            if art != _dynamic_control.INVALID_HANDLE:
                head_tilt_dof = dc.find_articulation_dof(art, "head_tilt_joint")
                if head_tilt_dof != _dynamic_control.DofType.DOF_NONE:
                    dc.set_dof_position_target(head_tilt_dof, tilt_angle)
                    print(f">>> [WORLD] Set head_tilt_joint to {tilt_angle:.3f} rad ({tilt_angle * 180 / np.pi:.1f} deg)")
                    self._pending_head_tilt = None
                    return True
                else:
                    print(f">>> [WORLD] Warning: Could not find head_tilt_joint DOF")
            else:
                print(f">>> [WORLD] Warning: Articulation not yet available, head tilt will be set after simulation starts")
                self._pending_head_tilt = tilt_angle
                return False
        except Exception as e:
            print(f">>> [WORLD] Warning: Could not set head tilt position: {e}")
            self._pending_head_tilt = tilt_angle
            return False
    
    def set_arm_neutral_position(self):
        """Set the arm to neutral position."""
        # Try MoveGroupCommander first
        try:
            import moveit_commander
            if not rospy.get_node_uri():
                self._pending_arm_neutral = True
                return False
            
            move_group = moveit_commander.MoveGroupCommander("arm")
            joint_values = move_group.get_current_joint_values()
            active_joints = move_group.get_active_joints()
            
            if "arm_roll_joint" in active_joints:
                joint_values[active_joints.index("arm_roll_joint")] = ARM_ROLL_JOINT_ANGLE
            if "wrist_flex_joint" in active_joints:
                joint_values[active_joints.index("wrist_flex_joint")] = WRIST_FLEX_JOINT_ANGLE
            
            move_group.go(joint_values, wait=True)
            move_group.stop()
            self._pending_arm_neutral = None
            return True
        except:
            return self._set_arm_neutral_dynamic_control()
    
    def _set_arm_neutral_dynamic_control(self):
        """Set arm to neutral position using dynamic_control (fallback)."""
        try:
            from omni.isaac.dynamic_control import _dynamic_control
            dc = _dynamic_control.acquire_dynamic_control_interface()
            art = dc.get_articulation('/World/hsrb')
            
            if art == _dynamic_control.INVALID_HANDLE:
                self._pending_arm_neutral = True
                return False
            
            arm_roll_dof = dc.find_articulation_dof(art, "arm_roll_joint")
            if arm_roll_dof != _dynamic_control.DofType.DOF_NONE:
                dc.set_dof_position_target(arm_roll_dof, ARM_ROLL_JOINT_ANGLE)
            
            wrist_flex_dof = dc.find_articulation_dof(art, "wrist_flex_joint")
            if wrist_flex_dof != _dynamic_control.DofType.DOF_NONE:
                dc.set_dof_position_target(wrist_flex_dof, WRIST_FLEX_JOINT_ANGLE)
            
            self._pending_arm_neutral = None
            return True
        except:
            self._pending_arm_neutral = True
            return False
    
    # ========================================================================
    # Ground Truth Pose Publishing
    # ========================================================================
    
    def publish_ground_truth_pose(self):
        """Publish ground truth object pose (same as original implementation)."""
        # Check if object prim exists
        if not hasattr(self, 'object_prim') or not self.object_prim:
            return
        
        try:
            # Get object transform (same as original implementation)
            object_world_transform = omni.usd.get_world_transform_matrix(self.object_prim)
            
            # Extract position and rotation in world frame (same as original)
            object_world_pos = np.array(object_world_transform.ExtractTranslation())
            object_world_rot_quat = object_world_transform.ExtractRotationQuat()
            object_world_rot = Gf.Quatd(
                object_world_rot_quat.real,
                object_world_rot_quat.imaginary[0],
                object_world_rot_quat.imaginary[1],
                object_world_rot_quat.imaginary[2]
            )
            
            # Create and publish PoseStamped message (same as original)
            gt_pose = PoseStamped()
            gt_pose.header.frame_id = 'map'  # Use 'map' frame like original
            gt_pose.header.stamp = rospy.Time.now()
            gt_pose.pose.position.x = float(object_world_pos[0])
            gt_pose.pose.position.y = float(object_world_pos[1])
            gt_pose.pose.position.z = float(object_world_pos[2])
            # Gf.Quatd: (real, imag_x, imag_y, imag_z) = (w, x, y, z)
            gt_pose.pose.orientation.x = float(object_world_rot.imaginary[0])
            gt_pose.pose.orientation.y = float(object_world_rot.imaginary[1])
            gt_pose.pose.orientation.z = float(object_world_rot.imaginary[2])
            gt_pose.pose.orientation.w = float(object_world_rot.real)
            
            self.object_gt_pub.publish(gt_pose)
            
            # Publish ground truth for additional objects (mustard)
            self._publish_all_object_ground_truth_poses()
            
        except Exception as e:
            print(f">>> [WORLD] ERROR in publish_ground_truth_pose: {e}", flush=True)
    
    def _publish_all_object_ground_truth_poses(self):
        """Publish ground truth poses for all additional objects."""
        if not hasattr(self, 'object_prims') or not hasattr(self, 'object_gt_pubs'):
            return
        
        for obj_name, obj_prim in self.object_prims.items():
            # Skip main object (already published in publish_ground_truth_pose)
            if obj_name == self.object_name:
                continue
            
            # Check if publisher exists
            if obj_name not in self.object_gt_pubs:
                continue
            
            if not obj_prim or not obj_prim.IsValid():
                continue
            
            try:
                # Get object transform (same as original implementation)
                obj_world_transform = omni.usd.get_world_transform_matrix(obj_prim)
                obj_world_rot_quat = obj_world_transform.ExtractRotationQuat()
                
                # Extract position and rotation in world frame (same as original)
                obj_world_pos = np.array(obj_world_transform.ExtractTranslation())
                
                # Convert rotation to Gf.Quatd (same as original)
                obj_world_rot = Gf.Quatd(
                    obj_world_rot_quat.real,
                    obj_world_rot_quat.imaginary[0],
                    obj_world_rot_quat.imaginary[1],
                    obj_world_rot_quat.imaginary[2]
                )
                
                # Create and publish PoseStamped message
                # For mustard_bottle, use 'map' frame like original for comparison
                # For other objects, use 'odom' frame
                frame_id = 'map' if obj_name == 'mustard_bottle' else 'odom'
                
                gt_pose = PoseStamped()
                gt_pose.header.frame_id = frame_id
                gt_pose.header.stamp = rospy.Time.now()
                gt_pose.pose.position.x = float(obj_world_pos[0])
                gt_pose.pose.position.y = float(obj_world_pos[1])
                gt_pose.pose.position.z = float(obj_world_pos[2])
                # Gf.Quatd: (real, imag_x, imag_y, imag_z) = (w, x, y, z) - same as original
                gt_pose.pose.orientation.x = float(obj_world_rot.imaginary[0])
                gt_pose.pose.orientation.y = float(obj_world_rot.imaginary[1])
                gt_pose.pose.orientation.z = float(obj_world_rot.imaginary[2])
                gt_pose.pose.orientation.w = float(obj_world_rot.real)
                
                self.object_gt_pubs[obj_name].publish(gt_pose)
                
                # Debug output for mustard (like original)
                if obj_name == 'mustard_bottle' and not hasattr(self, '_mustard_gt_debug_logged'):
                    print(f">>> [WORLD] Publishing ground truth pose for mustard_bottle in {frame_id} frame:")
                    print(f">>> [WORLD]   World position:    ({obj_world_pos[0]:.4f}, {obj_world_pos[1]:.4f}, {obj_world_pos[2]:.4f}) m")
                    print(f">>> [WORLD]   World orientation: ({obj_world_rot.imaginary[0]:.4f}, {obj_world_rot.imaginary[1]:.4f}, {obj_world_rot.imaginary[2]:.4f}, {obj_world_rot.real:.4f})")
                    self._mustard_gt_debug_logged = True
            except Exception as e:
                print(f">>> [WORLD] ERROR publishing ground truth for {obj_name}: {e}")
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def decompose_matrix(self, mat: Gf.Matrix4d):
        """Decompose a 4x4 transformation matrix into translation, rotation, and scale."""
        reversed_ident_mtx = reversed(Gf.Matrix3d())
        translate = Gf.Vec3d(mat.ExtractTranslation())
        scale = Gf.Vec3d(*(v.GetLength() for v in mat.ExtractRotationMatrix()))
        mat.Orthonormalize()
        rotate = Gf.Vec3d(*reversed(mat.ExtractRotation().Decompose(*reversed_ident_mtx)))
        return np.array(translate), np.array(rotate), np.array(scale)
    
    def get_world_transform_xform(self, prim: Usd.Prim):
        """Get world transform of a prim decomposed into translation, rotation, and scale."""
        world_transform = omni.usd.get_world_transform_matrix(prim)
        return self.decompose_matrix(world_transform)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print('>>> [WORLD] Starting world setup...')
    
    # Create world instance
    try:
        world = final_project_world_2()
    except Exception as e:
        print(f'>>> [WORLD] ERROR creating world: {e}')
        import traceback
        print(traceback.format_exc())
        raise
    
    # Simulation state tracking
    sim_world_onetime_trigger = True
    head_tilt_set = False
    arm_neutral_set = False
    
    print('>>> [WORLD] Starting simulation loop...')
    
    # Main simulation loop
    while sim_app.is_running():
        # Step simulation
        world.sim_world.step(render=True)
        world.hsr_instance.step()
        world.sim_world.play()
        
        # Handle one-time setup when simulation starts
        if world.sim_world.is_playing():
            if sim_world_onetime_trigger:
                print('>>> [WORLD] Simulation started')
                sim_world_onetime_trigger = False
                
                # Lock table orientation
                if hasattr(world, 'table_prim') and world.table_prim:
                    try:
                        from pxr import UsdPhysics
                        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(world.table_prim)
                        if rigid_body_api:
                            rigid_body_api.CreateKinematicEnabledAttr(True)
                    except Exception as e:
                        print(f'>>> [WORLD] Warning: Could not lock table: {e}')
            
            # Set head tilt after simulation starts when articulation is available
            if not head_tilt_set and hasattr(world, '_pending_head_tilt') and world._pending_head_tilt is not None:
                if world.set_head_tilt_position(world._pending_head_tilt):
                    head_tilt_set = True
                    print(f">>> [WORLD] Head tilt successfully set in main loop")
            
            # Set arm neutral position after simulation starts when articulation is available
            if not arm_neutral_set and hasattr(world, '_pending_arm_neutral') and world._pending_arm_neutral is not None:
                if world.set_arm_neutral_position():
                    arm_neutral_set = True
                    print(f">>> [WORLD] Arm neutral position successfully set in main loop")
        
        # Start robot behavior
        world.isaac_robot_behavior_start.start()
        
        # Publish robot pose
        chosen_prim = world.stage.GetPrimAtPath('/World/hsrb/base_footprint')
        world_translate, world_rotate, world_scale = world.get_world_transform_xform(prim=chosen_prim)
        prim_world_pose = np.array([world_translate[0], world_translate[1], np.radians(world_rotate[2])])
        world.isaac_robot_pose_pub.publish(pose=prim_world_pose)
        
        # Publish ground truth object pose
        if world.sim_world.is_playing():
            try:
                world.publish_ground_truth_pose()
            except Exception as e:
                print(f'>>> [WORLD] ERROR in publish_ground_truth_pose: {e}')
    
    # Cleanup
    sim_app.close()
