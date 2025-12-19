#!/usr/bin/env python3
"""
FoundationPose ROS node for 6D pose estimation.
"""

import os
import sys
import ctypes
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from copy import deepcopy
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import rospy
import numpy as np
import cv2
import message_filters
import tf2_ros
from tf import TransformListener
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

# Fix libffi conflicts with cv_bridge (must be before cv_bridge import)
conda_prefix = os.environ.get('CONDA_PREFIX', '')
if conda_prefix:
    libffi_path = os.path.join(conda_prefix, 'lib', 'libffi.so.7')
    if os.path.exists(libffi_path):
        current_preload = os.environ.get('LD_PRELOAD', '')
        if libffi_path not in current_preload:
            os.environ['LD_PRELOAD'] = f"{libffi_path}:{current_preload}" if current_preload else libffi_path
            # Try to force load conda's libffi using dlopen
            try:
                libdl = ctypes.CDLL('libdl.so.2')
                libdl.dlopen(libffi_path, ctypes.RTLD_GLOBAL | ctypes.RTLD_NOW)
            except Exception:
                # If it fails, log but continue - the real fix is in the wrapper script
                pass

# Find FoundationPose installation
# Try common workspace locations
FOUNDATIONPOSE_PATHS = [
    os.path.join(os.path.expanduser('~'), 'catkin_ws', 'src', 'FoundationPose'),
    os.path.join(os.path.expanduser('~'), 'ros_ws', 'src', 'FoundationPose'),
    os.path.join(os.path.expanduser('~'), 'workspace', 'src', 'FoundationPose'),
    '/opt/foundationpose',
    # Legacy paths (for backward compatibility)
    os.path.join(os.path.expanduser('~'), 'hsr_robocanes_omniverse', 'src', 'FoundationPose'),
]

FOUNDATIONPOSE_PATH = None
for path in FOUNDATIONPOSE_PATHS:
    if os.path.exists(path):
        FOUNDATIONPOSE_PATH = path
        break

if FOUNDATIONPOSE_PATH is None:
    print(f"ERROR: FoundationPose not found. Tried: {FOUNDATIONPOSE_PATHS}")
    sys.exit(1)

sys.path.insert(0, FOUNDATIONPOSE_PATH)

try:
    from estimater import FoundationPose
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from Utils import set_seed, set_logging_format, draw_posed_3d_box, draw_xyz_axis
    import trimesh
    import nvdiffrast.torch as dr
    import imageio
except ImportError as e:
    print(f"ERROR: Failed to import FoundationPose modules: {e}")
    print("Make sure you're running in the foundationpose conda environment")
    sys.exit(1)


class FoundationPoseNode:
    """
    FoundationPose ROS Node
    
    6D pose estimation using FoundationPose.
    using FoundationPose, and publishes pose estimates and visualizations.
    """
    
    # Constants for visualization
    AXIS_LENGTH = 0.15  # 15cm axis length for coordinate frame markers
    # Standard RGB convention: X=Red, Y=Green, Z=Blue
    AXIS_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue for x, y, z axes
    AXIS_NAMES = ['x', 'y', 'z']
    AXIS_SHAFT_DIAMETER = 0.01  # Shaft diameter for arrow markers
    AXIS_HEAD_DIAMETER = 0.02   # Head diameter for arrow markers
    
    # Comparison printing intervals
    COMPARISON_PRINT_INTERVAL = 1.0  # Print comparison every 1 second
    GT_ALL_FRAMES_PRINT_INTERVAL = 5.0  # Print GT in all frames every 5 seconds
    
    def __init__(self):
        """Initialize the FoundationPose node."""
        rospy.init_node('foundationpose_pose_estimation', anonymous=True)
        
        # Set seed for reproducibility
        set_seed(0)
        # set_logging_format()
        rospy.loginfo("Set random seed to 0 for reproducible pose estimation")
        
        # Load parameters
        self._load_parameters()
        
        # Initialize state variables
        self.camera_K = None
        self.pose_initialized = False
        # Track last published pose for temporal consistency checking
        self.last_published_pose = None
        self.last_published_time = None
        self.pose_jump_threshold = 0.5  # Maximum allowed position jump in meters
        self.gt_pose = None
        self.gt_pose_received = False
        
        # Manual synchronization buffers (for exact timestamp matching)
        # Store latest messages keyed by timestamp
        self.rgb_buffer = {}  # {timestamp: rgb_msg}
        self.depth_buffer = {}  # {timestamp: depth_msg}
        self.info_buffer = {}  # {timestamp: info_msg}
        self.mask_buffer = {}  # {timestamp: mask_msg}
        self.buffer_lock = threading.Lock()  # Thread lock for buffer access
        self.buffer_max_age = rospy.Duration(2.0)  # Keep messages for max 2 seconds
        
        # Circular buffer for pose consensus
        self.pose_buffer_size = rospy.get_param('~pose_buffer_size', 10)  # Store last 10 poses
        self.pose_buffer = []  # Circular buffer of recent pose estimates (stores tuples: (pose_matrix, header))
        self.pose_consensus_threshold = rospy.get_param('~pose_consensus_threshold', 0.1)  # 10cm position threshold for consensus
        self.pose_consensus_orientation_threshold = rospy.get_param('~pose_consensus_orientation_threshold', 15.0)  # 15 degrees orientation threshold
        self.pose_consensus_time_threshold = rospy.get_param('~pose_consensus_time_threshold', 0.5)  # Time threshold for consensus clustering
        rospy.loginfo(f"Pose consensus buffer: size={self.pose_buffer_size}, pos_threshold={self.pose_consensus_threshold}m, orient_threshold={self.pose_consensus_orientation_threshold}°, time_threshold={self.pose_consensus_time_threshold}s")
        
        # Marker color alternation for visual debugging
        self.marker_color_counter = 0  # Counter to alternate between red and blue
        # These will be loaded in _load_parameters() after config file is loaded
        
        # Throttling for synchronized printing
        self._last_performance_print_time = 0
        self._last_comparison_time = 0
        
        # Parallel processing setup
        self.max_parallel_workers = rospy.get_param('~max_parallel_workers', 8)  # Increased to 16 for higher throughput and more buffer updates
        self.pose_executor = ThreadPoolExecutor(max_workers=self.max_parallel_workers, thread_name_prefix="FoundationPose")
        self.pose_lock = threading.Lock()  # Thread lock for buffer and publisher access
        self.active_poses = {}  # Track active pose estimations by timestamp: {timestamp: Future}
        rospy.loginfo(f"Parallel processing enabled: {self.max_parallel_workers} workers")
        
        # Metrics saving
        import os
        workspace_root = os.path.expanduser('~/hsr_robocanes_omniverse')
        self.metrics_dir = os.path.join(workspace_root, 'src', 'ofpgs_ros', 'metrics', 'data')
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.metrics_dir, 'foundationpose_metrics.json')
        self.metrics_data = []
        rospy.loginfo(f"Metrics will be saved to: {self.metrics_file}")
        
        # Initialize pynvml if available (for GPU utilization monitoring)
        self.pynvml_initialized = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml_initialized = True
        except (ImportError, Exception):
            pass
        
        
        # Setup ROS communication
        self._setup_ros_communication()
        
        # Load mesh and initialize FoundationPose
        self._load_mesh()
        self._initialize_foundationpose()
        
        # Setup subscribers
        self._setup_subscribers()
        
        rospy.loginfo("FoundationPose node initialized")
        rospy.loginfo(f"Subscribing to RGB: {self.rgb_topic}")
        rospy.loginfo(f"Subscribing to Depth: {self.depth_topic}")
        rospy.loginfo(f"Subscribing to CameraInfo: {self.camera_info_topic}")
        if not PSUTIL_AVAILABLE:
            rospy.logwarn("psutil not available, CPU monitoring will be disabled")
        # Object name will be set in _load_parameters, log it after
    
    # ========================================================================
    # ========================================================================
    
    def _load_parameters(self):
        """Load ROS parameters from config file (organized structure)."""
        # Object name parameter (used for topics and default paths)
        self.object_name = rospy.get_param('~object_name', 'cracker_box')
        rospy.loginfo(f"Object name: {self.object_name}")
        
        # Mesh file parameter (top level)
        # Default mesh paths based on object name
        workspace_root = os.path.expanduser('~/hsr_robocanes_omniverse')
        meshes_dir = os.path.join(workspace_root, 'src', 'ofpgs_ros', 'meshes')
        default_mesh_paths = [
            # Try in object-specific folder (organized structure) - YCB naming convention
            os.path.join(meshes_dir, self.object_name, 'textured.obj'),  # YCB standard name
            os.path.join(meshes_dir, self.object_name, 'textured_simple.obj'),  # FoundationPose demo naming
            os.path.join(meshes_dir, self.object_name, 'mesh.obj'),  # Generic name
            os.path.join(meshes_dir, self.object_name, f'{self.object_name}.obj'),  # Object name as filename
            # Try exact object name (flat structure) - for backward compatibility
            os.path.join(meshes_dir, f'{self.object_name}.obj'),
            # Try without underscores (flat structure)
            os.path.join(meshes_dir, f'{self.object_name.replace("_", "")}.obj'),
            # Special case: if object_name is "cracker_box", also try "craker_box" (typo in filename)
            os.path.join(meshes_dir, 'craker_box.obj') if 'cracker' in self.object_name.lower() else None,
            # Fallback to FoundationPose demo data
            os.path.join(FOUNDATIONPOSE_PATH, 'demo_data', 'mustard0', 'mesh', 'textured_simple.obj'),
            os.path.join(os.path.expanduser('~'), 'hsr_robocanes_omniverse', 'src', 'FoundationPose', 'demo_data', 'mustard0', 'mesh', 'textured_simple.obj'),
        ]
        # Remove None entries
        default_mesh_paths = [p for p in default_mesh_paths if p is not None]
        
        self.mesh_file = rospy.get_param('~mesh_file', '')
        
        # Resolve path relative to workspace root if needed
        if self.mesh_file:
            # Get workspace root from ROS package path or environment
            workspace_root = None
            if 'ROS_PACKAGE_PATH' in os.environ:
                # Use first path in ROS_PACKAGE_PATH (usually workspace/src)
                package_paths = os.environ['ROS_PACKAGE_PATH'].split(':')
                if package_paths:
                    # Go up from src/ to workspace root
                    src_path = package_paths[0]
                    if src_path.endswith('/src'):
                        workspace_root = os.path.dirname(src_path)
            
            # Fallback to common workspace location
            if not workspace_root:
                workspace_root = os.path.expanduser('~/hsr_robocanes_omniverse')
            
            # If path starts with 'src/', resolve relative to workspace root
            if self.mesh_file.startswith('src/'):
                self.mesh_file = os.path.join(workspace_root, self.mesh_file)
            # Expand user home directory if path starts with ~
            elif self.mesh_file.startswith('~'):
                self.mesh_file = os.path.expanduser(self.mesh_file)
            # If relative path, try resolving relative to workspace root
            elif not os.path.isabs(self.mesh_file):
                potential_path = os.path.join(workspace_root, self.mesh_file)
                if os.path.exists(potential_path):
                    self.mesh_file = potential_path
        
        if not self.mesh_file:
            # Try default paths
            for default_path in default_mesh_paths:
                if os.path.exists(default_path):
                    self.mesh_file = default_path
                    rospy.loginfo(f"No mesh_file parameter provided, using default: {self.mesh_file}")
                    break
        
        if not self.mesh_file or not os.path.exists(self.mesh_file):
            rospy.logerr(f"Mesh file not found: {self.mesh_file}")
            rospy.logerr("Please set the ~mesh_file parameter to a valid mesh file")
            if default_mesh_paths:
                rospy.logerr(f"Suggested default: {default_mesh_paths[0]}")
            sys.exit(1)
        
        # Camera topics (from config file nested structure)
        self.rgb_topic = rospy.get_param('~camera/rgb_topic', rospy.get_param('~rgb_topic', '/hsrb/head_rgbd_sensor/rgb/image_rect_color'))
        self.depth_topic = rospy.get_param('~camera/depth_topic', rospy.get_param('~depth_topic', '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'))
        self.camera_info_topic = rospy.get_param('~camera/camera_info_topic', rospy.get_param('~camera_info_topic', '/hsrb/head_rgbd_sensor/rgb/camera_info'))
        self.frame_id = rospy.get_param('~camera/frame_id', rospy.get_param('~frame_id', 'head_rgbd_sensor_rgb_frame'))
        self.object_frame_id = rospy.get_param('~object_frame_id', 'object_pose')
        
        # FoundationPose parameters (from config file nested structure)
        self.est_refine_iter = rospy.get_param('~foundationpose/est_refine_iter', rospy.get_param('~est_refine_iter', 5))
        self.track_refine_iter = rospy.get_param('~foundationpose/track_refine_iter', rospy.get_param('~track_refine_iter', 2))
        self.debug = rospy.get_param('~foundationpose/debug', rospy.get_param('~debug', 1))
        
        # Mask parameters (from config file nested structure)
        self.use_mask = rospy.get_param('~mask/use_mask', rospy.get_param('~use_mask', False))
        self.mask_topic = rospy.get_param('~mask/mask_topic', rospy.get_param('~mask_topic', ''))
        # If mask_topic is empty, use default based on object_name
        if not self.mask_topic:
            self.mask_topic = f'/segmentation/{self.object_name}_mask'
            rospy.loginfo(f"No mask_topic specified, using default: {self.mask_topic}")
        
        # Depth mask parameters (from config file nested structure)
        self.depth_min = rospy.get_param('~depth_mask/depth_min', rospy.get_param('~depth_min', 0.3))
        self.depth_max = rospy.get_param('~depth_mask/depth_max', rospy.get_param('~depth_max', 2.0))
        
        # Coordinate frame correction (to fix upside-down objects in RViz)
        self.coord_correction_enabled = rospy.get_param('~coordinate_frame_correction/enabled', False)
        if self.coord_correction_enabled:
            euler_zyx_deg = rospy.get_param('~coordinate_frame_correction/rotation_euler_zyx_degrees', [0, 0, 180])
            from scipy.spatial.transform import Rotation
            # Convert degrees to radians and create rotation matrix
            euler_zyx_rad = np.deg2rad(euler_zyx_deg)
            self.coord_correction_rotation = Rotation.from_euler('zyx', euler_zyx_rad, degrees=False)
            self.coord_correction_matrix = self.coord_correction_rotation.as_matrix()
            # Store original Euler angles for debug output (don't convert back from matrix)
            self.coord_correction_euler_deg = euler_zyx_deg
            rospy.loginfo(f"Coordinate frame correction enabled: Euler ZYX (degrees) = {euler_zyx_deg}")
        else:
            self.coord_correction_matrix = None
            self.coord_correction_euler_deg = None
            rospy.loginfo("Coordinate frame correction disabled")
        
        # Pose quality thresholds for publishing (from config file nested structure)
        self.publish_position_error_threshold = rospy.get_param('~publish_quality/publish_position_error_threshold',
                                                                rospy.get_param('~publish_position_error_threshold', 0.2))
        self.publish_orientation_error_threshold = rospy.get_param('~publish_quality/publish_orientation_error_threshold',
                                                                     rospy.get_param('~publish_orientation_error_threshold', 45.0))
    
    def _setup_ros_communication(self):
        """Setup ROS publishers, subscribers, and TF broadcaster."""
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # TF listener for transforming poses to odom frame (using old tf API for PoseStamped)
        self.tf_listener = TransformListener()
        
        # Publishers (queue_size=1 to avoid buffering old poses)
        self.pose_pub = rospy.Publisher('~pose', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('~markers', MarkerArray, queue_size=1)
        
        # Subscriber for ground truth pose (topic based on object name)
        gt_topic = f'/{self.object_name}/ground_truth_pose'
        rospy.loginfo(f"Subscribing to Ground Truth: {gt_topic}")
        self.gt_pose_sub = rospy.Subscriber(
            gt_topic, 
            PoseStamped, 
            self.gt_pose_callback,
            queue_size=1  # Small queue for latest pose only
        )
        
        # Debug: Check if topic exists after a short delay
        rospy.Timer(rospy.Duration(2.0), self._check_gt_topic, oneshot=True)
    
    def _load_mesh(self):
        """Load the object mesh file."""
        rospy.loginfo(f"Loading mesh from: {self.mesh_file}")
        try:
            # Store original mesh before FoundationPose modifies it
            self.mesh_original = trimesh.load(self.mesh_file)
            self.mesh = self.mesh_original.copy()  # FoundationPose will modify this
            rospy.loginfo(f"Mesh loaded: {len(self.mesh.vertices)} vertices")
            
            # Compute to_origin transformation from oriented bounds (same as run_demo.py)
            # This transforms from original mesh coordinate system to oriented bounding box coordinate system
            self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh_original)
            rospy.loginfo(f"Computed to_origin transformation from oriented bounds")
            rospy.loginfo(f"Oriented bounds extents: {extents}")
            
            # Store extents and compute bbox for visualization (same as run_demo.py)
            self.extents = extents
            self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
            
            # Check mesh orientation and bounds for debugging
            if self.mesh.vertices.shape[0] > 0:
                min_bounds = self.mesh.vertices.min(axis=0)
                max_bounds = self.mesh.vertices.max(axis=0)
                center = (min_bounds + max_bounds) / 2
                extents = max_bounds - min_bounds
                rospy.loginfo(f"Mesh bounds: min={min_bounds}, max={max_bounds}")
                rospy.loginfo(f"Mesh center: {center}, extents: {extents}")
                rospy.loginfo(f"Mesh extent ratios (X:Y:Z): {extents[0]/extents.max():.2f} : {extents[1]/extents.max():.2f} : {extents[2]/extents.max():.2f}")
        except Exception as e:
            rospy.logerr(f"Failed to load mesh: {e}")
            sys.exit(1)
    
    def _initialize_foundationpose(self):
        """Initialize FoundationPose estimator."""
        rospy.loginfo("Initializing FoundationPose...")
        try:
            # Clear CUDA cache before initialization to reduce memory fragmentation
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    rospy.loginfo("Cleared CUDA cache before FoundationPose initialization")
            except Exception as e:
                rospy.logwarn(f"Could not clear CUDA cache: {e}")
            
            # Get package path for debug folders (before FoundationPose initialization)
            try:
                import rospkg
                rospack = rospkg.RosPack()
                self.package_path = rospack.get_path('ofpgs_ros')
            except Exception as e:
                rospy.logwarn(f"Could not get package path using rospack: {e}")
                # Fallback to workspace root
                self.package_path = os.path.join(os.path.expanduser('~'), 'catkin_ws', 'src', 'ofpgs_ros')
            
            # Create debug folder in our package for FoundationPose internal debug
            foundationpose_debug_dir = os.path.join(self.package_path, 'debug', 'foundationpose')
            os.makedirs(foundationpose_debug_dir, exist_ok=True)
            
            # Create track_vis folder for our visualization images
            self.debug_dir = os.path.join(self.package_path, 'debug', 'track_vis')
            os.makedirs(self.debug_dir, exist_ok=True)
            rospy.loginfo(f"FoundationPose debug will be saved to: {foundationpose_debug_dir}")
            rospy.loginfo(f"Track visualization will be saved to: {self.debug_dir}")
            
            self.scorer = ScorePredictor()
            self.refiner = PoseRefinePredictor()
            self.glctx = dr.RasterizeCudaContext()
            
            self.estimator = FoundationPose(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                glctx=self.glctx,
                debug=self.debug,
                debug_dir=foundationpose_debug_dir  # Use our project folder
            )
            rospy.loginfo("FoundationPose initialized successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize FoundationPose: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _setup_subscribers(self):
        """Setup image subscribers with manual timestamp-based synchronization."""
        # Use regular ROS subscribers instead of message_filters
        # We'll manually match messages by exact timestamp
        if self.use_mask and self.mask_topic:
            # Subscribe to all topics individually
            rospy.Subscriber(self.rgb_topic, Image, self._rgb_callback)
            rospy.Subscriber(self.depth_topic, Image, self._depth_callback)
            rospy.Subscriber(self.camera_info_topic, CameraInfo, self._info_callback)
            rospy.Subscriber(self.mask_topic, Image, self._mask_callback)
            rospy.loginfo("Using manual timestamp-based synchronization (exact matching)")
            rospy.loginfo(f"Subscribing to RGB: {self.rgb_topic}")
            rospy.loginfo(f"Subscribing to Depth: {self.depth_topic}")
            rospy.loginfo(f"Subscribing to CameraInfo: {self.camera_info_topic}")
            rospy.loginfo(f"Subscribing to Mask: {self.mask_topic}")
            rospy.logwarn(f"Waiting for mask messages on {self.mask_topic}. Make sure Grounded SAM is running!")
        else:
            # For non-mask case, use TimeSynchronizer (RGB/depth/info are already synchronized)
            rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
            depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)
            self.ts = message_filters.TimeSynchronizer([rgb_sub, depth_sub, info_sub], 10)
            self.ts.registerCallback(self.image_callback)
    
    # Message synchronization callbacks
    
    def _rgb_callback(self, rgb_msg):
        """Store RGB message in buffer and try to match with other messages."""
        with self.buffer_lock:
            timestamp = rgb_msg.header.stamp
            self.rgb_buffer[timestamp] = rgb_msg
            self._cleanup_old_messages()
            self._try_match_messages(timestamp)
        rospy.logdebug(f"[SYNC] RGB message received: {timestamp.secs}.{timestamp.nsecs}")
    
    def _depth_callback(self, depth_msg):
        """Store depth message in buffer and try to match with other messages."""
        with self.buffer_lock:
            timestamp = depth_msg.header.stamp
            self.depth_buffer[timestamp] = depth_msg
            self._cleanup_old_messages()
            self._try_match_messages(timestamp)
    
    def _info_callback(self, info_msg):
        """Store camera info message in buffer and try to match with other messages."""
        with self.buffer_lock:
            timestamp = info_msg.header.stamp
            self.info_buffer[timestamp] = info_msg
            self._cleanup_old_messages()
            self._try_match_messages(timestamp)
    
    def _mask_callback(self, mask_msg):
        """Store mask message in buffer and try to match with other messages."""
        with self.buffer_lock:
            timestamp = mask_msg.header.stamp
            self.mask_buffer[timestamp] = mask_msg
            self._cleanup_old_messages()
            # Masks arrive later, so try matching when mask arrives
            self._try_match_messages(timestamp)
        rospy.loginfo(f"[SYNC] Mask message received: {timestamp.secs}.{timestamp.nsecs}")
    
    def _cleanup_old_messages(self):
        """Remove messages older than buffer_max_age."""
        now = rospy.Time.now()
        cutoff_time = now - self.buffer_max_age
        
        # Clean up each buffer
        for buffer in [self.rgb_buffer, self.depth_buffer, self.info_buffer, self.mask_buffer]:
            timestamps_to_remove = [ts for ts in buffer.keys() if ts < cutoff_time]
            for ts in timestamps_to_remove:
                del buffer[ts]
    
    def _try_match_messages(self, timestamp):
        """
        Try to match messages with the given timestamp.
        If all required messages (RGB, depth, info, mask) are available, process them.
        """
        # Check if we have all required messages for this timestamp
        if timestamp in self.rgb_buffer and timestamp in self.depth_buffer and timestamp in self.info_buffer:
            if self.use_mask and self.mask_topic:
                # Need mask too
                if timestamp in self.mask_buffer:
                    rospy.loginfo(f"[SYNC] All messages matched for timestamp {timestamp.secs}.{timestamp.nsecs} - processing")
                    # All messages available - process them
                    rgb_msg = self.rgb_buffer[timestamp]
                    depth_msg = self.depth_buffer[timestamp]
                    info_msg = self.info_buffer[timestamp]
                    mask_msg = self.mask_buffer[timestamp]
                    
                    # Remove from buffers (they've been matched)
                    del self.rgb_buffer[timestamp]
                    del self.depth_buffer[timestamp]
                    del self.info_buffer[timestamp]
                    del self.mask_buffer[timestamp]
                    
                    # Process the matched messages
                    self.image_callback_with_mask(rgb_msg, depth_msg, info_msg, mask_msg)
            else:
                # No mask needed - process without mask
                rgb_msg = self.rgb_buffer[timestamp]
                depth_msg = self.depth_buffer[timestamp]
                info_msg = self.info_buffer[timestamp]
                
                # Remove from buffers
                del self.rgb_buffer[timestamp]
                del self.depth_buffer[timestamp]
                del self.info_buffer[timestamp]
                
                # Process the matched messages
                self.image_callback(rgb_msg, depth_msg, info_msg)
    
    # ========================================================================
    # Image Processing
    # ========================================================================
    
    def extract_camera_matrix(self, camera_info):
        """Extract camera intrinsic matrix K from CameraInfo."""
        K = np.array(camera_info.K).reshape(3, 3)
        return K
    
    def ros_image_to_numpy(self, img_msg, desired_encoding='rgb8'):
        """
        Convert ROS Image message to numpy array without cv_bridge.
        This avoids libffi conflicts between system ROS and conda FoundationPose.
        
        Args:
            img_msg: sensor_msgs.msg.Image
            desired_encoding: Desired output encoding (e.g., 'rgb8', 'bgr8', '32FC1', 'mono8')
        
        Returns:
            numpy.ndarray: Image as numpy array
        """
        height = img_msg.height
        width = img_msg.width
        
        # Convert raw data to numpy array based on encoding
        if img_msg.encoding in ['8UC1', 'mono8']:
            img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width)
            if desired_encoding == 'rgb8':
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_msg.encoding in ['8UC3', 'rgb8', 'bgr8']:
            img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width, 3)
            if img_msg.encoding == 'bgr8' and desired_encoding == 'rgb8':
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            elif img_msg.encoding == 'rgb8' and desired_encoding == 'bgr8':
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding in ['16UC1', '16SC1']:
            img_array = np.frombuffer(
                img_msg.data, 
                dtype=np.uint16 if '16UC' in img_msg.encoding else np.int16
            ).reshape(height, width)
            if desired_encoding == '32FC1':
                img_array = img_array.astype(np.float32) / 1000.0  # Convert mm to meters
        elif img_msg.encoding in ['32FC1']:
            img_array = np.frombuffer(img_msg.data, dtype=np.float32).reshape(height, width)
        elif img_msg.encoding in ['32FC3']:
            img_array = np.frombuffer(img_msg.data, dtype=np.float32).reshape(height, width, 3)
        else:
            rospy.logwarn(f"Unsupported encoding: {img_msg.encoding}, attempting default conversion")
            if len(img_msg.data) == height * width:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width)
            elif len(img_msg.data) == height * width * 3:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width, 3)
            else:
                raise ValueError(f"Cannot convert encoding {img_msg.encoding} to {desired_encoding}")
        
        return img_array
    
    def image_callback(self, rgb_msg, depth_msg, info_msg):
        """Callback for synchronized RGB, depth, and camera info."""
        try:
            # Extract camera matrix
            if self.camera_K is None:
                self.camera_K = self.extract_camera_matrix(info_msg)
                rospy.loginfo(f"Camera matrix K:\n{self.camera_K}")
            
            # Convert ROS images to numpy arrays
            rgb_image = self.ros_image_to_numpy(rgb_msg, desired_encoding='rgb8')
            depth_image = self.ros_image_to_numpy(depth_msg, desired_encoding='32FC1')
            
            # Process pose estimation
            self.process_pose_estimation(rgb_image, depth_image, rgb_msg.header)
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def image_callback_with_mask(self, rgb_msg, depth_msg, info_msg, mask_msg):
        """Callback with mask for object segmentation."""
        try:
            # Verify timestamps match
            # If timestamps don't match, the mask is from a different image and will cause incorrect pose
            rgb_stamp = rgb_msg.header.stamp
            depth_stamp = depth_msg.header.stamp
            info_stamp = info_msg.header.stamp
            mask_stamp = mask_msg.header.stamp
            
            # With manual synchronization, timestamps should already match exactly
            # But verify as a safety check (allow tiny tolerance for floating-point precision)
            stamp_diff = abs((rgb_stamp - mask_stamp).to_sec())
            if stamp_diff > 0.001:  # 1ms tolerance for floating-point precision
                rospy.logwarn(f"[FOUNDATIONPOSE] Timestamp mismatch in matched messages! RGB: {rgb_stamp.secs}.{rgb_stamp.nsecs}, "
                            f"Mask: {mask_stamp.secs}.{mask_stamp.nsecs}, diff: {stamp_diff*1000:.3f}ms. This shouldn't happen with manual sync.")
                return
            
            # Also check depth and info timestamps (should match exactly)
            if abs((rgb_stamp - depth_stamp).to_sec()) > 0.05:
                rospy.logwarn_throttle(5.0, f"Depth timestamp mismatch! RGB: {rgb_stamp.secs}.{rgb_stamp.nsecs}, "
                                          f"Depth: {depth_stamp.secs}.{depth_stamp.nsecs}, diff: {abs((rgb_stamp - depth_stamp).to_sec())*1000:.1f}ms.")
            
            # Extract camera matrix
            if self.camera_K is None:
                self.camera_K = self.extract_camera_matrix(info_msg)
                rospy.loginfo(f"Camera matrix K:\n{self.camera_K}")
            
            # Convert ROS images to numpy arrays
            rgb_image = self.ros_image_to_numpy(rgb_msg, desired_encoding='rgb8')
            depth_image = self.ros_image_to_numpy(depth_msg, desired_encoding='32FC1')
            mask_image = self.ros_image_to_numpy(mask_msg, desired_encoding='mono8')
            
            # Verify mask dimensions match RGB image
            if mask_image.shape[:2] != rgb_image.shape[:2]:
                rospy.logwarn_throttle(5.0, f"Mask size mismatch! RGB: {rgb_image.shape[:2]}, Mask: {mask_image.shape[:2]}. "
                                          f"Skipping pose estimation.")
                return
            
            # Convert mask to boolean
            ob_mask = mask_image.astype(bool)
            
            # Check if mask is actually empty BEFORE further processing
            mask_pixels = ob_mask.sum() if ob_mask is not None else 0
            if mask_pixels == 0:
                rospy.logwarn_throttle(5.0, f"[FOUNDATIONPOSE] Empty mask detected! RGB: {rgb_stamp.secs}.{rgb_stamp.nsecs}, "
                                          f"Mask: {mask_stamp.secs}.{mask_stamp.nsecs}, diff: {abs((rgb_stamp - mask_stamp).to_sec())*1000:.1f}ms. "
                                          f"This mask is likely from a different frame or no detection. Skipping.")
                return  # Skip if mask is completely empty
            
            # Log timestamp info for debugging
            # rospy.loginfo_throttle(2.0, f"[SYNC] Received synchronized images: RGB={rgb_stamp.secs}.{rgb_stamp.nsecs}, "
            #               f"Depth={depth_stamp.secs}.{depth_stamp.nsecs}, Mask={mask_stamp.secs}.{mask_stamp.nsecs}, "
            #               f"RGB-Mask diff: {abs((rgb_stamp - mask_stamp).to_sec())*1000:.1f}ms")
            
            # Process pose estimation with mask (use RGB header as it's the primary timestamp)
            
            # Validate mask size BEFORE submitting to thread pool (avoid unnecessary task creation)
            if ob_mask is not None:
                H, W = ob_mask.shape[:2]
                min_mask_pixels = max(500, int(W * H * 0.005))  # At least 500 pixels or 0.5% of image
                if mask_pixels < min_mask_pixels:
                    rospy.logwarn(f"[FOUNDATIONPOSE] Mask too small ({mask_pixels} pixels < {min_mask_pixels}), skipping submission")
                    return
                else:
                    rospy.loginfo(f"[FOUNDATIONPOSE] Valid mask: {mask_pixels} pixels (required: {min_mask_pixels})")
            
            # Check if image is too stale (skip if >2 seconds old to avoid processing outdated frames)
            time_since_image = (rospy.Time.now() - rgb_msg.header.stamp).to_sec()
            if time_since_image > 2.0:
                rospy.logwarn(f"[FOUNDATIONPOSE] Skipping stale image ({time_since_image:.1f}s old)")
                return
            
            # Check number of active poses (limit to prevent GPU memory issues)
            with self.pose_lock:
                active_count = len([f for f in self.active_poses.values() if not f.done()])
                # Allow queuing up to 1.5x max_workers to keep pipeline full
                max_queue_size = int(self.max_parallel_workers * 1.5)
                if active_count >= max_queue_size:
                    rospy.logwarn(f"[FOUNDATIONPOSE] Queue full ({active_count}/{max_queue_size}), skipping frame")
                    return
            
            # Log active worker count before submission
            rospy.loginfo(f"[FOUNDATIONPOSE] Submitting task (active before: {active_count}/{self.max_parallel_workers}, max_queue: {max_queue_size})")
            
            # Submit pose estimation to thread pool (non-blocking)
            # Make deep copies to avoid race conditions with shared numpy arrays
            rgb_copy = rgb_image.copy()
            depth_copy = depth_image.copy()
            mask_copy = ob_mask.copy() if ob_mask is not None else None
            header_copy = deepcopy(rgb_msg.header)
            
            future = self.pose_executor.submit(
                self._process_pose_estimation_async,
                rgb_copy,
                depth_copy,
                header_copy,
                mask_copy
            )
            
            # Store future with timestamp for tracking (update active_count after adding)
            with self.pose_lock:
                self.active_poses[rgb_msg.header.stamp] = future
                active_count_after = len([f for f in self.active_poses.values() if not f.done()])
            
            # Log immediately after submission
            rospy.loginfo(f"[FOUNDATIONPOSE] Submitted task (active: {active_count_after}/{self.max_parallel_workers})")
            
            # Clean up completed futures periodically (but don't do it every time to reduce lock contention)
            if not hasattr(self, '_last_cleanup_time'):
                self._last_cleanup_time = 0
            current_time = rospy.get_time()
            if current_time - self._last_cleanup_time >= 0.5:  # Cleanup every 0.5 seconds
                self._cleanup_completed_futures()
                self._last_cleanup_time = current_time
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback_with_mask: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    # ========================================================================
    # Performance Monitoring
    # ========================================================================
    
    def get_gpu_usage(self):
        """Get GPU memory and utilization usage."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                memory_allocated_pct = (memory_allocated / memory_total) * 100
                memory_reserved_pct = (memory_reserved / memory_total) * 100
                
                # Try to get utilization using pynvml if available
                gpu_util = None
                if self.pynvml_initialized:
                    try:
                        import pynvml
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                    except Exception:
                        # pynvml failed, that's okay
                        pass
                
                return {
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved,
                    'memory_total_gb': memory_total,
                    'memory_allocated_pct': memory_allocated_pct,
                    'memory_reserved_pct': memory_reserved_pct,
                    'gpu_util_pct': gpu_util
                }
        except Exception:
            pass
        return None
    
    def get_cpu_usage(self):
        """Get CPU usage percentage."""
        if PSUTIL_AVAILABLE:
            try:
                # Get CPU usage for current process
                process = psutil.Process(os.getpid())
                cpu_percent = process.cpu_percent(interval=0.1)
                # Get system-wide CPU usage
                cpu_system = psutil.cpu_percent(interval=0.1)
                # Get memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # MB
                return {
                    'cpu_percent': cpu_percent,
                    'cpu_system': cpu_system,
                    'memory_mb': memory_mb
                }
            except Exception:
                pass
        return None
    
    def _save_metrics(self, elapsed_time, gpu_info, cpu_info):
        """Save performance metrics to JSON file."""
        try:
            import json
            import rospy
            
            metric_entry = {
                'timestamp': rospy.get_time(),
                'time_ms': elapsed_time * 1000.0,
                'time_s': elapsed_time
            }
            
            if gpu_info:
                metric_entry.update({
                    'gpu_memory_allocated_gb': gpu_info.get('memory_allocated_gb', 0),
                    'gpu_memory_reserved_gb': gpu_info.get('memory_reserved_gb', 0),
                    'gpu_memory_total_gb': gpu_info.get('memory_total_gb', 0),
                    'gpu_memory_allocated_pct': gpu_info.get('memory_allocated_pct', 0),
                    'gpu_memory_reserved_pct': gpu_info.get('memory_reserved_pct', 0),
                    'gpu_utilization_pct': gpu_info.get('gpu_util_pct', 0) if gpu_info.get('gpu_util_pct') is not None else 0
                })
            
            if cpu_info:
                metric_entry.update({
                    'cpu_process_pct': cpu_info.get('cpu_percent', 0),
                    'cpu_system_pct': cpu_info.get('cpu_system', 0),
                    'memory_mb': cpu_info.get('memory_mb', 0)
                })
            
            with self.pose_lock:
                self.metrics_data.append(metric_entry)
                
                # Save to file periodically (every 10 entries to reduce I/O)
                if len(self.metrics_data) % 10 == 0:
                    with open(self.metrics_file, 'w') as f:
                        json.dump(self.metrics_data, f, indent=2)
        except Exception as e:
            rospy.logwarn(f"Failed to save metrics: {e}")
    
    def print_performance_metrics(self, elapsed_time, gpu_info, cpu_info, estimation_type="FoundationPose Pose Estimation"):
        """Print performance metrics in a formatted way."""
        rospy.loginfo("=" * 80)
        rospy.loginfo(f"PERFORMANCE METRICS - {estimation_type.upper()}")
        rospy.loginfo("=" * 80)
        rospy.loginfo(f"Time: {elapsed_time*1000:.2f} ms ({elapsed_time:.3f} s)")
        
        if gpu_info:
            rospy.loginfo(f"GPU Memory: {gpu_info['memory_allocated_gb']:.2f} GB / {gpu_info['memory_total_gb']:.2f} GB ({gpu_info['memory_allocated_pct']:.1f}%)")
            rospy.loginfo(f"GPU Reserved: {gpu_info['memory_reserved_gb']:.2f} GB ({gpu_info['memory_reserved_pct']:.1f}%)")
            if gpu_info['gpu_util_pct'] is not None:
                rospy.loginfo(f"GPU Utilization: {gpu_info['gpu_util_pct']:.1f}%")
        
        if cpu_info:
            rospy.loginfo(f"CPU (Process): {cpu_info['cpu_percent']:.1f}%")
            rospy.loginfo(f"CPU (System): {cpu_info['cpu_system']:.1f}%")
            rospy.loginfo(f"Memory (Process): {cpu_info['memory_mb']:.1f} MB")
        
        rospy.loginfo("=" * 80)
    
    # ========================================================================
    # Parallel Processing Helpers
    # ========================================================================
    
    def _cleanup_completed_futures(self):
        """Remove completed futures from active_poses dictionary."""
        with self.pose_lock:
            completed = [stamp for stamp, future in self.active_poses.items() if future.done()]
            for stamp in completed:
                del self.active_poses[stamp]
            
            # Log active worker count when tasks complete (throttled)
            active_count = len([f for f in self.active_poses.values() if not f.done()])
            if not hasattr(self, '_last_completion_log_time'):
                self._last_completion_log_time = 0
            current_time = rospy.get_time()
            if current_time - self._last_completion_log_time >= 1.0:  # Log every 1 second on completion
                if len(completed) > 0 or active_count > 0:  # Only log if there's activity
                    rospy.loginfo(f"[FOUNDATIONPOSE] Active workers: {active_count}/{self.max_parallel_workers} (completed: {len(completed)}, queue: {active_count})")
                    self._last_completion_log_time = current_time
    
    def _process_pose_estimation_async(self, rgb_image, depth_image, header, ob_mask=None):
        """
        Thread-safe async version of process_pose_estimation.
        This runs in a worker thread and handles pose estimation without blocking the main callback.
        
        Args:
            rgb_image: RGB image as numpy array (already copied)
            depth_image: Depth image as numpy array (already copied)
            header: ROS message header (already deep copied)
            ob_mask: Optional object mask (already copied)
        """
        try:
            # Use the existing process_pose_estimation logic but ensure thread safety
            if self.camera_K is None:
                rospy.logwarn("[FOUNDATIONPOSE] [ASYNC] Camera matrix not available yet")
                return
            
            # Convert depth to meters if needed
            if depth_image.dtype != np.float32:
                depth_image = depth_image.astype(np.float32)
            else:
                depth_image = depth_image.copy()
            
            # Validate and clean depth image
            invalid_mask = (depth_image < 0.001) | (depth_image >= self.depth_max)
            depth_image[invalid_mask] = 0.0
            
            # Validate mask
            if ob_mask is None:
                valid_depth = (depth_image > self.depth_min) & (depth_image < self.depth_max) & (depth_image >= 0.001)
                ob_mask = valid_depth.astype(bool)
                mask_pixels = ob_mask.sum()
                H, W = depth_image.shape[:2]
                min_mask_pixels = max(500, int(W * H * 0.005))
                if mask_pixels < min_mask_pixels:
                    rospy.logwarn_throttle(5.0, f"[FOUNDATIONPOSE] Mask too small ({mask_pixels} pixels < {min_mask_pixels}). Skipping.")
                    return
            else:
                if ob_mask.dtype != bool:
                    ob_mask = ob_mask.astype(bool)
                mask_pixels = ob_mask.sum()
                H, W = ob_mask.shape[:2]
                min_mask_pixels = max(500, int(W * H * 0.005))
                if mask_pixels < min_mask_pixels:
                    rospy.logwarn_throttle(2.0, f"[FOUNDATIONPOSE] Mask too small ({mask_pixels} pixels < {min_mask_pixels}). Skipping.")
                    return
                # else:
                #     rospy.loginfo(f"[FOUNDATIONPOSE] Valid mask: {mask_pixels} pixels (required: {min_mask_pixels})")
            
            # Clear CUDA cache before pose registration
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            
            # rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Performing pose registration... (mask pixels: {ob_mask.sum() if ob_mask is not None else 'N/A'})")
            
            # Get initial resource usage
            gpu_info_before = self.get_gpu_usage()
            cpu_info_before = self.get_cpu_usage()
            
            # Start timing
            start_time = time.time()
            
            try:
                image_timestamp = header.stamp
                processing_start_time = rospy.Time.now()
                time_since_image = (processing_start_time - image_timestamp).to_sec()
                if time_since_image > 0.1:
                    rospy.logdebug(f"[FOUNDATIONPOSE] [ASYNC] Processing image from {time_since_image*1000:.1f}ms ago")
                
                # Try running without lock - PyTorch CUDA operations are generally thread-safe
                # If we get CUDA errors, we can add the lock back
                active_before_gpu = len([f for f in self.active_poses.values() if not f.done()])
                rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Starting registration (active: {active_before_gpu}/{self.max_parallel_workers})")
                pose = self.estimator.register(
                    K=self.camera_K,
                    rgb=rgb_image,
                    depth=depth_image,
                    ob_mask=ob_mask,
                    iteration=self.est_refine_iter
                )
                active_after_gpu = len([f for f in self.active_poses.values() if not f.done()])
                rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Registration completed (active: {active_after_gpu}/{self.max_parallel_workers})")
                
                # Log registration result
                if pose is None:
                    rospy.logwarn("[FOUNDATIONPOSE] [ASYNC] Registration returned None - pose estimation failed")
                    return
                # else:
                #     pose_pos = pose[:3, 3]
                #     rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Registration successful! Pose position: ({pose_pos[0]:.3f}, {pose_pos[1]:.3f}, {pose_pos[2]:.3f})")
                
                # End timing
                elapsed_time = time.time() - start_time
                processing_end_time = rospy.Time.now()
                total_delay = (processing_end_time - image_timestamp).to_sec()
                
                # Get resource usage after estimation
                gpu_info_after = self.get_gpu_usage()
                cpu_info_after = self.get_cpu_usage()
                
                # Use after values for reporting
                gpu_info = gpu_info_after if gpu_info_after else gpu_info_before
                cpu_info = cpu_info_after if cpu_info_after else cpu_info_before
                
                # Print performance metrics (throttled)
                current_time = rospy.get_time()
                estimation_type = "FoundationPose Pose Estimation [ASYNC]"
                if current_time - self._last_performance_print_time >= self.COMPARISON_PRINT_INTERVAL:
                    self.print_performance_metrics(elapsed_time, gpu_info, cpu_info, estimation_type)
                    self._last_performance_print_time = current_time
                else:
                    rospy.logdebug(f"{estimation_type}: {elapsed_time*1000:.1f} ms")
                
                # Save metrics to file
                self._save_metrics(elapsed_time, gpu_info, cpu_info)
                
                # Validate pose position before processing
                pose_position = pose[:3, 3]
                pose_depth = pose_position[2]
                
                # Check if pose is at or near camera origin
                if np.linalg.norm(pose_position) < 0.05:
                    rospy.logwarn_throttle(5.0, f"[FOUNDATIONPOSE] [ASYNC] Pose at camera origin, skipping.")
                    return
                
                # Check if depth is reasonable
                if pose_depth < 0.1 or pose_depth > 5.0:
                    rospy.logwarn_throttle(5.0, f"[FOUNDATIONPOSE] [ASYNC] Unreasonable depth ({pose_depth:.3f}m), skipping.")
                    return
                
                # Apply to_origin transformation
                pose_corrected = pose.copy()
                if hasattr(self, 'to_origin') and self.to_origin is not None:
                    try:
                        pose_corrected = pose @ np.linalg.inv(self.to_origin)
                    except Exception as e:
                        rospy.logwarn(f"[FOUNDATIONPOSE] [ASYNC] Failed to apply to_origin transform: {e}")
                        return
                
                # Thread-safe: Add pose to buffer and publish
                with self.pose_lock:
                    # Add to circular buffer
                    self._add_pose_to_buffer(pose_corrected.copy(), deepcopy(header))
                    
                    # Get consensus pose from buffer
                    consensus_result = self._get_consensus_pose()
                    
                    # Use consensus pose if available, otherwise use latest
                    if consensus_result is not None:
                        consensus_pose, consensus_header, consensus_count = consensus_result
                        pose_to_publish = consensus_pose
                        header_to_use = consensus_header
                    elif len(self.pose_buffer) > 0:
                        latest_pose, latest_header = self.pose_buffer[-1]
                        pose_to_publish = latest_pose.copy()
                        header_to_use = latest_header
                    else:
                        pose_to_publish = pose_corrected
                        header_to_use = header
                    
                    # Check pose quality before publishing
                    quality_check = self._check_pose_quality(pose_to_publish)
                    if quality_check is not None:
                        pos_error, orient_error = quality_check
                        if pos_error <= self.publish_position_error_threshold and orient_error <= self.publish_orientation_error_threshold:
                            if self._check_temporal_consistency(pose_to_publish):
                                pose_pos = pose_to_publish[:3, 3]
                                # rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] PUBLISHING POSE - Camera frame: ({pose_pos[0]:.3f}, {pose_pos[1]:.3f}, {pose_pos[2]:.3f})m")
                                self.publish_pose(pose_to_publish, header_to_use)
                                self.publish_markers(pose_to_publish, header_to_use)
                                self.last_published_pose = pose_to_publish.copy()
                                self.last_published_time = rospy.get_time()
                                # if consensus_result is not None:
                                #     rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Published consensus pose: consensus={consensus_count}/{len(self.pose_buffer)}")
                                # else:
                                #     rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Published latest pose (no consensus)")
                    else:
                        # No ground truth available - apply strict validation
                        if self._validate_pose_without_gt(pose_to_publish):
                            pose_pos = pose_to_publish[:3, 3]
                            # rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] PUBLISHING POSE (no GT) - Camera frame: ({pose_pos[0]:.3f}, {pose_pos[1]:.3f}, {pose_pos[2]:.3f})m")
                            self.publish_pose(pose_to_publish, header_to_use)
                            self.publish_markers(pose_to_publish, header_to_use)
                            self.last_published_pose = pose_to_publish.copy()
                            self.last_published_time = rospy.get_time()
                            # if consensus_result is not None:
                            #     rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Published consensus pose (no GT): consensus={consensus_count}/{len(self.pose_buffer)}")
                            # else:
                            #     rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Published latest pose (no consensus, no GT)")
                    
                    # Always compare with ground truth for logging
                    self.compare_with_ground_truth(pose_corrected, header)
                
            except Exception as e:
                elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                rospy.logerr(f"[FOUNDATIONPOSE] [ASYNC] Pose registration failed (took {elapsed_time*1000:.1f} ms): {e}")
                import traceback
                rospy.logerr(traceback.format_exc())
                # Aggressive memory cleanup after failure
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.empty_cache()
                        rospy.loginfo("[FOUNDATIONPOSE] [ASYNC] Performed aggressive CUDA memory cleanup")
                except Exception:
                    pass
                return  # Skip this frame, try again next time
        
        except Exception as e:
            rospy.logerr(f"[FOUNDATIONPOSE] [ASYNC] Error in async pose estimation: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
        finally:
            # Clean up completed future
            with self.pose_lock:
                if header.stamp in self.active_poses:
                    del self.active_poses[header.stamp]
                remaining_active = len([f for f in self.active_poses.values() if not f.done()])
                rospy.loginfo(f"[FOUNDATIONPOSE] [ASYNC] Task completed, remaining active: {remaining_active}/{self.max_parallel_workers}")
    
    # ========================================================================
    # Pose Estimation
    # ========================================================================
    
    def process_pose_estimation(self, rgb_image, depth_image, header, ob_mask=None):
        """
        Process pose estimation using FoundationPose.
        
        Args:
            rgb_image: RGB image as numpy array
            depth_image: Depth image as numpy array (in meters)
            header: ROS message header
            ob_mask: Optional object mask (boolean array)
        """
        if self.camera_K is None:
            rospy.logwarn("Camera matrix not available yet")
            return
        
        try:
            # Convert depth to meters if needed and make a writable copy
            # The array from ROS message might be read-only, so we need to copy it
            if depth_image.dtype != np.float32:
                depth_image = depth_image.astype(np.float32)
            else:
                # Make a writable copy even if dtype is already float32
                depth_image = depth_image.copy()
            
            # Validate and clean depth image (matching FoundationPose expectations)
            # FoundationPose expects invalid depths (< 0.001) to be set to 0
            # Also filter depths beyond max range (FoundationPose uses zfar=np.inf in demo, but we use depth_max)
            invalid_mask = (depth_image < 0.001) | (depth_image >= self.depth_max)
            depth_image[invalid_mask] = 0.0
            
            # Create mask if not provided
            if ob_mask is None:
                # Use tighter depth bounds to focus on object (table is at ~0.6m, object is on table)
                # Only consider valid depths (not zero/invalid)
                valid_depth = (depth_image > self.depth_min) & (depth_image < self.depth_max) & (depth_image >= 0.001)
                ob_mask = valid_depth.astype(bool)
                
                # Validate mask has sufficient pixels (object must be visible)
                # Stricter to reduce false positives
                mask_pixels = ob_mask.sum()
                H, W = depth_image.shape[:2]
                min_mask_pixels = max(500, int(W * H * 0.005))  # At least 500 pixels or 0.5% of image (stricter)
                if mask_pixels < min_mask_pixels:
                    rospy.logwarn_throttle(5.0, f"Mask too small ({mask_pixels} pixels < {min_mask_pixels}). Object not visible, skipping pose estimation.")
                    return  # Don't attempt pose estimation if mask is invalid
                else:
                    rospy.logwarn_once(f"No mask provided, using depth-based heuristic (depth: {self.depth_min}-{self.depth_max}m, {ob_mask.sum()} pixels). Consider using proper segmentation.")
            else:
                # Validate provided mask (object must be visible)
                if ob_mask.dtype != bool:
                    ob_mask = ob_mask.astype(bool)
                # Stricter to reduce false positives
                mask_pixels = ob_mask.sum()
                # Estimate image size from mask shape
                H, W = ob_mask.shape[:2]
                min_mask_pixels = max(500, int(W * H * 0.005))  # At least 500 pixels or 0.5% of image (stricter)
                if mask_pixels < min_mask_pixels:
                    rospy.logwarn_throttle(2.0, f"[FOUNDATIONPOSE] Mask too small ({mask_pixels} pixels < {min_mask_pixels} required). Skipping pose estimation.")
                    return  # Don't attempt pose estimation if mask is invalid
                # else:
                #     rospy.loginfo(f"[FOUNDATIONPOSE] Valid mask: {mask_pixels} pixels (required: {min_mask_pixels}, {mask_pixels/min_mask_pixels:.1f}x threshold)")
            
            # Estimate pose
            # Clear CUDA cache before pose registration to reduce memory fragmentation
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass  # Ignore if torch not available
            
            # Always perform registration (tracking disabled)
            # rospy.loginfo(f"[FOUNDATIONPOSE] Performing pose registration... (mask pixels: {ob_mask.sum() if ob_mask is not None else 'N/A'})")
            
            # Get initial resource usage
            gpu_info_before = self.get_gpu_usage()
            cpu_info_before = self.get_cpu_usage()
            
            # Start timing
            start_time = time.time()
            
            try:
                # Log image timestamp for debugging
                image_timestamp = header.stamp
                processing_start_time = rospy.Time.now()
                time_since_image = (processing_start_time - image_timestamp).to_sec()
                if time_since_image > 0.1:
                    rospy.logdebug(f"FoundationPose: Processing image from {time_since_image*1000:.1f}ms ago "
                                  f"(image: {image_timestamp.secs}.{image_timestamp.nsecs}, now: {processing_start_time.secs}.{processing_start_time.nsecs})")
                
                pose = self.estimator.register(
                    K=self.camera_K,
                    rgb=rgb_image,
                    depth=depth_image,
                    ob_mask=ob_mask,
                    iteration=self.est_refine_iter
                )
                
                # Log registration result
                if pose is None:
                    rospy.logwarn("[FOUNDATIONPOSE] Registration returned None - pose estimation failed")
                else:
                    pose_pos = pose[:3, 3]
                    # rospy.loginfo(f"[FOUNDATIONPOSE] Registration successful! Pose position in camera frame: ({pose_pos[0]:.3f}, {pose_pos[1]:.3f}, {pose_pos[2]:.3f})")
                
                # End timing
                elapsed_time = time.time() - start_time
                processing_end_time = rospy.Time.now()
                total_delay = (processing_end_time - image_timestamp).to_sec()
                if total_delay > 0.2:
                    rospy.logwarn_throttle(2.0, f"FoundationPose processing took {elapsed_time*1000:.1f}ms, "
                                              f"total delay from image capture: {total_delay*1000:.1f}ms. "
                                              f"Robot may have moved during processing.")
                
                # Get resource usage after estimation
                gpu_info_after = self.get_gpu_usage()
                cpu_info_after = self.get_cpu_usage()
                
                # Use after values for reporting (they represent peak usage during computation)
                gpu_info = gpu_info_after if gpu_info_after else gpu_info_before
                cpu_info = cpu_info_after if cpu_info_after else cpu_info_before
                
                # Print performance metrics (throttled to match pose comparison interval)
                current_time = rospy.get_time()
                estimation_type = "FoundationPose Pose Estimation"
                if current_time - self._last_performance_print_time >= self.COMPARISON_PRINT_INTERVAL:
                    self.print_performance_metrics(elapsed_time, gpu_info, cpu_info, estimation_type)
                    self._last_performance_print_time = current_time
                else:
                    rospy.logdebug(f"{estimation_type}: {elapsed_time*1000:.1f} ms")
                
                self.pose_initialized = True
                rospy.loginfo("Pose registration successful")
            except Exception as e:
                elapsed_time = time.time() - start_time
                rospy.logerr(f"Pose registration failed (took {elapsed_time*1000:.1f} ms): {e}")
                # Aggressive memory cleanup after failure
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        # Clear cache multiple times
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        # Force Python garbage collection
                        gc.collect()
                        torch.cuda.empty_cache()
                        rospy.loginfo("Performed aggressive CUDA memory cleanup")
                except Exception:
                    pass
                rospy.logwarn("Will retry registration on next frame...")
                # Add a delay to let memory free up before retry
                rospy.sleep(1.0)
                return  # Skip this frame, try again next time
            
            # If we reach here, pose was successfully assigned
            # Only proceed if pose is valid (not None)
            if pose is None:
                rospy.logwarn_throttle(2.0, "Pose estimation returned None, skipping publication.")
                return
            
            # Don't store last_pose - always use latest detection only
            # self.last_pose = pose  # Removed to prevent republishing old poses
            
            # Validate pose position before processing
            # FoundationPose should return a pose with reasonable position (not at camera origin)
            pose_position = pose[:3, 3]
            pose_depth = pose_position[2]  # Z coordinate in camera frame (forward)
            
            # Check if pose is at or near camera origin (this would cause it to appear at robot head)
            if np.linalg.norm(pose_position) < 0.05:  # Less than 5cm from origin
                rospy.logwarn_throttle(5.0, f"Pose position is at/near camera origin ({pose_position[0]:.4f}, {pose_position[1]:.4f}, {pose_position[2]:.4f}). "
                                          f"This will cause pose to appear at robot head. Skipping pose estimation.")
                return  # Skip this pose - it's invalid
            
            # Check if depth is reasonable (object should be in front of camera, not too close/far)
            if pose_depth < 0.1 or pose_depth > 5.0:
                rospy.logwarn_throttle(5.0, f"Pose depth is unreasonable ({pose_depth:.3f}m). "
                                          f"Object should be between 0.1m and 5.0m from camera. Skipping pose estimation.")
                return  # Skip this pose - depth is invalid
            
            # Log pose position for debugging
            rospy.logdebug(f"FoundationPose returned pose in camera frame: pos=({pose_position[0]:.3f}, {pose_position[1]:.3f}, {pose_position[2]:.3f}), depth={pose_depth:.3f}m")
            
            # Apply to_origin transformation to match run_demo.py behavior
            # FoundationPose returns pose in centered mesh coordinate system (center at origin)
            # We need to transform it back to original mesh coordinate system (with oriented bounds alignment)
            # This is the same transformation applied in run_demo.py line 69: center_pose = pose@np.linalg.inv(to_origin)
            # Note: to_origin moves the center of the bounding box to the origin, so inv(to_origin) moves origin back to center
            # The pose position should already be at the center (origin in centered frame)
            pose_corrected = pose @ np.linalg.inv(self.to_origin)
            
            # Validate pose after correction too
            pose_corrected_position = pose_corrected[:3, 3]
            pose_corrected_depth = pose_corrected_position[2]
            if np.linalg.norm(pose_corrected_position) < 0.05:
                rospy.logwarn_throttle(5.0, f"Pose position after correction is at/near origin ({pose_corrected_position[0]:.4f}, {pose_corrected_position[1]:.4f}, {pose_corrected_position[2]:.4f}). Skipping.")
                return
            
            # Debug: Log the to_origin translation to understand the center offset
            if self.debug >= 2 and not hasattr(self, '_to_origin_translation_logged'):
                to_origin_translation = self.to_origin[:3, 3]
                inv_to_origin_translation = np.linalg.inv(self.to_origin)[:3, 3]
                rospy.loginfo(f">>> [MESH_CENTER] to_origin translation (moves center to origin): {to_origin_translation}")
                rospy.loginfo(f">>> [MESH_CENTER] inv(to_origin) translation (moves origin to center): {inv_to_origin_translation}")
                rospy.loginfo(f">>> [MESH_CENTER] Pose position before inv(to_origin): {pose[:3, 3]}")
                rospy.loginfo(f">>> [MESH_CENTER] Pose position after inv(to_origin): {pose_corrected[:3, 3]}")
                rospy.loginfo(f">>> [MESH_CENTER] Mesh extents: {self.extents}")
                self._to_origin_translation_logged = True
            
            # Apply coordinate frame correction in camera frame BEFORE error calculation
            # Check if correction should be applied based on Z-axis direction
            correction_applied_cam = False
            if self.coord_correction_enabled and self.coord_correction_matrix is not None:
                from scipy.spatial.transform import Rotation
                R_cam = pose_corrected[:3, :3]
                
                # Check if blue axis (Z-axis) is pointing down in camera frame
                # Z-axis in object frame is [0, 0, 1], transform to camera frame
                z_axis_obj = np.array([0, 0, 1])
                z_axis_cam = R_cam @ z_axis_obj
                
                # In camera frame, "down" typically means negative Y or we check if Z-axis points downward
                # Check if the Z-axis is pointing more downward than upward (dot product with camera's down direction)
                # Camera frame: X right, Y down, Z forward
                # So "down" in camera frame is negative Y direction [0, -1, 0]
                camera_down = np.array([0, -1, 0])
                z_axis_dot_down = np.dot(z_axis_cam, camera_down)
                
                # Apply correction only if Z-axis is pointing down (dot product > threshold, e.g., > 0.5)
                if z_axis_dot_down > 0.5:
                    # Apply correction: R_corrected = R_corr @ R_cam
                    R_corrected_cam = self.coord_correction_matrix @ R_cam
                    # Update pose_corrected with corrected rotation
                    pose_corrected[:3, :3] = R_corrected_cam
                    # Position remains unchanged
                    correction_applied_cam = True
                # Removed verbose debug prints for camera frame correction
            
            # Store flag for error calculation - if correction wasn't applied in camera frame but will be in odom frame,
            # we need to apply it for accurate error calculation
            self._correction_applied_cam = correction_applied_cam
            
            # NOTE: Coordinate frame correction is also applied AFTER transforming to odom frame
            # in publish_pose() and publish_markers() for RViz visualization consistency
            
            # Debug: Log transformation details (throttled, only once)
            if self.debug >= 2 and not hasattr(self, '_to_origin_debug_logged'):
                from scipy.spatial.transform import Rotation
                R_before = pose[:3, :3]
                R_after = pose_corrected[:3, :3]
                rospy.loginfo("="*60)
                rospy.loginfo("POSE TRANSFORMATION DEBUG")
                rospy.loginfo("="*60)
                rospy.loginfo(f"to_origin transformation:\n{self.to_origin}")
                rospy.loginfo(f"Pose before to_origin (from FoundationPose):\n{pose}")
                rospy.loginfo(f"Pose after to_origin:\n{pose_corrected}")
                rot_before = Rotation.from_matrix(R_before)
                rot_after = Rotation.from_matrix(R_after)
                rospy.loginfo(f"Rotation before (euler ZYX): {rot_before.as_euler('zyx', degrees=True)}")
                rospy.loginfo(f"Rotation after (euler ZYX): {rot_after.as_euler('zyx', degrees=True)}")
                rospy.loginfo("="*60)
                self._to_origin_debug_logged = True

            # Save the visualization with the box (when debug >= 2)
            if self.debug >= 2:
                # Draw 3D box and axes on the image (same as run_demo.py lines 70-71)
                # Note: pose_corrected is already the center_pose (pose @ inv(to_origin))
                # Work in RGB throughout - only convert to BGR temporarily for draw_posed_3d_box (uses cv2.line which expects BGR)
                vis_rgb = rgb_image.copy()
                # Convert to BGR only for draw_posed_3d_box (cv2.line expects BGR)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                vis_bgr = draw_posed_3d_box(self.camera_K, img=vis_bgr, ob_in_cam=pose_corrected, bbox=self.bbox)
                # Convert back to RGB for draw_xyz_axis (expects RGB when is_input_rgb=True, returns RGB)
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                vis = draw_xyz_axis(vis_rgb, ob_in_cam=pose_corrected, scale=0.1, K=self.camera_K, thickness=3, transparency=0, is_input_rgb=True)
                
                # Save the visualization image in RGB format
                timestamp = int(time.time() * 1000)  # Use timestamp as filename
                image_path = os.path.join(self.debug_dir, f'{timestamp}.png')
                imageio.imwrite(image_path, vis)
                rospy.loginfo_throttle(2.0, f"Saved visualization to: {image_path}")
                
                # Debug: Log pose information for orientation investigation
                if not hasattr(self, '_orientation_debug_logged'):
                    from scipy.spatial.transform import Rotation
                    R = pose_corrected[:3, :3]
                    t = pose_corrected[:3, 3]
                    rot = Rotation.from_matrix(R)
                    euler_zyx = rot.as_euler('zyx', degrees=True)
                    quat = rot.as_quat()  # [x, y, z, w]
                    rospy.loginfo("="*60)
                    rospy.loginfo("POSE ORIENTATION DEBUG (track_vis)")
                    rospy.loginfo("="*60)
                    rospy.loginfo(f"Position (camera frame): [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
                    rospy.loginfo(f"Quaternion (camera frame): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
                    rospy.loginfo(f"Euler ZYX (degrees): [{euler_zyx[0]:.2f}, {euler_zyx[1]:.2f}, {euler_zyx[2]:.2f}]")
                    rospy.loginfo(f"Rotation matrix:\n{R}")
                    rospy.loginfo("="*60)
                    self._orientation_debug_logged = True
            
            # Add pose to circular buffer for consensus voting (with header for timestamp)
            self._add_pose_to_buffer(pose_corrected, header)
            
            # Get consensus pose from buffer (returns pose and header)
            consensus_result = self._get_consensus_pose()
            
            # Use consensus pose if available, otherwise use latest pose from buffer
            if consensus_result is not None:
                consensus_pose, consensus_header, consensus_count = consensus_result
                pose_to_publish = consensus_pose
                header_to_use = consensus_header  # Use header from consensus (timestamp from latest pose in cluster)
            elif len(self.pose_buffer) > 0:
                # No consensus, use latest pose (most recent in buffer)
                latest_pose, latest_header = self.pose_buffer[-1]
                pose_to_publish = latest_pose.copy()
                header_to_use = latest_header  # Use header from latest pose
            else:
                # Buffer is empty (shouldn't happen), use current pose
                pose_to_publish = pose_corrected
                header_to_use = header  # Use current header
            
            # Check pose quality before publishing (only publish if error is acceptable)
            quality_check = self._check_pose_quality(pose_to_publish)
            if quality_check is not None:
                pos_error, orient_error = quality_check
                if pos_error <= self.publish_position_error_threshold and orient_error <= self.publish_orientation_error_threshold:
                    # Quality is good, but also check temporal consistency
                    if self._check_temporal_consistency(pose_to_publish):
                        # Use the correct header (with timestamp from when pose was captured)
                        pose_pos = pose_to_publish[:3, 3]
                        # rospy.loginfo(f"[FOUNDATIONPOSE] PUBLISHING POSE - Camera frame: ({pose_pos[0]:.3f}, {pose_pos[1]:.3f}, {pose_pos[2]:.3f})m")
                        self.publish_pose(pose_to_publish, header_to_use)
                        self.publish_markers(pose_to_publish, header_to_use)
                        # Update last published pose
                        self.last_published_pose = pose_to_publish.copy()
                        self.last_published_time = rospy.get_time()
                        # if consensus_result is not None:
                        #     rospy.loginfo(f"[FOUNDATIONPOSE] Published consensus pose: pos_error={pos_error:.4f}m, orient_error={orient_error:.2f}°, consensus={consensus_count}/{len(self.pose_buffer)}")
                        # else:
                        #     rospy.loginfo(f"[FOUNDATIONPOSE] Published latest pose (no consensus): pos_error={pos_error:.4f}m, orient_error={orient_error:.2f}°")
                    else:
                        rospy.logwarn_throttle(2.0, f"Pose quality good but failed temporal consistency check. Not publishing.")
                else:
                    # Quality is not good enough, don't publish
                    rospy.logwarn_throttle(2.0, f"NOT publishing pose (quality too low): pos_error={pos_error:.4f}m (threshold={self.publish_position_error_threshold:.4f}m), "
                                                f"orient_error={orient_error:.2f}° (threshold={self.publish_orientation_error_threshold:.2f}°)")
            else:  # No ground truth available
                # No ground truth available - apply strict validation before publishing
                # Check temporal consistency and position sanity
                if self._validate_pose_without_gt(pose_to_publish):
                    if not hasattr(self, '_no_gt_publish_warning_logged'):
                        rospy.logwarn("No ground truth available. Publishing pose with strict validation.")
                        self._no_gt_publish_warning_logged = True
                    # Use the correct header (with timestamp from when pose was captured)
                    pose_pos = pose_to_publish[:3, 3]
                    # rospy.loginfo(f"[FOUNDATIONPOSE] PUBLISHING POSE (no GT) - Camera frame: ({pose_pos[0]:.3f}, {pose_pos[1]:.3f}, {pose_pos[2]:.3f})m")
                    self.publish_pose(pose_to_publish, header_to_use)
                    self.publish_markers(pose_to_publish, header_to_use)
                    # Update last published pose
                    self.last_published_pose = pose_to_publish.copy()
                    self.last_published_time = rospy.get_time()
                    # if consensus_result is not None:
                    #     rospy.loginfo(f"[FOUNDATIONPOSE] Published consensus pose (no GT): consensus={consensus_count}/{len(self.pose_buffer)}")
                    # else:
                    #     rospy.loginfo(f"[FOUNDATIONPOSE] Published latest pose (no consensus, no GT): buffer_size={len(self.pose_buffer)}")
                else:
                    rospy.logwarn_throttle(2.0, "Pose validation failed (no GT). Not publishing.")
            
            # Always compare with ground truth for logging/debugging
            self.compare_with_ground_truth(pose_corrected, header)
            
        except Exception as e:
            rospy.logerr(f"Error in pose estimation: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.pose_initialized = False
    
    # ========================================================================
    # Pose Publishing
    # ========================================================================
    
    def publish_pose(self, pose, header):
        """
        Publish pose as PoseStamped message and TF transform.
        Transforms pose from camera frame to odom frame for visualization.
        
        Args:
            pose: 4x4 transformation matrix (object in camera frame)
            header: ROS message header
        """
        from scipy.spatial.transform import Rotation
        
        # Create PoseStamped message in camera frame using helper method
        pose_msg_camera = self._pose_matrix_to_pose_stamped(pose, header, self.frame_id)
        
        # Extract quaternion for TF transform
        R = pose[:3, :3]
        t = pose[:3, 3]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w]
        
        # Transform pose from camera frame to odom frame for visualization
        # Use strict_timestamp=True to ensure we use the exact transform from when image was captured
        # Use original timestamp to avoid movement artifacts
        # The pose is calculated in camera frame at image capture time T1
        # Use the robot position at T1, not the current position
        image_timestamp = header.stamp
        current_time = rospy.Time.now()
        time_diff = (current_time - image_timestamp).to_sec()
        
        if time_diff > 0.1:  # If processing took more than 100ms
            rospy.logwarn_throttle(2.0, f"Pose processing delay: {time_diff*1000:.1f}ms between image capture ({image_timestamp.secs}.{image_timestamp.nsecs}) and transform ({current_time.secs}.{current_time.nsecs})")
        
        # Log pose in camera frame before transform for debugging
        camera_pos = pose[:3, 3]
        rospy.loginfo_throttle(2.0, f"[TRANSFORM] Pose in camera frame ({self.frame_id}): pos=({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f})")
        
        pose_msg = self._transform_pose_with_fallback('odom', pose_msg_camera, timeout=0.5, strict_timestamp=True)
        if pose_msg is None:
            # If strict transform fails, DO NOT use latest transform as it would use wrong robot position
            # Instead, log error and skip publishing to avoid incorrect pose
            rospy.logerr_throttle(5.0, f"Strict timestamp transform failed for image at {image_timestamp.secs}.{image_timestamp.nsecs}. "
                                      f"Cannot transform pose - robot may have moved. Skipping pose publication to avoid drift.")
            return  # Skip publishing to avoid incorrect pose
        
        # Verify the transform actually succeeded and frame_id is correct
        if pose_msg.header.frame_id != 'odom':
            rospy.logerr_throttle(5.0, f"Transform returned pose in wrong frame '{pose_msg.header.frame_id}' instead of 'odom'. Skipping publication.")
            return
        
        # Log pose in odom frame after transform for debugging
        odom_pos = (pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z)
        rospy.loginfo_throttle(2.0, f"[TRANSFORM] Pose in odom frame: pos=({odom_pos[0]:.3f}, {odom_pos[1]:.3f}, {odom_pos[2]:.3f}), "
                      f"frame_id={pose_msg.header.frame_id}, stamp={pose_msg.header.stamp.secs}.{pose_msg.header.stamp.nsecs}")
        
        # Check if pose position seems reasonable (not at origin, not too close to camera)
        if abs(odom_pos[0]) < 0.01 and abs(odom_pos[1]) < 0.01 and abs(odom_pos[2]) < 0.01:
            rospy.logwarn_throttle(5.0, f"WARNING: Transformed pose is at origin (0, 0, 0) in odom frame. This is likely incorrect!")
        if camera_pos[2] < 0.1:  # Depth too small
            rospy.logwarn_throttle(5.0, f"WARNING: Pose depth in camera frame is very small ({camera_pos[2]:.3f}m). This might be incorrect.")
        
        # Apply coordinate frame correction (same as in publish_markers) for consistency
        # Only apply if blue axis (Z-axis) is pointing down
        if self.coord_correction_enabled and self.coord_correction_matrix is not None:
            import numpy as np
            from scipy.spatial.transform import Rotation
            
            # Convert pose_msg to matrix
            quat_before = np.array([
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w
            ])
            R_odom = Rotation.from_quat(quat_before).as_matrix()
            
            # Check if blue axis (Z-axis) is pointing down in odom frame
            # Z-axis in object frame is [0, 0, 1], transform to odom frame
            z_axis_obj = np.array([0, 0, 1])
            z_axis_odom = R_odom @ z_axis_obj
            
            # In odom frame (ROS convention: X forward, Y left, Z up), "down" is negative Z
            # Check if Z-axis is pointing down (negative Z component)
            z_axis_z_component = z_axis_odom[2]
            
            # Apply correction only if Z-axis is pointing down (z_component < -0.5)
            if z_axis_z_component < -0.5:
                # Apply correction: only rotate orientation, keep position unchanged
                R_corrected = self.coord_correction_matrix @ R_odom
                quat_corrected = Rotation.from_matrix(R_corrected).as_quat()
                
                # Update pose_msg with corrected orientation
                pose_msg.pose.orientation.x = quat_corrected[0]
                pose_msg.pose.orientation.y = quat_corrected[1]
                pose_msg.pose.orientation.z = quat_corrected[2]
                pose_msg.pose.orientation.w = quat_corrected[3]
                # Position remains unchanged
        
        # Keep the original image timestamp, not current time
        # The pose represents the object's position at the time the image was captured
        # The transform to odom frame was done using the exact robot position at that time
        # The timestamp should reflect when the measurement was made, not when it was published
        # The transform function should preserve the original timestamp, but ensure it's set correctly
        if pose_msg.header.stamp != header.stamp:
            # If transform changed the timestamp, restore the original image timestamp
            pose_msg.header.stamp = header.stamp
        
        # Publish transformed pose (in odom frame if transform succeeded, camera frame otherwise)
        # This pose includes coordinate frame correction if enabled
        odom_pos = (pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z)
        rospy.loginfo(f"[FOUNDATIONPOSE] PUBLISHED POSE TO TOPIC - Odom frame: ({odom_pos[0]:.3f}, {odom_pos[1]:.3f}, {odom_pos[2]:.3f})m, "
                      f"timestamp: {pose_msg.header.stamp.secs}.{pose_msg.header.stamp.nsecs}")
        self.pose_pub.publish(pose_msg)
        
        # Publish as TF transform (in camera frame - this is the actual object pose)
        transform_tf = TransformStamped()
        transform_tf.header = header
        transform_tf.header.frame_id = self.frame_id
        transform_tf.child_frame_id = self.object_frame_id
        transform_tf.transform.translation.x = t[0]
        transform_tf.transform.translation.y = t[1]
        transform_tf.transform.translation.z = t[2]
        transform_tf.transform.rotation.x = quat[0]
        transform_tf.transform.rotation.y = quat[1]
        transform_tf.transform.rotation.z = quat[2]
        transform_tf.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(transform_tf)
    
    def _check_pose_quality(self, pose_corrected):
        """
        Check both position and orientation errors compared to ground truth.
        
        Args:
            pose_corrected: Pose after to_origin transformation
            
        Returns:
            tuple: (position_error_m, orientation_error_deg) or None if no ground truth available
        """
        if self.gt_pose is None or not hasattr(self, 'gt_pose_camera_frame') or self.gt_pose_camera_frame is None:
            return None
        
        from scipy.spatial.transform import Rotation
        
        # Get ground truth pose in camera frame
        gt_pose_for_comparison = self.gt_pose_camera_frame
        t_gt, quat_gt = self._extract_pose_arrays(gt_pose_for_comparison)
        rot_gt = Rotation.from_quat(quat_gt)
        R_gt = rot_gt.as_matrix()
        
        # Extract estimated pose
        R_est = pose_corrected[:3, :3]
        t_est = pose_corrected[:3, 3]
        
        # Calculate position error
        pos_error = np.linalg.norm(t_est - t_gt)
        
        # Calculate orientation error
        R_diff = R_est @ R_gt.T
        rot_diff = Rotation.from_matrix(R_diff)
        angle_error_rad = np.linalg.norm(rot_diff.as_rotvec())
        angle_error_deg = angle_error_rad * 180 / np.pi
        
        return (pos_error, angle_error_deg)
    
    def _check_temporal_consistency(self, pose):
        """
        Check if pose is consistent with last published pose (not jumping around).
        
        Args:
            pose: Current pose matrix
            
        Returns:
            bool: True if pose is consistent, False if it jumps too much
        """
        if self.last_published_pose is None:
            # First pose, always accept
            return True
        
        # Calculate position difference
        current_pos = pose[:3, 3]
        last_pos = self.last_published_pose[:3, 3]
        pos_diff = np.linalg.norm(current_pos - last_pos)
        
        # Reject if position jumps too much (likely false detection)
        if pos_diff > self.pose_jump_threshold:
            rospy.logwarn_throttle(2.0, f"Pose position jump too large ({pos_diff:.3f}m > {self.pose_jump_threshold:.3f}m). Rejecting.")
            return False
        
        return True
    
    def _validate_pose_without_gt(self, pose):
        """
        Validate pose when ground truth is not available.
        Checks position sanity, depth, and temporal consistency.
        
        Args:
            pose: Pose matrix to validate
            
        Returns:
            bool: True if pose is valid, False otherwise
        """
        # Extract position
        pos = pose[:3, 3]
        
        # Check depth is reasonable (object should be in front of camera, not too far)
        depth = pos[2]  # Z coordinate in camera frame (forward)
        if depth < 0.1 or depth > 5.0:  # Object should be between 10cm and 5m
            rospy.logwarn_throttle(2.0, f"Pose depth invalid ({depth:.3f}m). Rejecting.")
            return False
        
        # Check position is reasonable (not too far from camera center in X/Y)
        # Object should be within reasonable field of view
        if abs(pos[0]) > 2.0 or abs(pos[1]) > 2.0:  # Within 2m in X/Y
            rospy.logwarn_throttle(2.0, f"Pose position too far from camera center (x={pos[0]:.3f}, y={pos[1]:.3f}). Rejecting.")
            return False
        
        # Check temporal consistency
        if not self._check_temporal_consistency(pose):
            return False
        
        return True
    
    # ========================================================================
    # Pose Consensus Buffer
    # ========================================================================
    
    def _print_buffer_contents(self):
        """Print circular buffer contents in a clean, structured format."""
        # Commented out to reduce verbosity - uncomment for debugging
        # if len(self.pose_buffer) == 0:
        #     rospy.loginfo("[BUFFER] Circular buffer is EMPTY")
        #     return
        # 
        # from scipy.spatial.transform import Rotation
        # 
        # rospy.loginfo("=" * 80)
        # rospy.loginfo(f"[BUFFER] Circular Buffer Contents ({len(self.pose_buffer)}/{self.pose_buffer_size} poses)")
        # rospy.loginfo("=" * 80)
        # 
        # for i, (pose, header) in enumerate(self.pose_buffer):
        #     pos = pose[:3, 3]
        #     rot = Rotation.from_matrix(pose[:3, :3])
        #     quat = rot.as_quat()
        #     euler = rot.as_euler('zyx', degrees=True)
        #     timestamp = f"{header.stamp.secs}.{header.stamp.nsecs:09d}"
        #     
        #     rospy.loginfo(f"  [{i}] Timestamp: {timestamp}")
        #     rospy.loginfo(f"       Position (camera): ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m")
        #     rospy.loginfo(f"       Orientation (quat): ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
        #     rospy.loginfo(f"       Orientation (euler ZYX): ({euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°)")
        #     if i < len(self.pose_buffer) - 1:
        #         rospy.loginfo("")  # Empty line between entries
        # 
        # rospy.loginfo("=" * 80)
        pass  # Buffer printing disabled for cleaner output
    
    def _add_pose_to_buffer(self, pose, header):
        """
        Add pose to circular buffer along with its header (timestamp).
        
        Args:
            pose: 4x4 pose matrix to add
            header: ROS message header (contains timestamp from image capture)
        """
        # Store pose with its header (timestamp) for correct transforms
        from copy import deepcopy
        old_size = len(self.pose_buffer)
        removed_pose = None
        
        # Keep buffer size fixed (circular buffer)
        if len(self.pose_buffer) >= self.pose_buffer_size:
            removed_pose = self.pose_buffer.pop(0)  # Remove oldest pose
        
        self.pose_buffer.append((pose.copy(), deepcopy(header)))
        
        # Print buffer contents whenever it changes
        if removed_pose is not None:
            rospy.loginfo(f"[BUFFER] Added new pose, removed oldest (buffer full: {self.pose_buffer_size})")
        else:
            rospy.loginfo(f"[BUFFER] Added new pose (buffer size: {len(self.pose_buffer)}/{self.pose_buffer_size})")
        
        self._print_buffer_contents()
    
    def _compute_pose_similarity(self, pose1, pose2):
        """
        Compute similarity between two poses.
        Returns a score where lower is more similar.
        
        Args:
            pose1: First pose matrix (4x4)
            pose2: Second pose matrix (4x4)
            
        Returns:
            float: Similarity score (lower = more similar)
        """
        from scipy.spatial.transform import Rotation
        
        # Position difference
        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]
        pos_diff = np.linalg.norm(pos1 - pos2)
        
        # Orientation difference
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        R_diff = R1 @ R2.T
        rot_diff = Rotation.from_matrix(R_diff)
        angle_diff_rad = np.linalg.norm(rot_diff.as_rotvec())
        angle_diff_deg = angle_diff_rad * 180 / np.pi
        
        # Combined similarity score (weighted)
        # Position weight: 1.0 per meter
        # Orientation weight: 0.01 per degree (so 15 degrees = 0.15)
        similarity_score = pos_diff + (angle_diff_deg * 0.01)
        
        return similarity_score
    
    def _get_consensus_pose(self):
        """
        Get consensus pose from buffer by clustering similar poses and averaging the largest cluster.
        Finds poses that are close to each other, computes their average, and returns that.
        
        Returns:
            tuple: (consensus_pose, consensus_count) or (None, 0) if buffer is empty or no consensus
        """
        if len(self.pose_buffer) == 0:
            return None
        
        if len(self.pose_buffer) == 1:
            pose, header = self.pose_buffer[0]
            from copy import deepcopy
            return (pose.copy(), deepcopy(header), 1)
        
        from scipy.spatial.transform import Rotation
        
        # Build similarity matrix: for each pair of poses, check if they're similar
        # Only cluster poses that are from similar times (within 0.5 seconds)
        # Don't average poses from different robot positions
        n = len(self.pose_buffer)
        similar_pairs = []
        max_time_diff = rospy.Duration(0.5)  # Only cluster poses within 0.5 seconds
        
        for i in range(n):
            for j in range(i + 1, n):
                pose_i, header_i = self.pose_buffer[i]  # Extract pose and header from tuple
                pose_j, header_j = self.pose_buffer[j]  # Extract pose and header from tuple
                
                # Check timestamp difference first - only cluster poses from similar times
                time_diff = abs((header_i.stamp - header_j.stamp).to_sec())
                if time_diff > max_time_diff.to_sec():
                    # Poses are from different times - don't cluster them
                    # Skip if robot was at different position
                    continue
                
                # Check position difference
                pos_diff = np.linalg.norm(pose_i[:3, 3] - pose_j[:3, 3])
                
                # Check orientation difference
                R_diff = pose_i[:3, :3] @ pose_j[:3, :3].T
                rot_diff = Rotation.from_matrix(R_diff)
                angle_diff_deg = np.linalg.norm(rot_diff.as_rotvec()) * 180 / np.pi
                
                # If similar (within thresholds) AND from similar times, add to similar pairs
                if pos_diff <= self.pose_consensus_threshold and angle_diff_deg <= self.pose_consensus_orientation_threshold:
                    similar_pairs.append((i, j))
        
        # Find the largest cluster of similar poses using union-find (connected components)
        # Each pose is a node, similar pairs are edges
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union all similar pairs
        for i, j in similar_pairs:
            union(i, j)
        
        # Find cluster sizes
        cluster_sizes = {}
        for i in range(n):
            root = find(i)
            if root not in cluster_sizes:
                cluster_sizes[root] = []
            cluster_sizes[root].append(i)
        
        # Find the largest cluster
        largest_cluster = max(cluster_sizes.values(), key=len)
        cluster_size = len(largest_cluster)
        
        # Require at least 50% consensus (majority)
        min_consensus = max(1, len(self.pose_buffer) // 2 + 1)
        
        if cluster_size >= min_consensus:
            # Compute average pose of the largest cluster
            # Average position
            avg_position = np.zeros(3)
            for idx in largest_cluster:
                pose, _ = self.pose_buffer[idx]  # Extract pose from (pose, header) tuple
                avg_position += pose[:3, 3]
            avg_position /= cluster_size
            
            # Average rotation using quaternion averaging
            quaternions = []
            for idx in largest_cluster:
                pose, _ = self.pose_buffer[idx]  # Extract pose from (pose, header) tuple
                R = pose[:3, :3]
                rot = Rotation.from_matrix(R)
                quat = rot.as_quat()  # [x, y, z, w]
                quaternions.append(quat)
            
            # Average quaternions (simple mean, then normalize)
            avg_quat = np.mean(quaternions, axis=0)
            avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize
            
            # Convert back to rotation matrix
            avg_rot = Rotation.from_quat(avg_quat)
            avg_R = avg_rot.as_matrix()
            
            # Build average pose matrix
            avg_pose = np.eye(4)
            avg_pose[:3, :3] = avg_R
            avg_pose[:3, 3] = avg_position
            
            # Use header from the LATEST pose in the cluster (most recent timestamp)
            # Since we only cluster poses from similar times (within 0.5s), using latest timestamp is safe
            # All poses in cluster were calculated when robot was at similar positions
            latest_idx_in_cluster = max(largest_cluster)  # Latest index = most recent pose
            _, consensus_header = self.pose_buffer[latest_idx_in_cluster]
            from copy import deepcopy
            consensus_header_copy = deepcopy(consensus_header)
            
            # Log timestamp range for debugging
            timestamps = [self.pose_buffer[idx][1].stamp for idx in largest_cluster]
            time_range = max([t.to_sec() for t in timestamps]) - min([t.to_sec() for t in timestamps])
            rospy.logdebug(f"Consensus cluster timestamp range: {time_range*1000:.1f}ms (poses from similar times)")
            
            rospy.logdebug(f"Consensus pose found: {cluster_size}/{len(self.pose_buffer)} poses in cluster (threshold: {min_consensus})")
            return (avg_pose, consensus_header_copy, cluster_size)
        else:
            rospy.logdebug(f"No consensus: largest cluster size {cluster_size} < required {min_consensus} (buffer size: {len(self.pose_buffer)}). Will use latest pose.")
            return None
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _pose_matrix_to_pose_stamped(self, pose_matrix, header, frame_id):
        """
        Convert 4x4 pose matrix to PoseStamped message.
        
        Args:
            pose_matrix: 4x4 transformation matrix
            header: ROS message header
            frame_id: Target frame ID
        
        Returns:
            PoseStamped: Pose message
        """
        from scipy.spatial.transform import Rotation
        
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w]
        
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = t[0]
        pose_msg.pose.position.y = t[1]
        pose_msg.pose.position.z = t[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        return pose_msg
    
    def _extract_pose_arrays(self, pose_stamped):
        """
        Extract position and orientation arrays from PoseStamped.
        
        Args:
            pose_stamped: PoseStamped message
        
        Returns:
            tuple: (position_array, quaternion_array) as numpy arrays
        """
        pos = np.array([
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z
        ])
        quat = np.array([
            pose_stamped.pose.orientation.x,
            pose_stamped.pose.orientation.y,
            pose_stamped.pose.orientation.z,
            pose_stamped.pose.orientation.w
        ])
        return pos, quat
    
    def _transform_pose_with_fallback(self, target_frame, pose_msg, timeout=0.5, strict_timestamp=True):
        """
        Transform a PoseStamped message to target_frame with fallback logic.
        
        When strict_timestamp=True, always uses the original image timestamp
        to ensure pose is calculated with the exact camera position when image was captured.
        
        Args:
            target_frame: Target frame ID (e.g., 'map', 'odom', 'head_rgbd_sensor_rgb_frame')
            pose_msg: PoseStamped message to transform (must have correct timestamp from image)
            timeout: Timeout duration for waitForTransform (default: 0.5 seconds)
            strict_timestamp: If True, only use original timestamp (default: True)
        
        Returns:
            PoseStamped: Transformed pose, or None if transform fails
        """
        from copy import deepcopy
        
        source_frame = pose_msg.header.frame_id
        original_timestamp = pose_msg.header.stamp  # Preserve original image timestamp
        
        # If source and target are the same, return as-is
        if source_frame == target_frame:
            return pose_msg
        
        error_msgs = []
        
        # Try direct transform first - ALWAYS use original timestamp
        try:
            # Use the original image timestamp to get exact transform from when image was captured
            # Use robot position at image capture time
            self.tf_listener.waitForTransform(
                target_frame,
                source_frame,
                original_timestamp,  # Use original timestamp, not current time
                rospy.Duration(timeout)
            )
            # Ensure we use the original timestamp for the transform
            pose_msg_with_timestamp = deepcopy(pose_msg)
            pose_msg_with_timestamp.header.stamp = original_timestamp
            
            # Log transform details for debugging (only if significant delay)
            if strict_timestamp:
                current_time = rospy.Time.now()
                time_diff = (current_time - original_timestamp).to_sec()
                if time_diff > 0.1:  # Log if there's significant delay (>100ms)
                    rospy.logdebug(f"Transform: Using robot position at {original_timestamp.secs}.{original_timestamp.nsecs} "
                                  f"(current: {current_time.secs}.{current_time.nsecs}, diff: {time_diff*1000:.1f}ms)")
            
            transformed_pose = self.tf_listener.transformPose(target_frame, pose_msg_with_timestamp)
            
            # Verify frame_id is correct
            if transformed_pose.header.frame_id != target_frame:
                rospy.logerr_throttle(5.0, f"Transform returned wrong frame_id '{transformed_pose.header.frame_id}' instead of '{target_frame}'. "
                                          f"Source frame: {source_frame}, timestamp: {original_timestamp.secs}.{original_timestamp.nsecs}")
                return None  # Return None to indicate failure
            
            # Verify the transformed pose still has the correct timestamp
            if transformed_pose.header.stamp != original_timestamp:
                rospy.logwarn_throttle(5.0, f"Transform changed timestamp from {original_timestamp.secs}.{original_timestamp.nsecs} "
                                          f"to {transformed_pose.header.stamp.secs}.{transformed_pose.header.stamp.nsecs}. Restoring original.")
                transformed_pose.header.stamp = original_timestamp
            
            # Ensure frame_id is explicitly set (double-check)
            transformed_pose.header.frame_id = target_frame
            
            return transformed_pose
        except Exception as e1:
            error_msgs.append(f"Direct transform (with original timestamp): {str(e1)[:150]}")
            
            # If strict_timestamp is False, try with current time as fallback
            # But for pose estimation, we should always use strict_timestamp=True
            if not strict_timestamp:
                try:
                    pose_msg_latest = deepcopy(pose_msg)
                    pose_msg_latest.header.stamp = rospy.Time.now()
                    self.tf_listener.waitForTransform(
                        target_frame,
                        source_frame,
                        rospy.Time(0),  # Latest available for waitForTransform
                        rospy.Duration(timeout)
                    )
                    return self.tf_listener.transformPose(target_frame, pose_msg_latest)
                except Exception as e2:
                    error_msgs.append(f"Direct transform (latest time): {str(e2)[:150]}")
            
            # Try multi-hop transform through intermediate frames
            # Use original timestamp if strict, otherwise try to get latest common time
            intermediate_frames = ['odom', 'base_link', 'base_footprint']
            
            for intermediate in intermediate_frames:
                try:
                    # Transform source -> intermediate -> target
                    if strict_timestamp:
                        # Use original timestamp to ensure we get the transform from when image was captured
                        timestamp_to_use = original_timestamp
                    else:
                        # Try to get latest common time, fallback to original timestamp
                        try:
                            latest_time = self.tf_listener.getLatestCommonTime(source_frame, intermediate)
                            if latest_time is not None and latest_time.to_sec() > 0:
                                timestamp_to_use = latest_time
                            else:
                                timestamp_to_use = original_timestamp
                        except:
                            timestamp_to_use = original_timestamp
                    
                    pose_intermediate = deepcopy(pose_msg)
                    pose_intermediate.header.stamp = timestamp_to_use
                    
                    self.tf_listener.waitForTransform(
                        intermediate,
                        source_frame,
                        timestamp_to_use,
                        rospy.Duration(timeout)
                    )
                    pose_intermediate = self.tf_listener.transformPose(intermediate, pose_intermediate)
                    
                    # Second: intermediate -> target
                    # Get timestamp for second transform
                    if strict_timestamp:
                        timestamp_to_use2 = original_timestamp
                    else:
                        try:
                            latest_time2 = self.tf_listener.getLatestCommonTime(intermediate, target_frame)
                            if latest_time2 is not None and latest_time2.to_sec() > 0:
                                timestamp_to_use2 = latest_time2
                            else:
                                timestamp_to_use2 = original_timestamp
                        except:
                            timestamp_to_use2 = original_timestamp
                    
                    pose_intermediate.header.stamp = timestamp_to_use2
                    self.tf_listener.waitForTransform(
                        target_frame,
                        intermediate,
                        timestamp_to_use2,
                        rospy.Duration(timeout)
                    )
                    result = self.tf_listener.transformPose(target_frame, pose_intermediate)
                    rospy.logdebug(f"Successfully transformed {source_frame} -> {intermediate} -> {target_frame}")
                    return result
                except Exception as e3:
                    error_msgs.append(f"Multi-hop via {intermediate}: {str(e3)[:150]}")
                    continue  # Try next intermediate frame
                
                # All transforms failed - log detailed error
                error_key = f"{source_frame}->{target_frame}"
                if not hasattr(self, '_tf_error_logged'):
                    self._tf_error_logged = set()
                
                if error_key not in self._tf_error_logged:
                    rospy.logwarn(f"Transform failed: {source_frame} -> {target_frame}")
                    for err in error_msgs[-3:]:  # Show last 3 errors
                        rospy.logwarn(f"  {err}")
                    rospy.logwarn(f"  Tried intermediate frames: {intermediate_frames}")
                    # Check if frames exist in TF tree (old tf API doesn't have getFrameStrings)
                    try:
                        # Try to get all frames using tf API
                        # Note: old tf.TransformListener doesn't have getFrameStrings()
                        # We can check if transform exists by trying a very short wait
                        import tf
                        # Just log that we can't easily query the TF tree with old API
                        rospy.logwarn(f"  Note: Cannot easily query TF tree with old tf API")
                        rospy.logwarn(f"  Try: rosrun tf view_frames (to see TF tree structure)")
                    except Exception as e:
                        rospy.logwarn(f"  Could not query TF tree: {e}")
                    self._tf_error_logged.add(error_key)
                
                return None
    
    def _check_gt_topic(self, event):
        """Check if ground truth topic exists and has publishers."""
        import sys
        try:
            topic = f'/{self.object_name}/ground_truth_pose'
            pub_list = rospy.get_published_topics()
            topic_exists = any(t[0] == topic for t in pub_list)
            if topic_exists:
                print(f"[FOUNDATIONPOSE] Ground truth topic exists: {topic}", file=sys.stderr, flush=True)
            else:
                print(f"[FOUNDATIONPOSE] Ground truth topic NOT found: {topic}", file=sys.stderr, flush=True)
                print(f"[FOUNDATIONPOSE] Available topics with '{self.object_name}' or 'ground_truth':", file=sys.stderr, flush=True)
                for t, msg_type in pub_list:
                    if self.object_name.lower() in t.lower() or 'ground_truth' in t.lower() or 'pose' in t.lower():
                        print(f"  - {t} ({msg_type})", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[FOUNDATIONPOSE] Error checking topic: {e}", file=sys.stderr, flush=True)
    
    def gt_pose_callback(self, msg):
        """Callback for ground truth pose from Isaac Sim."""
        self.gt_pose = msg
        
        # Always try to transform to camera frame when new GT pose arrives
        # Keep gt_pose_camera_frame updated
        self.gt_pose_camera_frame = self._transform_pose_with_fallback(self.frame_id, msg, timeout=0.5)
        if self.gt_pose_camera_frame is None:
            if not hasattr(self, '_gt_transform_fail_logged'):
                rospy.logwarn_throttle(5.0, f"Could not transform GT pose to camera frame")
                self._gt_transform_fail_logged = True
        
        if not self.gt_pose_received:
            self.gt_pose_received = True
            import sys
            print(f"[FOUNDATIONPOSE] Received first ground truth pose from Isaac Sim: "
                  f"pos=[{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}], "
                  f"orient=[{msg.pose.orientation.x:.3f}, {msg.pose.orientation.y:.3f}, "
                  f"{msg.pose.orientation.z:.3f}, {msg.pose.orientation.w:.3f}]",
                  file=sys.stderr, flush=True)
            rospy.loginfo("Received first ground truth pose from Isaac Sim")
            # Print ground truth pose in different coordinate systems
            self._print_gt_pose_in_all_frames(msg)
    
    def _print_gt_pose_in_all_frames(self, gt_pose_msg):
        """
        Print ground truth pose in different coordinate systems:
        - World/Map frame (as received from Isaac Sim)
        - Camera frame (transformed for comparison with FoundationPose)
        - Odom frame (via TF transform)
        - Grid coordinates (approximate)
        """
        import sys
        
        # Extract pose in world/map frame (as received)
        gt_world_pos = np.array([
            gt_pose_msg.pose.position.x,
            gt_pose_msg.pose.position.y,
            gt_pose_msg.pose.position.z
        ])
        gt_world_quat = np.array([
            gt_pose_msg.pose.orientation.x,
            gt_pose_msg.pose.orientation.y,
            gt_pose_msg.pose.orientation.z,
            gt_pose_msg.pose.orientation.w
        ])
        
        print("\n" + "="*70, file=sys.stderr, flush=True)
        print("GROUND TRUTH POSE - ALL COORDINATE SYSTEMS", file=sys.stderr, flush=True)
        print("="*70, file=sys.stderr, flush=True)
        
        # 1. World/Map frame (as received from Isaac Sim)
        print(f"[FOUNDATIONPOSE] World/Map frame ({gt_pose_msg.header.frame_id}):", file=sys.stderr, flush=True)
        print(f"  Position:    [{gt_world_pos[0]:.4f}, {gt_world_pos[1]:.4f}, {gt_world_pos[2]:.4f}] m", file=sys.stderr, flush=True)
        print(f"  Orientation: [{gt_world_quat[0]:.4f}, {gt_world_quat[1]:.4f}, {gt_world_quat[2]:.4f}, {gt_world_quat[3]:.4f}]", file=sys.stderr, flush=True)
        
        # 2. Transform to camera frame for comparison with FoundationPose
        # For ground truth comparison, use strict_timestamp=False to allow fallback if exact timestamp unavailable
        gt_camera_pose = self._transform_pose_with_fallback(self.frame_id, gt_pose_msg, timeout=1.0, strict_timestamp=False)
        if gt_camera_pose is not None:
            gt_camera_pos, gt_camera_quat = self._extract_pose_arrays(gt_camera_pose)
            
            print(f"[FOUNDATIONPOSE] Camera frame ({self.frame_id}) - for comparison:", file=sys.stderr, flush=True)
            print(f"  Position:    [{gt_camera_pos[0]:.4f}, {gt_camera_pos[1]:.4f}, {gt_camera_pos[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{gt_camera_quat[0]:.4f}, {gt_camera_quat[1]:.4f}, {gt_camera_quat[2]:.4f}, {gt_camera_quat[3]:.4f}]", file=sys.stderr, flush=True)
            
            # Store transformed pose for comparison (already done in gt_pose_callback, but update here too)
            self.gt_pose_camera_frame = gt_camera_pose
        else:
            print(f"[FOUNDATIONPOSE] Could not transform to camera frame", file=sys.stderr, flush=True)
        
        # 3. Transform to odom frame
        # For ground truth comparison, use strict_timestamp=False to allow fallback if exact timestamp unavailable
        gt_odom_pose = self._transform_pose_with_fallback('odom', gt_pose_msg, timeout=1.0, strict_timestamp=False)
        if gt_odom_pose is not None:
            gt_odom_pos, gt_odom_quat = self._extract_pose_arrays(gt_odom_pose)
            
            print(f"[FOUNDATIONPOSE] Odom frame (transformed via TF):", file=sys.stderr, flush=True)
            print(f"  Position:    [{gt_odom_pos[0]:.4f}, {gt_odom_pos[1]:.4f}, {gt_odom_pos[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{gt_odom_quat[0]:.4f}, {gt_odom_quat[1]:.4f}, {gt_odom_quat[2]:.4f}, {gt_odom_quat[3]:.4f}]", file=sys.stderr, flush=True)
            
            # 4. Grid coordinates (approximate, from odom position)
            grid_x = int(round(gt_odom_pos[0]))
            grid_y = int(round(gt_odom_pos[1]))
            print(f"[FOUNDATIONPOSE] Grid coordinates (approximate from odom):", file=sys.stderr, flush=True)
            print(f"  Grid (x, y): ({grid_x}, {grid_y})", file=sys.stderr, flush=True)
        else:
            print(f"[FOUNDATIONPOSE] Could not transform to odom frame", file=sys.stderr, flush=True)
        
        print("="*70 + "\n", file=sys.stderr, flush=True)
    
    def compare_with_ground_truth(self, estimated_pose, header):
        """
        Compare estimated pose with ground truth and print both.
        
        Args:
            estimated_pose: 4x4 transformation matrix (estimated)
            header: ROS message header
        """
        if self.gt_pose is None:
            # Log periodically that we're waiting for ground truth (not just once)
            if not hasattr(self, '_gt_wait_count'):
                self._gt_wait_count = 0
            self._gt_wait_count += 1
            if self._gt_wait_count % 100 == 1:  # Every ~4 seconds at 24Hz
                import sys
                gt_topic = f'/{self.object_name}/ground_truth_pose'
                print(f"[FOUNDATIONPOSE] Still waiting for ground truth pose (checked {self._gt_wait_count} times). "
                      f"Topic: {gt_topic}", file=sys.stderr, flush=True)
                rospy.logwarn_throttle(5.0, f"Waiting for ground truth pose from {gt_topic} topic...")
            return
        
        from scipy.spatial.transform import Rotation
        
        # Extract estimated pose
        # If coordinate frame correction was applied in odom frame but not in camera frame,
        # we need to apply it here for accurate error calculation
        estimated_pose_for_error = estimated_pose.copy()
        if (hasattr(self, '_correction_applied_cam') and not self._correction_applied_cam and 
            self.coord_correction_enabled and self.coord_correction_matrix is not None):
            # Correction wasn't applied in camera frame, but it will be in odom frame
            # Apply it here for consistent error calculation
            R_est_uncorrected = estimated_pose[:3, :3]
            R_est_corrected = self.coord_correction_matrix @ R_est_uncorrected
            estimated_pose_for_error[:3, :3] = R_est_corrected
            # Removed verbose debug print for error calculation correction
        
        R_est = estimated_pose_for_error[:3, :3]
        t_est = estimated_pose_for_error[:3, 3]
        rot_est = Rotation.from_matrix(R_est)
        quat_est = rot_est.as_quat()  # [x, y, z, w]
        
        # Extract ground truth pose - use camera frame if available, otherwise try to transform
        # Both poses MUST be in the same frame (camera frame) for accurate comparison
        gt_actual_frame = None
        if hasattr(self, 'gt_pose_camera_frame') and self.gt_pose_camera_frame is not None:
            # Use transformed pose in camera frame (preferred - already transformed)
            gt_pose_for_comparison = self.gt_pose_camera_frame
            gt_actual_frame = self.frame_id  # Camera frame
        else:
            # Try to transform on-the-fly
            gt_pose_for_comparison = self._transform_pose_with_fallback(self.frame_id, self.gt_pose, timeout=0.1)
            if gt_pose_for_comparison is None:
                # If transform fails, we cannot do accurate comparison
                # Log warning and skip comparison
                if not hasattr(self, '_gt_comparison_skip_logged'):
                    rospy.logwarn_throttle(5.0, f"Cannot compare poses: GT transform to camera frame failed. "
                                                f"GT is in '{self.gt_pose.header.frame_id}' but need '{self.frame_id}'")
                    self._gt_comparison_skip_logged = True
                return  # Skip comparison if frames don't match
            gt_actual_frame = self.frame_id  # Successfully transformed to camera frame
        
        # At this point, gt_actual_frame should always be self.frame_id (camera frame)
        # This check is just for safety
        if gt_actual_frame is None:
            rospy.logwarn("gt_actual_frame is None - this should not happen!")
            gt_actual_frame = self.frame_id
        
        t_gt, quat_gt = self._extract_pose_arrays(gt_pose_for_comparison)
        
        # Convert GT quaternion to rotation matrix
        rot_gt = Rotation.from_quat(quat_gt)
        R_gt = rot_gt.as_matrix()
        
        # Calculate errors
        pos_error = np.linalg.norm(t_est - t_gt)
        
        # Calculate orientation error using rotation matrix difference
        # This handles quaternion sign ambiguity correctly (q and -q represent same rotation)
        R_diff = R_est @ R_gt.T
        rot_diff = Rotation.from_matrix(R_diff)
        angle_error_rad = np.linalg.norm(rot_diff.as_rotvec())
        angle_error_deg = angle_error_rad * 180 / np.pi
        
        # Handle the case where the error is close to 180° - this might indicate
        # that the quaternions have opposite signs but represent the same orientation
        # Check if quaternions are negated versions of each other
        quat_est_normalized = quat_est / np.linalg.norm(quat_est)
        quat_gt_normalized = quat_gt / np.linalg.norm(quat_gt)
        
        # Check both q and -q (quaternion sign ambiguity)
        dot_product_1 = np.abs(np.dot(quat_est_normalized, quat_gt_normalized))
        dot_product_2 = np.abs(np.dot(quat_est_normalized, -quat_gt_normalized))
        max_dot = max(dot_product_1, dot_product_2)
        
        # Removed verbose debug print for quaternion dot products
        
        # If quaternions are nearly opposite (dot product close to 1), the actual error is small
        # The rotation matrix method should handle this, but if angle is ~180°, check quaternion alignment
        # Lower threshold to 0.9 to catch more cases
        if angle_error_deg > 90 and max_dot > 0.9:
            # Quaternions are nearly aligned (possibly with sign flip), recalculate error
            # Use the better-aligned quaternion
            if dot_product_2 > dot_product_1:
                quat_gt_aligned = -quat_gt_normalized
            else:
                quat_gt_aligned = quat_gt_normalized
            
            rot_est = Rotation.from_quat(quat_est_normalized)
            rot_gt_aligned = Rotation.from_quat(quat_gt_aligned)
            R_est_aligned = rot_est.as_matrix()
            R_gt_aligned = rot_gt_aligned.as_matrix()
            R_diff_aligned = R_est_aligned @ R_gt_aligned.T
            rot_diff_aligned = Rotation.from_matrix(R_diff_aligned)
            angle_error_rad_aligned = np.linalg.norm(rot_diff_aligned.as_rotvec())
            angle_error_deg_aligned = angle_error_rad_aligned * 180 / np.pi
            
            # Use the smaller error (should be the correct one)
            if angle_error_deg_aligned < angle_error_deg:
                angle_error_deg = angle_error_deg_aligned
                # Removed verbose debug print for quaternion sign ambiguity
        
        # Print comparison (use print for visibility, throttled to match performance metrics)
        current_time = rospy.get_time()
        if current_time - self._last_comparison_time >= self.COMPARISON_PRINT_INTERVAL:
            # Use sys.stderr for immediate visibility (not buffered)
            import sys
            # Get frame IDs for clarity - both should be in camera frame now
            estimated_frame = self.frame_id
            gt_original_frame = self.gt_pose.header.frame_id if self.gt_pose else "unknown"
            
            print("\n" + "="*60, file=sys.stderr, flush=True)
            print("POSE COMPARISON", file=sys.stderr, flush=True)
            print("="*60, file=sys.stderr, flush=True)
            
            # Camera frame comparison
            print(f"--- Camera Frame ({estimated_frame}) ---", file=sys.stderr, flush=True)
            print(f"ESTIMATED (FoundationPose):", file=sys.stderr, flush=True)
            print(f"  Position:    [{t_est[0]:.4f}, {t_est[1]:.4f}, {t_est[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{quat_est[0]:.4f}, {quat_est[1]:.4f}, {quat_est[2]:.4f}, {quat_est[3]:.4f}]", file=sys.stderr, flush=True)
            print(f"GROUND TRUTH (Isaac Sim) - Original frame: {gt_original_frame}, Transformed to: {gt_actual_frame}:", file=sys.stderr, flush=True)
            print(f"  Position:    [{t_gt[0]:.4f}, {t_gt[1]:.4f}, {t_gt[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{quat_gt[0]:.4f}, {quat_gt[1]:.4f}, {quat_gt[2]:.4f}, {quat_gt[3]:.4f}]", file=sys.stderr, flush=True)
            
            # World/map frame comparison
            # Create PoseStamped message from estimated pose (in camera frame) using helper
            est_pose_camera = self._pose_matrix_to_pose_stamped(estimated_pose, header, self.frame_id)
            
            # Transform estimated pose to map frame
            est_pose_map = self._transform_pose_with_fallback('map', est_pose_camera, timeout=0.1)
            
            # Ground truth is already in map frame, but extract it properly
            gt_map_pos = np.array([
                self.gt_pose.pose.position.x,
                self.gt_pose.pose.position.y,
                self.gt_pose.pose.position.z
            ])
            gt_map_quat = np.array([
                self.gt_pose.pose.orientation.x,
                self.gt_pose.pose.orientation.y,
                self.gt_pose.pose.orientation.z,
                self.gt_pose.pose.orientation.w
            ])
            
            print(f"\n--- World/Map Frame (map) ---", file=sys.stderr, flush=True)
            if est_pose_map is not None:
                est_map_pos, est_map_quat = self._extract_pose_arrays(est_pose_map)
                
                print(f"ESTIMATED (FoundationPose):", file=sys.stderr, flush=True)
                print(f"  Position:    [{est_map_pos[0]:.4f}, {est_map_pos[1]:.4f}, {est_map_pos[2]:.4f}] m", file=sys.stderr, flush=True)
                print(f"  Orientation: [{est_map_quat[0]:.4f}, {est_map_quat[1]:.4f}, {est_map_quat[2]:.4f}, {est_map_quat[3]:.4f}]", file=sys.stderr, flush=True)
            else:
                print(f"ESTIMATED (FoundationPose): [Transform to map frame failed]", file=sys.stderr, flush=True)
            
            print(f"GROUND TRUTH (Isaac Sim) - Original frame: {gt_original_frame}:", file=sys.stderr, flush=True)
            print(f"  Position:    [{gt_map_pos[0]:.4f}, {gt_map_pos[1]:.4f}, {gt_map_pos[2]:.4f}] m", file=sys.stderr, flush=True)
            print(f"  Orientation: [{gt_map_quat[0]:.4f}, {gt_map_quat[1]:.4f}, {gt_map_quat[2]:.4f}, {gt_map_quat[3]:.4f}]", file=sys.stderr, flush=True)
            
            # Grid coordinates
            print(f"\n--- Grid Coordinates ---", file=sys.stderr, flush=True)
            if est_pose_map is not None:
                # Try to get grid from odom first, then fall back to map
                est_pose_odom = self._transform_pose_with_fallback('odom', est_pose_camera, timeout=0.1)
                if est_pose_odom is not None:
                    est_odom_pos, _ = self._extract_pose_arrays(est_pose_odom)
                    est_grid_x = int(round(est_odom_pos[0]))
                    est_grid_y = int(round(est_odom_pos[1]))
                else:
                    est_grid_x = int(round(est_map_pos[0]))
                    est_grid_y = int(round(est_map_pos[1]))
                
                print(f"ESTIMATED (FoundationPose): Grid (x, y) = ({est_grid_x}, {est_grid_y})", file=sys.stderr, flush=True)
            else:
                print(f"ESTIMATED (FoundationPose): [Grid coordinates unavailable]", file=sys.stderr, flush=True)
            
            # Ground truth grid coordinates (from map frame directly)
            gt_grid_x = int(round(gt_map_pos[0]))
            gt_grid_y = int(round(gt_map_pos[1]))
            print(f"GROUND TRUTH (Isaac Sim): Grid (x, y) = ({gt_grid_x}, {gt_grid_y})", file=sys.stderr, flush=True)
            
            # Error metrics
            print(f"\n--- Error Metrics (both poses in {gt_actual_frame} frame) ---", file=sys.stderr, flush=True)
            print(f"  Position error:    {pos_error:.4f} m", file=sys.stderr, flush=True)
            print(f"  Orientation error: {angle_error_deg:.2f} degrees", file=sys.stderr, flush=True)
            
            # Debug: Check if quaternions are nearly opposite (quaternion sign ambiguity)
            quat_est_normalized = quat_est / np.linalg.norm(quat_est)
            quat_gt_normalized = quat_gt / np.linalg.norm(quat_gt)
            dot_product_1 = np.abs(np.dot(quat_est_normalized, quat_gt_normalized))
            dot_product_2 = np.abs(np.dot(quat_est_normalized, -quat_gt_normalized))
            max_dot = max(dot_product_1, dot_product_2)
            
            if angle_error_deg > 90 and max_dot > 0.99:
                print(f"  NOTE: Quaternion sign ambiguity detected (dot product: {max_dot:.4f}). "
                      f"Actual error may be smaller than reported.", file=sys.stderr, flush=True)
            
            print("="*60, file=sys.stderr, flush=True)
            
            # Also print ground truth in all coordinate systems periodically
            if not hasattr(self, '_last_gt_all_frames_time'):
                self._last_gt_all_frames_time = 0
            if current_time - self._last_gt_all_frames_time >= self.GT_ALL_FRAMES_PRINT_INTERVAL:
                self._print_gt_pose_in_all_frames(self.gt_pose)
                self._last_gt_all_frames_time = current_time
            
            print("", file=sys.stderr, flush=True)  # Empty line after comparison
            self._last_comparison_time = current_time
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def publish_markers(self, pose, header):
        """
        Publish visualization markers for the object.
        Transforms marker to odom frame for RViz visualization.
        
        Args:
            pose: 4x4 transformation matrix (object in camera frame)
            header: ROS message header
        """
        from scipy.spatial.transform import Rotation
        import numpy as np  # Import at function level to avoid issues
        
        marker_array = MarkerArray()
        
        # Create PoseStamped message from pose (in camera frame) using helper method
        pose_camera = self._pose_matrix_to_pose_stamped(pose, header, self.frame_id)
        
        # Transform pose to odom frame for RViz visualization (RViz uses odom as fixed frame)
        # Use strict_timestamp to get robot position at image capture time (prevents drift)
        # Log pose in camera frame before transform for debugging
        rospy.logdebug(f"Marker pose in camera frame ({self.frame_id}): pos=({pose[:3, 3][0]:.3f}, {pose[:3, 3][1]:.3f}, {pose[:3, 3][2]:.3f})")
        
        pose_odom = self._transform_pose_with_fallback('odom', pose_camera, timeout=0.5, strict_timestamp=True)
        if pose_odom is None:
            # If strict transform fails, DO NOT use latest transform (would use wrong robot position)
            # Skip marker publication to avoid incorrect visualization
            rospy.logerr_throttle(5.0, f"Could not transform marker to odom frame using strict timestamp {header.stamp.secs}.{header.stamp.nsecs}. "
                                      f"Skipping marker publication to avoid drift.")
            return  # Don't publish marker if transform fails
        
        # Verify the transform actually succeeded and frame_id is correct
        if pose_odom.header.frame_id != 'odom':
            rospy.logerr_throttle(5.0, f"Marker transform returned pose in wrong frame '{pose_odom.header.frame_id}' instead of 'odom'. Skipping marker publication.")
            return
        
        # Keep the original image timestamp for markers too
        # The pose position/orientation are already correctly transformed to odom frame using the exact timestamp
        # The timestamp should reflect when the measurement was made, not when it was published
        # The transform function should preserve the original timestamp, but ensure it's set correctly
        if pose_odom.header.stamp != header.stamp:
            # If transform changed the timestamp, restore the original image timestamp
            pose_odom.header.stamp = header.stamp
        # Ensure frame_id is set correctly (double-check)
        pose_odom.header.frame_id = 'odom'
        
        # Log pose in odom frame after transform for debugging
        rospy.logdebug(f"Marker pose in odom frame: pos=({pose_odom.pose.position.x:.3f}, {pose_odom.pose.position.y:.3f}, {pose_odom.pose.position.z:.3f}), "
                      f"frame_id={pose_odom.header.frame_id}, stamp={pose_odom.header.stamp.secs}.{pose_odom.header.stamp.nsecs}")
        
        # Apply coordinate frame correction AFTER transforming to odom frame
        # Apply correction in odom frame (where RViz visualizes)
        if self.coord_correction_enabled and self.coord_correction_matrix is not None:
            from scipy.spatial.transform import Rotation
            # Convert pose_odom to matrix
            pos_odom = np.array([
                pose_odom.pose.position.x,
                pose_odom.pose.position.y,
                pose_odom.pose.position.z
            ])
            quat_odom_before = np.array([
                pose_odom.pose.orientation.x,
                pose_odom.pose.orientation.y,
                pose_odom.pose.orientation.z,
                pose_odom.pose.orientation.w
            ])
            R_odom = Rotation.from_quat(quat_odom_before).as_matrix()
            euler_odom_before = Rotation.from_quat(quat_odom_before).as_euler('zyx', degrees=True)
            
            # Check if blue axis (Z-axis) is pointing down in odom frame
            # Z-axis in object frame is [0, 0, 1], transform to odom frame
            z_axis_obj = np.array([0, 0, 1])
            z_axis_odom = R_odom @ z_axis_obj
            
            # In odom frame (ROS convention: X forward, Y left, Z up), "down" is negative Z
            # Check if Z-axis is pointing down (negative Z component)
            z_axis_z_component = z_axis_odom[2]
            
            # Only apply correction if Z-axis is pointing down
            should_apply_correction = z_axis_z_component < -0.5
            
            # Removed verbose debug prints for BEFORE correction
            
            # Apply correction only if Z-axis is pointing down
            if should_apply_correction:
                # Apply correction: only rotate orientation, keep position unchanged
                R_corrected = self.coord_correction_matrix @ R_odom
                quat_corrected = Rotation.from_matrix(R_corrected).as_quat()
                euler_odom_after = Rotation.from_quat(quat_corrected).as_euler('zyx', degrees=True)
            else:
                # No correction applied, use original values
                quat_corrected = quat_odom_before
                euler_odom_after = euler_odom_before
                R_corrected = R_odom
            
            # Removed verbose debug prints for AFTER correction
            
            # Update pose_odom with corrected orientation (or original if no correction applied)
            pose_odom.pose.orientation.x = quat_corrected[0]
            pose_odom.pose.orientation.y = quat_corrected[1]
            pose_odom.pose.orientation.z = quat_corrected[2]
            pose_odom.pose.orientation.w = quat_corrected[3]
            # Position remains unchanged
        
        # Debug: Log orientation comparison between camera frame (track_vis) and odom frame (RViz)
        if self.debug >= 2 and not hasattr(self, '_rviz_orientation_debug_logged'):
            from scipy.spatial.transform import Rotation
            
            # Camera frame orientation (what's shown in track_vis)
            R_cam = pose[:3, :3]
            rot_cam = Rotation.from_matrix(R_cam)
            quat_cam = rot_cam.as_quat()
            euler_cam = rot_cam.as_euler('zyx', degrees=True)
            
            # Odom frame orientation (what's shown in RViz)
            quat_odom = pose_odom.pose.orientation
            rot_odom = Rotation.from_quat([quat_odom.x, quat_odom.y, quat_odom.z, quat_odom.w])
            euler_odom = rot_odom.as_euler('zyx', degrees=True)
            
            # Get the camera-to-odom transform to understand the rotation difference
            try:
                latest_time = self.tf_listener.getLatestCommonTime(self.frame_id, 'odom')
                self.tf_listener.waitForTransform('odom', self.frame_id, latest_time, rospy.Duration(0.1))
                (trans_cam_to_odom, rot_cam_to_odom) = self.tf_listener.lookupTransform('odom', self.frame_id, latest_time)
                rot_cam_to_odom_obj = Rotation.from_quat([rot_cam_to_odom[0], rot_cam_to_odom[1], rot_cam_to_odom[2], rot_cam_to_odom[3]])
                euler_cam_to_odom = rot_cam_to_odom_obj.as_euler('zyx', degrees=True)
                R_cam_to_odom = rot_cam_to_odom_obj.as_matrix()
            except Exception as e:
                rospy.logwarn(f"Could not get camera-to-odom transform: {e}")
                R_cam_to_odom = None
                euler_cam_to_odom = None
            
            rospy.loginfo("="*60)
            rospy.loginfo("ORIENTATION COMPARISON: track_vis (camera) vs RViz (odom)")
            rospy.loginfo("="*60)
            rospy.loginfo("CAMERA FRAME (track_vis):")
            rospy.loginfo(f"  Quaternion: [{quat_cam[0]:.4f}, {quat_cam[1]:.4f}, {quat_cam[2]:.4f}, {quat_cam[3]:.4f}]")
            rospy.loginfo(f"  Euler ZYX (degrees): [{euler_cam[0]:.2f}, {euler_cam[1]:.2f}, {euler_cam[2]:.2f}]")
            rospy.loginfo("ODOM FRAME (RViz):")
            rospy.loginfo(f"  Quaternion: [{quat_odom.x:.4f}, {quat_odom.y:.4f}, {quat_odom.z:.4f}, {quat_odom.w:.4f}]")
            rospy.loginfo(f"  Euler ZYX (degrees): [{euler_odom[0]:.2f}, {euler_odom[1]:.2f}, {euler_odom[2]:.2f}]")
            if euler_cam_to_odom is not None:
                rospy.loginfo("CAMERA-TO-ODOM TRANSFORM:")
                rospy.loginfo(f"  Translation: [{trans_cam_to_odom[0]:.4f}, {trans_cam_to_odom[1]:.4f}, {trans_cam_to_odom[2]:.4f}]")
                rospy.loginfo(f"  Rotation Euler ZYX (degrees): [{euler_cam_to_odom[0]:.2f}, {euler_cam_to_odom[1]:.2f}, {euler_cam_to_odom[2]:.2f}]")
                rospy.loginfo(f"  Z-axis rotation difference: {euler_odom[0] - euler_cam[0]:.2f} degrees")
                rospy.loginfo(f"  Y-axis rotation difference: {euler_odom[1] - euler_cam[1]:.2f} degrees")
                rospy.loginfo(f"  X-axis rotation difference: {euler_odom[2] - euler_cam[2]:.2f} degrees")
            rospy.loginfo("="*60)
            rospy.loginfo("NOTE: The orientation difference is due to coordinate frame conventions:")
            rospy.loginfo("  - Camera frame: OpenCV convention (X right, Y down, Z forward)")
            rospy.loginfo("  - Odom frame: ROS convention (X forward, Y left, Z up)")
            rospy.loginfo("  - The camera-to-odom transform includes the camera's orientation on the robot")
            rospy.loginfo("="*60)
            self._rviz_orientation_debug_logged = True
        
        # Create mesh marker in odom frame
        # The pose position is at the object's center, but the mesh file has its origin at the bottom.
        # RViz places the mesh's origin at the marker position, so we need to offset the marker
        # position down by the offset from center to origin (in the object frame, then transform to odom frame).
        import numpy as np
        from scipy.spatial.transform import Rotation
        
        # Get object rotation in odom frame (will be reused for axes markers)
        obj_quat = pose_odom.pose.orientation
        obj_rot = Rotation.from_quat([obj_quat.x, obj_quat.y, obj_quat.z, obj_quat.w])
        obj_R = obj_rot.as_matrix()
        
        # Calculate offset from mesh origin to center in original mesh frame
        # inv(to_origin)[:3, 3] is the translation from origin to center
        inv_to_origin = np.linalg.inv(self.to_origin)
        mesh_origin_to_center = inv_to_origin[:3, 3]  # In original mesh frame
        
        # Transform offset to odom frame
        mesh_origin_to_center_odom = obj_R @ mesh_origin_to_center
        
        # Offset marker position: move DOWN by the offset (since mesh origin is below center)
        # Place mesh origin at marker position so center aligns with pose
        marker_position = np.array([
            pose_odom.pose.position.x,
            pose_odom.pose.position.y,
            pose_odom.pose.position.z
        ]) - mesh_origin_to_center_odom
        
        marker = Marker()
        marker.header = pose_odom.header
        marker.header.frame_id = pose_odom.header.frame_id  # Use odom frame
        marker.ns = "foundationpose"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = marker_position[0]
        marker.pose.position.y = marker_position[1]
        marker.pose.position.z = marker_position[2]
        marker.pose.orientation.x = pose_odom.pose.orientation.x
        marker.pose.orientation.y = pose_odom.pose.orientation.y
        marker.pose.orientation.z = pose_odom.pose.orientation.z
        marker.pose.orientation.w = pose_odom.pose.orientation.w
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.a = 0.5
        
        # Alternate between red and blue each time a new pose is published for visual debugging
        self.marker_color_counter += 1
        if self.marker_color_counter % 2 == 0:
            # Even: Red
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            # Odd: Blue
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        marker.mesh_resource = f"file://{self.mesh_file}"
        marker.mesh_use_embedded_materials = True
        
        marker_array.markers.append(marker)
        
        # Add coordinate frame markers (axes) - positioned at object location in odom frame
        axis_length = self.AXIS_LENGTH
        axis_colors = self.AXIS_COLORS
        axis_names = self.AXIS_NAMES
        
        # Get object position in odom frame (obj_R already computed above for mesh marker)
        obj_pos = np.array([
            pose_odom.pose.position.x,
            pose_odom.pose.position.y,
            pose_odom.pose.position.z
        ])
        
        # Axis directions in object frame (x, y, z axes)
        axis_dirs_obj = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1])
        }
        
        for i, (color, axis) in enumerate(zip(axis_colors, axis_names)):
            axis_marker = Marker()
            axis_marker.header = pose_odom.header
            axis_marker.header.frame_id = pose_odom.header.frame_id  # Use odom frame
            axis_marker.ns = "foundationpose_axes"
            axis_marker.id = i + 1
            axis_marker.type = Marker.ARROW
            axis_marker.action = Marker.ADD
            
            # Set a valid identity quaternion to avoid "Uninitialized quaternion" warning
            # (Even though we use points, RViz expects a valid quaternion)
            axis_marker.pose.orientation.w = 1.0
            axis_marker.pose.orientation.x = 0.0
            axis_marker.pose.orientation.y = 0.0
            axis_marker.pose.orientation.z = 0.0
            
            # Transform axis direction from object frame to odom frame
            axis_dir_obj = axis_dirs_obj[axis]
            axis_dir_odom = obj_R @ axis_dir_obj
            
            # Normalize direction vector
            axis_dir_odom = axis_dir_odom / np.linalg.norm(axis_dir_odom)
            
            # Calculate start and end points for the arrow
            start_point = obj_pos
            end_point = obj_pos + axis_dir_odom * axis_length
            
            # Set arrow points (start and end)
            axis_marker.points.append(Point(x=start_point[0], y=start_point[1], z=start_point[2]))
            axis_marker.points.append(Point(x=end_point[0], y=end_point[1], z=end_point[2]))
            
            # Set arrow scale (diameter, not length - length is determined by points)
            axis_marker.scale.x = self.AXIS_SHAFT_DIAMETER
            axis_marker.scale.y = self.AXIS_HEAD_DIAMETER
            axis_marker.scale.z = 0.0   # Not used for ARROW with points
            
            # Set color
            axis_marker.color.a = 1.0
            axis_marker.color.r = color[0]
            axis_marker.color.g = color[1]
            axis_marker.color.b = color[2]
            
            marker_array.markers.append(axis_marker)
        
        marker_pos = (pose_odom.pose.position.x, pose_odom.pose.position.y, pose_odom.pose.position.z)
        rospy.loginfo(f"[FOUNDATIONPOSE] PUBLISHED MARKERS TO TOPIC - Odom frame: ({marker_pos[0]:.3f}, {marker_pos[1]:.3f}, {marker_pos[2]:.3f})m, "
                      f"timestamp: {pose_odom.header.stamp.secs}.{pose_odom.header.stamp.nsecs}")
        self.marker_pub.publish(marker_array)
    
    # ========================================================================
    # Main Loop
    # ========================================================================
    
    def run(self):
        """Run the node (blocking)."""
        rospy.spin()

# MAIN EXECUTION

def main():
    """Main function."""
    node = None
    try:
        node = FoundationPoseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down FoundationPose node")
    finally:
        # Save final metrics on shutdown
        if node is not None:
            try:
                import json
                if hasattr(node, 'metrics_data') and node.metrics_data:
                    with open(node.metrics_file, 'w') as f:
                        json.dump(node.metrics_data, f, indent=2)
                    rospy.loginfo(f"Saved {len(node.metrics_data)} metrics entries to {node.metrics_file}")
            except Exception as e:
                rospy.logwarn(f"Failed to save final metrics: {e}")

if __name__ == '__main__':
    main()
