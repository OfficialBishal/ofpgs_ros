#!/usr/bin/env python3
"""
SAM segmentation ROS node.
"""

import os
import sys
import time
import numpy as np
import cv2
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import rospy
from sensor_msgs.msg import Image

# SAM
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
except ImportError as e:
    print(f"ERROR: Failed to import SAM modules: {e}")
    print("Make sure you're running in the sam conda environment")
    print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

# YOLO (optional, for object detection strategy)
try:
    os.environ['YOLO_VERBOSE'] = 'False'  # Suppress YOLO verbose output
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    # Note: rospy not imported yet, so we can't log here
    # Will log warning during initialization if detection strategy is used

# Main Node Class

class SAMSegmentationNode:
    """
    SAM Segmentation ROS Node
    
    Object segmentation using SAM and YOLO.
    and publishes masks for FoundationPose.
    """
    
    def __init__(self):
        """Initialize the SAM segmentation node."""
        rospy.init_node('sam_segmentation', anonymous=True)
        
        # Load parameters
        self._load_parameters()
        
        # Initialize object detection (if needed)
        if self.segmentation_strategy == 'detection':
            self._initialize_detection()
        
        # Initialize SAM model
        self._initialize_sam()
        
        # Setup ROS communication
        self._setup_ros_communication()
        
        # Initialize pynvml if available (for GPU utilization monitoring)
        self.pynvml_initialized = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml_initialized = True
        except (ImportError, Exception):
            pass
        
        # Performance metrics throttling (print every N frames)
        self.performance_print_interval = 10  # Print every 10 frames
        self.frame_counter = 0
        
        # Metrics saving
        import os
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
        
        self.metrics_dir = os.path.join(workspace_root, 'src', 'ofpgs_ros', 'metrics', 'data')
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.metrics_dir, 'yolo_sam_metrics.json')
        self.metrics_data = []
        rospy.loginfo(f"Metrics will be saved to: {self.metrics_file}")
        
        rospy.loginfo("SAM Segmentation node initialized")
        rospy.loginfo(f"Subscribing to RGB: {self.rgb_topic}")
        rospy.loginfo(f"Publishing mask to: {self.mask_topic}")
        rospy.loginfo(f"Using model: {self.model_type}")
        rospy.loginfo(f"Segmentation strategy: {self.segmentation_strategy}")
        if not PSUTIL_AVAILABLE:
            rospy.logwarn("psutil not available, CPU monitoring will be disabled")
    
    
    def _load_parameters(self):
        """Load ROS parameters from config file (organized structure)."""
        # Object name parameter (used for default mask topic and class mapping)
        self.object_name = rospy.get_param('~object_name', 'cracker_box')
        rospy.loginfo(f"Object name: {self.object_name}")
        
        # Topic parameters (from config file nested structure, with fallbacks)
        self.rgb_topic = rospy.get_param('~camera/rgb_topic', 
                                        rospy.get_param('~rgb_topic', '/hsrb/head_rgbd_sensor/rgb/image_rect_color'))
        self.mask_topic = rospy.get_param('~mask/mask_topic',
                                         rospy.get_param('~mask_topic', ''))
        # If mask_topic is empty, use default based on object_name
        if not self.mask_topic:
            self.mask_topic = f'/segmentation/{self.object_name}_mask'
            rospy.loginfo(f"No mask_topic specified, using default: {self.mask_topic}")
        self.frame_id = rospy.get_param('~camera/frame_id',
                                       rospy.get_param('~frame_id', 'head_rgbd_sensor_rgb_frame'))
        
        # SAM model parameters (from config file nested structure)
        self.model_type = rospy.get_param('~sam/sam_model_type',
                                        rospy.get_param('~sam_model_type', 'vit_b'))
        checkpoint_default = os.path.join(
            os.path.expanduser('~'),
            'hsr_robocanes_omniverse',
            'segment-anything',
            'checkpoints',
            f'sam_{self.model_type}.pth'
        )
        self.checkpoint_path = rospy.get_param('~sam/sam_checkpoint_path',
                                              rospy.get_param('~sam_checkpoint_path', checkpoint_default))
        
        # Resolve path relative to workspace root if needed
        if self.checkpoint_path:
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
            if self.checkpoint_path.startswith('src/'):
                self.checkpoint_path = os.path.join(workspace_root, self.checkpoint_path)
            # Expand user home directory if path starts with ~
            elif self.checkpoint_path.startswith('~'):
                self.checkpoint_path = os.path.expanduser(self.checkpoint_path)
            # If relative path, try resolving relative to workspace root
            elif not os.path.isabs(self.checkpoint_path):
                potential_path = os.path.join(workspace_root, self.checkpoint_path)
                if os.path.exists(potential_path):
                    self.checkpoint_path = potential_path
        
        # Segmentation strategy (from config file nested structure)
        self.segmentation_strategy = rospy.get_param('~sam/segmentation_strategy',
                                                    rospy.get_param('~segmentation_strategy', 'center_point'))
        
        # Point prompt parameters (from config file nested structure)
        self.prompt_point_x = rospy.get_param('~sam/prompt_point_x',
                                             rospy.get_param('~prompt_point_x', -1))
        self.prompt_point_y = rospy.get_param('~sam/prompt_point_y',
                                             rospy.get_param('~prompt_point_y', -1))
        
        # Box prompt parameters (from config file nested structure)
        self.box_x_min = rospy.get_param('~sam/box_x_min',
                                        rospy.get_param('~box_x_min', 0.3))
        self.box_y_min = rospy.get_param('~sam/box_y_min',
                                        rospy.get_param('~box_y_min', 0.3))
        self.box_x_max = rospy.get_param('~sam/box_x_max',
                                        rospy.get_param('~box_x_max', 0.7))
        self.box_y_max = rospy.get_param('~sam/box_y_max',
                                        rospy.get_param('~box_y_max', 0.7))
        
        # Automatic mask generation parameters (from config file nested structure)
        self.min_mask_area = rospy.get_param('~sam/min_mask_area',
                                            rospy.get_param('~min_mask_area', 100))
        self.max_mask_area = rospy.get_param('~sam/max_mask_area',
                                            rospy.get_param('~max_mask_area', 1000000))
        
        # Object detection parameters (from config file nested structure)
        self.yolo_model_path = rospy.get_param('~object_detection/yolo_model_path',
                                               rospy.get_param('~yolo_model_path', 'yolov8n.pt'))
        self.target_class_name = rospy.get_param('~object_detection/target_class_name',
                                                 rospy.get_param('~target_class_name', 'bottle'))
        self.target_class_id = rospy.get_param('~object_detection/target_class_id',
                                               rospy.get_param('~target_class_id', -1))
        self.detection_confidence = rospy.get_param('~object_detection/detection_confidence',
                                                    rospy.get_param('~detection_confidence', 0.25))
        self.detection_iou = rospy.get_param('~object_detection/detection_iou',
                                             rospy.get_param('~detection_iou', 0.45))
        
        # Device (from config file nested structure)
        self.device = rospy.get_param('~sam/device',
                                     rospy.get_param('~device', 'cuda'))
        
        # Verify checkpoint exists
        if not os.path.exists(self.checkpoint_path):
            rospy.logerr(f"SAM checkpoint not found: {self.checkpoint_path}")
            rospy.logerr("Please download checkpoint or set ~sam_checkpoint_path parameter")
            rospy.logerr("Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            sys.exit(1)
    
    def _initialize_detection(self):
        """Initialize object detection model (YOLO)."""
        if not YOLO_AVAILABLE:
            rospy.logerr("YOLO is not available but 'detection' strategy is selected!")
            rospy.logerr("Install YOLO with: pip install ultralytics")
            rospy.logerr("Note: YOLO should be installed in the 'sam' conda environment")
            sys.exit(1)
        
        rospy.loginfo(f"Loading YOLO model: {self.yolo_model_path}")
        try:
            self.detection_model = YOLO(self.yolo_model_path)
            rospy.loginfo(f"YOLO model loaded successfully")
            rospy.loginfo(f"Target class: {self.target_class_name} (ID: {self.target_class_id})")
        except Exception as e:
            rospy.logerr(f"Failed to initialize YOLO model: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _initialize_sam(self):
        """Initialize SAM model."""
        rospy.loginfo(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")
        try:
            # Load SAM model
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            
            # Initialize predictor based on strategy
            if self.segmentation_strategy == 'automatic':
                # Automatic mask generator
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=self.min_mask_area,
                )
                self.predictor = None
            else:
                # Point/box predictor
                self.predictor = SamPredictor(self.sam)
                self.mask_generator = None
            
            rospy.loginfo("SAM model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize SAM: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _setup_ros_communication(self):
        """Setup ROS publishers and subscribers."""
        # Publisher for mask
        self.mask_pub = rospy.Publisher(self.mask_topic, Image, queue_size=10)
        
        # Subscriber for RGB image
        self.image_sub = rospy.Subscriber(
            self.rgb_topic,
            Image,
            self.image_callback,
            queue_size=1
        )
        
        rospy.loginfo(f"Subscribed to: {self.rgb_topic}")
        rospy.loginfo(f"Publishing to: {self.mask_topic}")
    
    # Image Processing
    
    def ros_image_to_numpy(self, img_msg, desired_encoding='rgb8'):
        """
        Convert ROS Image message to numpy array.
        
        Args:
            img_msg: sensor_msgs.msg.Image
            desired_encoding: Desired output encoding (e.g., 'rgb8', 'bgr8')
        
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
        else:
            rospy.logwarn(f"Unsupported encoding: {img_msg.encoding}, attempting default conversion")
            if len(img_msg.data) == height * width:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width)
            elif len(img_msg.data) == height * width * 3:
                img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width, 3)
            else:
                raise ValueError(f"Cannot convert encoding {img_msg.encoding} to {desired_encoding}")
        
        return img_array
    
    def numpy_to_ros_image(self, img_array, encoding='mono8', frame_id=None, header=None):
        """
        Convert numpy array to ROS Image message.
        
        Args:
            img_array: numpy.ndarray image
            encoding: ROS image encoding (e.g., 'mono8', 'rgb8')
            frame_id: Frame ID for header (used if header is None)
            header: ROS message header to copy timestamp and frame_id from
        
        Returns:
            sensor_msgs.msg.Image: ROS Image message
        """
        img_msg = Image()
        if header is not None:
            # Use original header for timestamp sync
            img_msg.header.stamp = header.stamp
            img_msg.header.frame_id = header.frame_id
        else:
            # Fallback: use current time (should only happen if header not provided)
            img_msg.header.stamp = rospy.Time.now()
            if frame_id:
                img_msg.header.frame_id = frame_id
        
        if len(img_array.shape) == 2:
            # Grayscale
            height, width = img_array.shape
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = encoding
            img_msg.is_bigendian = 0
            img_msg.step = width
            img_msg.data = img_array.tobytes()
        elif len(img_array.shape) == 3:
            # Color
            height, width, channels = img_array.shape
            img_msg.height = height
            img_msg.width = width
            img_msg.encoding = encoding
            img_msg.is_bigendian = 0
            img_msg.step = width * channels
            img_msg.data = img_array.tobytes()
        else:
            raise ValueError(f"Unsupported image shape: {img_array.shape}")
        
        return img_msg
    
    # Performance Monitoring
    
    def get_gpu_usage(self):
        """Get GPU memory and utilization usage."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                memory_allocated_pct = (memory_allocated / memory_total) * 100
                memory_reserved_pct = (memory_reserved / memory_total) * 100
                
                gpu_util = None
                if self.pynvml_initialized:
                    try:
                        import pynvml
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                    except Exception:
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
                process = psutil.Process(os.getpid())
                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_system = psutil.cpu_percent(interval=0.1)
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
            
            self.metrics_data.append(metric_entry)
            
            # Save to file periodically (every 10 entries to reduce I/O)
            if len(self.metrics_data) % 10 == 0:
                with open(self.metrics_file, 'w') as f:
                    json.dump(self.metrics_data, f, indent=2)
        except Exception as e:
            rospy.logwarn(f"Failed to save metrics: {e}")
    
    def print_performance_metrics(self, elapsed_time, gpu_info, cpu_info, method_name="YOLO+SAM"):
        """Print performance metrics in a formatted way."""
        rospy.loginfo("=" * 80)
        rospy.loginfo(f"PERFORMANCE METRICS - {method_name.upper()} SEGMENTATION")
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
    
    # Segmentation Methods
    
    def segment_with_point(self, image, point_x, point_y):
        """
        Segment using point prompt.
        
        Args:
            image: RGB image as numpy array
            point_x: X coordinate of point (in image coordinates)
            point_y: Y coordinate of point (in image coordinates)
        
        Returns:
            numpy.ndarray: Binary mask
        """
        self.predictor.set_image(image)
        
        # Point prompt: (x, y) coordinates
        input_point = np.array([[point_x, point_y]])
        input_label = np.array([1])  # 1 = foreground point
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        
        return masks[0]  # Return best mask
    
    def segment_with_box(self, image, box):
        """
        Segment using box prompt.
        
        Args:
            image: RGB image as numpy array
            box: Bounding box [x_min, y_min, x_max, y_max]
        
        Returns:
            numpy.ndarray: Binary mask
        """
        self.predictor.set_image(image)
        
        input_box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        return masks[0]  # Return best mask
    
    def segment_automatic(self, image):
        """
        Segment using automatic mask generation, then filter by size.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            numpy.ndarray: Binary mask (largest valid mask)
        """
        masks = self.mask_generator.generate(image)
        
        if not masks:
            rospy.logwarn("No masks generated")
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        # Filter masks by area
        valid_masks = [
            m for m in masks
            if self.min_mask_area <= m['area'] <= self.max_mask_area
        ]
        
        if not valid_masks:
            rospy.logwarn(f"No masks in size range [{self.min_mask_area}, {self.max_mask_area}]")
            # Return largest mask regardless of size
            valid_masks = masks
        
        # Return largest mask
        largest_mask = max(valid_masks, key=lambda m: m['area'])
        return largest_mask['segmentation']
    
    def detect_object(self, image):
        """
        Detect target object using YOLO and return bounding box.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            tuple: (x_min, y_min, x_max, y_max) bounding box, or None if not found
        """
        if not YOLO_AVAILABLE or not hasattr(self, 'detection_model'):
            rospy.logerr("Detection model not initialized")
            return None
        
        # Convert RGB to BGR for YOLO (YOLO expects BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Run YOLO detection
        results = self.detection_model.predict(
            image_bgr,
            conf=self.detection_confidence,
            iou=self.detection_iou,
            verbose=False
        )
        
        if not results or len(results) == 0:
            rospy.logwarn_throttle(5.0, "No detections from YOLO")
            return None
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            rospy.logwarn_throttle(5.0, "No bounding boxes detected")
            return None
        
        # Debug: Log all detected classes
        detected_classes = {}
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.detection_model.names[class_id]
            confidence = float(box.conf[0].item())
            if class_name not in detected_classes or confidence > detected_classes[class_name]:
                detected_classes[class_name] = confidence
        
        if detected_classes:
            rospy.loginfo_throttle(5.0, f"YOLO detected classes: {detected_classes}")
        
        # Map object names to YOLO class names (COCO dataset classes)
        # COCO doesn't have "box" or "cracker_box", so we use alternative classes
        # COCO classes that might match a box-like object:
        # - 'bottle' (rectangular container)
        # - 'book' (rectangular object)
        # - 'suitcase' (box-like container)
        # - 'remote' (small rectangular object)
        # - 'mouse' (small rectangular object)
        class_name_map = {
            'cracker_box': ['bottle', 'book', 'suitcase', 'remote', 'mouse'],  # Try multiple box-like classes
            'box': ['bottle', 'book', 'suitcase', 'remote', 'mouse'],
            'mustard_bottle': ['bottle'],
            'bottle': ['bottle'],
        }
        
        # Get list of acceptable class names for this object
        acceptable_classes = class_name_map.get(self.object_name, [self.target_class_name])
        if self.target_class_name not in acceptable_classes:
            acceptable_classes.append(self.target_class_name)
        
        rospy.loginfo_throttle(10.0, f"Looking for object '{self.object_name}' using classes: {acceptable_classes}")
        
        # Find target object by class name or ID
        target_box = None
        best_confidence = 0.0
        
        for box in result.boxes:
            # Get class ID and name
            class_id = int(box.cls[0].item())
            class_name = self.detection_model.names[class_id]
            confidence = float(box.conf[0].item())
            
            # Check if this is the target object
            is_target = False
            if self.target_class_id >= 0:
                is_target = (class_id == self.target_class_id)
            else:
                # Match by class name - check against acceptable classes
                for acceptable_class in acceptable_classes:
                    if acceptable_class.lower() in class_name.lower() or class_name.lower() in acceptable_class.lower():
                        is_target = True
                        break
            
            if is_target and confidence > best_confidence:
                # Get bounding box coordinates
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                target_box = (int(x_min), int(y_min), int(x_max), int(y_max))
                best_confidence = confidence
        
        if target_box is None:
            rospy.logwarn_throttle(5.0, f"Target object '{self.object_name}' not detected. Detected classes: {list(detected_classes.keys())}")
            return None
        
        rospy.loginfo_throttle(5.0, 
            f"Detected {self.target_class_name} with confidence {best_confidence:.2f} at {target_box}")
        
        return target_box
    
    def segment_with_detection(self, image):
        """
        Segment using object detection to find bounding box, then use box prompt for SAM.
        
        Args:
            image: RGB image as numpy array
        
        Returns:
            numpy.ndarray: Binary mask
        """
        # Detect object and get bounding box
        bbox = self.detect_object(image)
        
        if bbox is None:
            rospy.logwarn("No object detected, returning empty mask")
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        # Use bounding box for SAM segmentation
        x_min, y_min, x_max, y_max = bbox
        box = [x_min, y_min, x_max, y_max]
        return self.segment_with_box(image, box)
    
    def image_callback(self, rgb_msg):
        """Callback for RGB image."""
        # Store original header - mask needs same timestamp as RGB
        original_header = rgb_msg.header
        
        # Get initial resource usage
        gpu_info_before = self.get_gpu_usage()
        cpu_info_before = self.get_cpu_usage()
        
        # Start timing
        start_time = time.time()
        
        try:
            # Convert ROS image to numpy
            rgb_image = self.ros_image_to_numpy(rgb_msg, desired_encoding='rgb8')
            height, width = rgb_image.shape[:2]
            
            # Perform segmentation based on strategy
            if self.segmentation_strategy == 'automatic':
                mask = self.segment_automatic(rgb_image)
            elif self.segmentation_strategy == 'center_point':
                # Use center of image as prompt
                point_x = width // 2 if self.prompt_point_x < 0 else int(self.prompt_point_x * width)
                point_y = height // 2 if self.prompt_point_y < 0 else int(self.prompt_point_y * height)
                mask = self.segment_with_point(rgb_image, point_x, point_y)
            elif self.segmentation_strategy == 'point':
                # Use specified point
                point_x = int(self.prompt_point_x * width) if self.prompt_point_x <= 1.0 else int(self.prompt_point_x)
                point_y = int(self.prompt_point_y * height) if self.prompt_point_y <= 1.0 else int(self.prompt_point_y)
                mask = self.segment_with_point(rgb_image, point_x, point_y)
            elif self.segmentation_strategy == 'box':
                # Use bounding box
                box = [
                    int(self.box_x_min * width),
                    int(self.box_y_min * height),
                    int(self.box_x_max * width),
                    int(self.box_y_max * height)
                ]
                mask = self.segment_with_box(rgb_image, box)
            elif self.segmentation_strategy == 'detection':
                # Use object detection to find bounding box, then segment
                mask = self.segment_with_detection(rgb_image)
            else:
                rospy.logerr(f"Unknown segmentation strategy: {self.segmentation_strategy}")
                return
            
            # Convert mask to uint8 (0 or 255)
            mask_uint8 = (mask.astype(np.uint8) * 255)
            
            # End timing
            elapsed_time = time.time() - start_time
            
            # Get resource usage after segmentation
            gpu_info_after = self.get_gpu_usage()
            cpu_info_after = self.get_cpu_usage()
            
            # Use after values for reporting (peak usage during computation)
            gpu_info = gpu_info_after if gpu_info_after else gpu_info_before
            cpu_info = cpu_info_after if cpu_info_after else cpu_info_before
            
            # Print performance metrics only at intervals (to reduce terminal clutter)
            self.frame_counter += 1
            method_name = f"YOLO+SAM ({self.segmentation_strategy})" if self.segmentation_strategy == 'detection' else f"SAM ({self.segmentation_strategy})"
            if self.frame_counter % self.performance_print_interval == 0:
                self.print_performance_metrics(elapsed_time, gpu_info, cpu_info, method_name)
            else:
                # Just log time for non-printed frames
                rospy.logdebug(f"{method_name}: {elapsed_time*1000:.1f} ms")
            
            # Save metrics to file
            self._save_metrics(elapsed_time, gpu_info, cpu_info)
            
            # Publish mask with original RGB timestamp
            mask_msg = self.numpy_to_ros_image(
                mask_uint8,
                encoding='mono8',
                frame_id=rgb_msg.header.frame_id,
                header=original_header
            )
            self.mask_pub.publish(mask_msg)
            
            # Log mask statistics (throttled)
            mask_area = mask.sum()
            rospy.loginfo_throttle(5.0, f"SAM segmentation: mask area = {mask_area} pixels ({mask_area/(width*height)*100:.1f}% of image)")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            rospy.logerr(f"Error in SAM segmentation (took {elapsed_time*1000:.2f} ms): {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    # Main Loop
    
    def run(self):
        """Run the node (blocking)."""
        rospy.spin()

# Main Execution

def main():
    """Main function."""
    node = None
    try:
        node = SAMSegmentationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down SAM Segmentation node")
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

