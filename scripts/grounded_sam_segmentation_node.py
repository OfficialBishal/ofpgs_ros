#!/usr/bin/env python3
"""
Grounded SAM segmentation ROS node.
"""

import os
import sys
import time
import numpy as np
import cv2
import torch
import torchvision
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import rospy
from sensor_msgs.msg import Image

# Add Grounded SAM paths
GROUNDED_SAM_PATH = os.path.expanduser("~/hsr_robocanes_omniverse/Grounded-Segment-Anything")
if os.path.exists(GROUNDED_SAM_PATH):
    sys.path.insert(0, GROUNDED_SAM_PATH)
    sys.path.insert(0, os.path.join(GROUNDED_SAM_PATH, "GroundingDINO"))

# Grounding DINO
try:
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.models import build_model
    from GroundingDINO.groundingdino.util.slconfig import SLConfig
    from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
except ImportError as e:
    print(f"ERROR: Failed to import Grounding DINO modules: {e}")
    print("Make sure you're running in the grounded_sam conda environment")
    sys.exit(1)

# SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    print(f"ERROR: Failed to import SAM modules: {e}")
    print("Make sure you're running in the grounded_sam conda environment")
    sys.exit(1)

# Main Node Class

class GroundedSAMSegmentationNode:
    """
    Grounded SAM Segmentation ROS Node
    
    Uses Grounding DINO + SAM for open-vocabulary object detection and segmentation.
    """
    
    def __init__(self):
        """Initialize the Grounded SAM segmentation node."""
        rospy.init_node('grounded_sam_segmentation', anonymous=True)
        
        # Load parameters
        self._load_parameters()
        
        # Initialize models
        self._initialize_grounding_dino()
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
        workspace_root = os.path.expanduser('~/hsr_robocanes_omniverse')
        self.metrics_dir = os.path.join(workspace_root, 'src', 'ofpgs_ros', 'metrics', 'data')
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.metrics_dir, 'grounded_sam_metrics.json')
        self.metrics_data = []
        rospy.loginfo(f"Metrics will be saved to: {self.metrics_file}")
        
        rospy.loginfo("Grounded SAM Segmentation node initialized")
        rospy.loginfo(f"Subscribing to RGB: {self.rgb_topic}")
        rospy.loginfo(f"Publishing mask to: {self.mask_topic}")
        rospy.loginfo(f"Text prompt: {self.text_prompt}")
    
    
    def _load_parameters(self):
        """Load ROS parameters from config file."""
        # Object name parameter (used for default mask topic and text prompt)
        self.object_name = rospy.get_param('~object_name', 'cracker_box')
        rospy.loginfo(f"Object name: {self.object_name}")
        
        # Topic parameters
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
        
        # Grounding DINO parameters
        grounded_sam_config = rospy.get_param('~grounded_sam', {})
        self.groundingdino_checkpoint = rospy.get_param('~grounded_sam/groundingdino_checkpoint',
                                                       grounded_sam_config.get('groundingdino_checkpoint',
                                                       os.path.join(GROUNDED_SAM_PATH, 'checkpoints', 'groundingdino_swint_ogc.pth')))
        self.groundingdino_config = rospy.get_param('~grounded_sam/groundingdino_config',
                                                    grounded_sam_config.get('groundingdino_config',
                                                    os.path.join(GROUNDED_SAM_PATH, 'GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')))
        
        # Text prompt (auto-generate from object_name if empty)
        self.text_prompt = rospy.get_param('~grounded_sam/text_prompt',
                                          grounded_sam_config.get('text_prompt', ''))
        if not self.text_prompt:
            # Generate text prompt from object_name (e.g., "cracker_box" -> "cracker box")
            self.text_prompt = self.object_name.replace('_', ' ')
            rospy.loginfo(f"No text_prompt specified, using: {self.text_prompt}")
        
        # Detection thresholds
        self.box_threshold = rospy.get_param('~grounded_sam/box_threshold',
                                            grounded_sam_config.get('box_threshold', 0.3))
        self.text_threshold = rospy.get_param('~grounded_sam/text_threshold',
                                             grounded_sam_config.get('text_threshold', 0.25))
        self.iou_threshold = rospy.get_param('~grounded_sam/iou_threshold',
                                            grounded_sam_config.get('iou_threshold', 0.5))
        
        # SAM model parameters
        self.sam_model_type = rospy.get_param('~sam/sam_model_type',
                                            rospy.get_param('~sam_model_type', 'vit_h'))
        sam_checkpoint_default = os.path.join(
            os.path.expanduser('~'),
            'hsr_robocanes_omniverse',
            'segment-anything',
            'checkpoints',
            f'sam_{self.sam_model_type}.pth'
        )
        self.sam_checkpoint = rospy.get_param('~sam/sam_checkpoint',
                                             rospy.get_param('~sam_checkpoint', sam_checkpoint_default))
        
        # Resolve path relative to workspace root if needed
        if self.sam_checkpoint and not os.path.isabs(self.sam_checkpoint):
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
            
            # Resolve relative path
            if workspace_root and os.path.exists(workspace_root):
                resolved_path = os.path.join(workspace_root, self.sam_checkpoint)
                if os.path.exists(resolved_path):
                    self.sam_checkpoint = resolved_path
                    rospy.loginfo(f"Resolved SAM checkpoint path to: {self.sam_checkpoint}")
                else:
                    rospy.logwarn(f"Relative path {self.sam_checkpoint} not found at {resolved_path}, using as-is")
            else:
                rospy.logwarn(f"Could not resolve workspace root, using SAM checkpoint path as-is: {self.sam_checkpoint}")
        
        # Device
        self.device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"Using device: {self.device}")
    
    def _initialize_grounding_dino(self):
        """Initialize Grounding DINO model."""
        rospy.loginfo(f"Loading Grounding DINO from {self.groundingdino_checkpoint}")
        try:
            # Expand user path
            config_path = os.path.expanduser(self.groundingdino_config)
            checkpoint_path = os.path.expanduser(self.groundingdino_checkpoint)
            
            if not os.path.exists(config_path):
                rospy.logerr(f"Grounding DINO config not found: {config_path}")
                sys.exit(1)
            if not os.path.exists(checkpoint_path):
                rospy.logerr(f"Grounding DINO checkpoint not found: {checkpoint_path}")
                sys.exit(1)
            
            # Load model
            args = SLConfig.fromfile(config_path)
            args.device = self.device
            self.grounding_dino_model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            load_res = self.grounding_dino_model.load_state_dict(
                clean_state_dict(checkpoint["model"]), strict=False
            )
            rospy.loginfo(f"Grounding DINO load result: {load_res}")
            self.grounding_dino_model = self.grounding_dino_model.to(self.device)
            self.grounding_dino_model.eval()
            
            # Image transform for Grounding DINO
            self.grounding_transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            rospy.loginfo("Grounding DINO model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to initialize Grounding DINO: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            sys.exit(1)
    
    def _initialize_sam(self):
        """Initialize SAM model."""
        rospy.loginfo(f"Loading SAM model: {self.sam_model_type} from {self.sam_checkpoint}")
        try:
            checkpoint_path = os.path.expanduser(self.sam_checkpoint)
            if not os.path.exists(checkpoint_path):
                rospy.logerr(f"SAM checkpoint not found: {checkpoint_path}")
                sys.exit(1)
            
            # Load SAM model
            self.sam = sam_model_registry[self.sam_model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam)
            
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
            # Use the original header timestamp and frame_id for proper synchronization
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
    
    # Grounding DINO Detection
    
    def get_grounding_output(self, image, caption, box_threshold, text_threshold):
        """
        Get object detection output from Grounding DINO.
        
        Args:
            image: PIL Image
            caption: Text prompt (e.g., "cracker box")
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
        
        Returns:
            tuple: (boxes_filt, scores, pred_phrases)
        """
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        # Transform image
        image_tensor, _ = self.grounding_transform(image, None)
        
        # Move to device
        self.grounding_dino_model = self.grounding_dino_model.to(self.device)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.grounding_dino_model(image_tensor[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        
        # Get phrases
        tokenizer = self.grounding_dino_model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())
        
        return boxes_filt, torch.Tensor(scores), pred_phrases
    
    # Performance Monitoring
    
    def get_gpu_usage(self):
        """Get GPU memory and utilization usage."""
        try:
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
            
            # Always include GPU metrics (use 0 if gpu_info is None)
            if gpu_info:
                metric_entry.update({
                    'gpu_memory_allocated_gb': gpu_info.get('memory_allocated_gb', 0),
                    'gpu_memory_reserved_gb': gpu_info.get('memory_reserved_gb', 0),
                    'gpu_memory_total_gb': gpu_info.get('memory_total_gb', 0),
                    'gpu_memory_allocated_pct': gpu_info.get('memory_allocated_pct', 0),
                    'gpu_memory_reserved_pct': gpu_info.get('memory_reserved_pct', 0),
                    'gpu_utilization_pct': gpu_info.get('gpu_util_pct', 0) if gpu_info.get('gpu_util_pct') is not None else 0
                })
            else:
                # Include GPU metrics with default values if gpu_info is None
                metric_entry.update({
                    'gpu_memory_allocated_gb': 0,
                    'gpu_memory_reserved_gb': 0,
                    'gpu_memory_total_gb': 0,
                    'gpu_memory_allocated_pct': 0,
                    'gpu_memory_reserved_pct': 0,
                    'gpu_utilization_pct': 0
                })
            
            # Always include CPU metrics (use 0 if cpu_info is None)
            if cpu_info:
                metric_entry.update({
                    'cpu_process_pct': cpu_info.get('cpu_percent', 0),
                    'cpu_system_pct': cpu_info.get('cpu_system', 0),
                    'memory_mb': cpu_info.get('memory_mb', 0)
                })
            else:
                # Include CPU metrics with default values if cpu_info is None
                metric_entry.update({
                    'cpu_process_pct': 0,
                    'cpu_system_pct': 0,
                    'memory_mb': 0
                })
            
            self.metrics_data.append(metric_entry)
            
            # Save to file periodically (every 10 entries to reduce I/O)
            if len(self.metrics_data) % 10 == 0:
                with open(self.metrics_file, 'w') as f:
                    json.dump(self.metrics_data, f, indent=2)
        except Exception as e:
            rospy.logwarn(f"Failed to save metrics: {e}")
    
    def print_performance_metrics(self, elapsed_time, gpu_info, cpu_info, method_name="Grounded SAM"):
        """Print performance metrics in a formatted way."""
        rospy.loginfo("=" * 80)
        rospy.loginfo(f"PERFORMANCE METRICS - {method_name.upper()}")
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
    
    def segment_with_grounded_sam(self, image_rgb):
        """
        Segment object using Grounding DINO + SAM.
        
        Args:
            image_rgb: RGB image as numpy array
        
        Returns:
            numpy.ndarray: Binary mask, or None if no object detected
        """
        # Get initial resource usage
        gpu_info_before = self.get_gpu_usage()
        cpu_info_before = self.get_cpu_usage()
        
        # Start timing
        start_time = time.time()
        
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            image_pil = Image.fromarray(image_rgb)
            
            # Get bounding boxes from Grounding DINO
            boxes_filt, scores, pred_phrases = self.get_grounding_output(
                image_pil, self.text_prompt, self.box_threshold, self.text_threshold
            )
            
            if boxes_filt.size(0) == 0:
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(5.0, f"No objects detected for prompt: {self.text_prompt} (took {elapsed_time*1000:.2f} ms)")
                return None
            
            rospy.loginfo_throttle(5.0, f"Detected {boxes_filt.size(0)} objects: {pred_phrases}")
            
            # Filter to only valid object detections
            # We pass the prompt to Grounding DINO, and it returns detected phrases
            # We filter to only accept: "cracker box", "craker box" (typo), "red cracker box", or "red box" (failsafe)
            # Reject: standalone "box", "table", or tokenization artifacts
            valid_indices = []
            
            for i, phrase in enumerate(pred_phrases):
                phrase_lower = phrase.lower().strip()
                # Clean up tokenization artifacts (e.g., "##er", "##", etc.)
                # Remove special tokens that Grounding DINO might produce
                phrase_clean = phrase_lower.replace('##', '').replace('#', '').strip()
                
                # Accept if phrase contains:
                # 1. "cracker" or "craker" (typo) - e.g., "cracker box", "craker box", "red cracker box"
                # 2. "red box" (failsafe) - in case "cracker" is not detected
                # Reject standalone "box" or "table" without these keywords
                # Also reject tokenization artifacts (too short)
                has_cracker = 'cracker' in phrase_clean or 'craker' in phrase_clean  # Handle typo
                has_red_box = 'red box' in phrase_clean or phrase_clean == 'red box'
                is_meaningful = len(phrase_clean) > 3  # Must be meaningful phrase (not just "box")
                
                is_valid = (has_cracker or has_red_box) and is_meaningful
                
                if is_valid:
                    valid_indices.append(i)
                    reason = "contains 'cracker/craker'" if has_cracker else "contains 'red box' (failsafe)"
                    rospy.logdebug(f"Accepting detection '{phrase}' -> '{phrase_clean}' - {reason}")
                else:
                    rospy.logdebug(f"Rejecting detection '{phrase}' -> '{phrase_clean}' - does not match target object")
            
            if len(valid_indices) == 0:
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(5.0, f"No valid cracker box detections found. Detected objects: {pred_phrases} (took {elapsed_time*1000:.2f} ms)")
                return None
            
            # Filter to only valid detections
            boxes_filt = boxes_filt[valid_indices]
            scores = scores[valid_indices]
            pred_phrases = [pred_phrases[i] for i in valid_indices]
            rospy.loginfo(f"After filtering: {len(valid_indices)} valid cracker box detections: {pred_phrases}")
            
            # Convert boxes from normalized [cx, cy, w, h] to [x1, y1, x2, y2] in image coordinates
            H, W = image_rgb.shape[:2]
            boxes_xyxy = boxes_filt.clone()
            for i in range(boxes_xyxy.size(0)):
                boxes_xyxy[i] = boxes_xyxy[i] * torch.Tensor([W, H, W, H])
                boxes_xyxy[i][:2] -= boxes_xyxy[i][2:] / 2
                boxes_xyxy[i][2:] += boxes_xyxy[i][:2]
            
            boxes_xyxy = boxes_xyxy.cpu()
            
            # Apply NMS to remove overlapping boxes
            if boxes_xyxy.size(0) > 1:
                nms_idx = torchvision.ops.nms(boxes_xyxy, scores, self.iou_threshold).numpy().tolist()
                boxes_xyxy = boxes_xyxy[nms_idx]
                scores = scores[nms_idx]
                pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                rospy.loginfo_throttle(5.0, f"After NMS: {boxes_xyxy.size(0)} boxes")
            
            # Use the highest confidence box, but validate confidence score
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()
            
            # Require very high confidence score (stricter than box_threshold)
            min_confidence = 0.80  # Require at least 80% confidence (very strict)
            if best_score < min_confidence:
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(2.0, f"Detection confidence too low ({best_score:.3f} < {min_confidence}). Skipping.")
                return None
            
            best_box = boxes_xyxy[best_idx].numpy()
            
            # Set image for SAM predictor
            self.sam_predictor.set_image(image_rgb)
            
            # Transform box for SAM
            transformed_box = self.sam_predictor.transform.apply_boxes_torch(
                torch.from_numpy(best_box[None, :]).to(self.device),
                image_rgb.shape[:2]
            )
            
            # Generate mask
            masks, scores_sam, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False,
            )
            
            # Get the mask (first and only one)
            mask = masks[0, 0].cpu().numpy().astype(np.uint8)
            
            # Validate mask has sufficient pixels (object must be visible)
            # Stricter requirements to reduce false positives
            H, W = image_rgb.shape[:2]
            mask_pixels = mask.sum()
            
            # Base requirement: at least 0.5% of image area (stricter)
            # For very high confidence (>0.85), allow slightly smaller masks (0.4% of image) for side views
            base_min_mask_pixels = max(500, int(W * H * 0.005))  # At least 500 pixels or 0.5% of image
            high_conf_min_mask_pixels = max(400, int(W * H * 0.004))  # At least 400 pixels or 0.4% of image for very high confidence
            
            # Use slightly more lenient threshold only if confidence is very high (for side views)
            if best_score > 0.85:
                min_mask_pixels = high_conf_min_mask_pixels
                rospy.logdebug(f"Very high confidence ({best_score:.3f}), using slightly lenient mask threshold ({min_mask_pixels} pixels)")
            else:
                min_mask_pixels = base_min_mask_pixels
            
            if mask_pixels < min_mask_pixels:
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(2.0, f"Mask too small ({mask_pixels} pixels < {min_mask_pixels}, confidence={best_score:.3f}). Object not properly detected, skipping.")
                return None
            
            # Validate bounding box is reasonable (not too small, not at extreme edges)
            # Stricter requirements to reduce false positives
            x1, y1, x2, y2 = best_box
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            image_area = W * H
            
            # Base requirement: at least 0.5% of image area (stricter)
            # For very high confidence (>0.85), allow slightly smaller boxes (0.4% of image area) for side views
            base_min_box_area = image_area * 0.005  # 0.5% of image area
            high_conf_min_box_area = image_area * 0.004  # 0.4% of image area for very high confidence
            max_box_area = image_area * 0.5    # At most 50% of image area
            
            if best_score > 0.85:
                min_box_area = high_conf_min_box_area
                rospy.logdebug(f"Very high confidence ({best_score:.3f}), using slightly lenient box area threshold ({min_box_area:.0f} pixels)")
            else:
                min_box_area = base_min_box_area
            
            if box_area < min_box_area or box_area > max_box_area:
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(2.0, f"Bounding box size invalid (area={box_area:.0f}, min={min_box_area:.0f}, max={max_box_area:.0f}, confidence={best_score:.3f}). Skipping.")
                return None
            
            # Check if box is too close to image edges (likely partial/false detection)
            # For high confidence, be more lenient with edge margin (object might be partially visible)
            # Lowered threshold to 0.82 so that confidence > 0.82 uses very lenient margin
            if best_score > 0.82:
                margin = min(W, H) * 0.02  # 2% margin for very high confidence (very lenient - allows edge detections)
                rospy.logdebug(f"Very high confidence ({best_score:.3f}), using lenient edge margin ({margin:.0f} pixels)")
            elif best_score > 0.80:
                margin = min(W, H) * 0.03  # 3% margin for high confidence (moderate leniency)
                rospy.logdebug(f"High confidence ({best_score:.3f}), using moderate edge margin ({margin:.0f} pixels)")
            else:
                margin = min(W, H) * 0.15  # 15% margin for lower confidence (strict)
            
            if x1 < margin or y1 < margin or x2 > (W - margin) or y2 > (H - margin):
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(2.0, f"Bounding box too close to edges (x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}, margin={margin:.0f}, confidence={best_score:.3f}). Skipping.")
                return None
            
            # Validate aspect ratio (cracker box should be roughly rectangular, not too elongated)
            aspect_ratio = box_width / box_height if box_height > 0 else 0
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(2.0, f"Bounding box aspect ratio invalid ({aspect_ratio:.2f}). Likely false detection, skipping.")
                return None
            
            # Additional validation: mask should cover reasonable portion of bounding box
            mask_coverage_ratio = mask_pixels / box_area if box_area > 0 else 0
            min_mask_coverage = 0.4  # Mask should cover at least 40% of bounding box (stricter)
            if mask_coverage_ratio < min_mask_coverage:
                elapsed_time = time.time() - start_time
                rospy.logwarn_throttle(2.0, f"Mask coverage too low ({mask_coverage_ratio:.2%} < {min_mask_coverage:.0%}). Likely false detection, skipping.")
                return None
            
            # Convert to binary mask (0 or 255)
            binary_mask = (mask * 255).astype(np.uint8)
            
            rospy.loginfo_throttle(5.0, f"Generated mask for: {pred_phrases[best_idx]} (mask pixels: {mask_pixels}, box area: {box_area:.0f})")
            
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
            if self.frame_counter % self.performance_print_interval == 0:
                self.print_performance_metrics(elapsed_time, gpu_info, cpu_info, "Grounded SAM Segmentation")
            else:
                # Just log time for non-printed frames
                rospy.logdebug(f"Grounded SAM: {elapsed_time*1000:.1f} ms")
            
            # Save metrics to file
            self._save_metrics(elapsed_time, gpu_info, cpu_info)
            
            return binary_mask
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            rospy.logerr(f"Error in segment_with_grounded_sam (took {elapsed_time*1000:.2f} ms): {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None
    
    
    def image_callback(self, img_msg):
        """Callback for RGB image messages."""
        try:
            # Store original header - mask needs same timestamp as RGB
            original_header = img_msg.header
            
            # Convert ROS image to numpy
            image_rgb = self.ros_image_to_numpy(img_msg, desired_encoding='rgb8')
            
            # Segment object
            mask = self.segment_with_grounded_sam(image_rgb)
            
            if mask is None:
                # Publish empty mask with ORIGINAL RGB timestamp
                empty_mask = np.zeros((img_msg.height, img_msg.width), dtype=np.uint8)
                mask_msg = self.numpy_to_ros_image(empty_mask, encoding='mono8', frame_id=self.frame_id, header=original_header)
                self.mask_pub.publish(mask_msg)
                return
            
            # Publish mask with original RGB timestamp
            mask_msg = self.numpy_to_ros_image(mask, encoding='mono8', frame_id=self.frame_id, header=original_header)
            self.mask_pub.publish(mask_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())


def main():
    """Main function."""
    node = None
    try:
        node = GroundedSAMSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error in Grounded SAM node: {e}")
        import traceback
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
        rospy.logerr(traceback.format_exc())


if __name__ == '__main__':
    main()

