"""
Video Preprocessing Utilities
Robust frame extraction and video processing
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def extract_frames(
    video_path: str,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (112, 112)
) -> Optional[np.ndarray]:
    """
    Extract frames from video with uniform sampling
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: Target frame size (width, height)
    
    Returns:
        numpy array of shape (C, T, H, W) or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            logger.error("Video has 0 frames")
            cap.release()
            return None
        
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Padding if necessary
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        # Convert to numpy array and normalize
        frames_array = np.array(frames, dtype=np.float32)
        
        # Normalize: [0, 255] -> [0, 1]
        # FIX: Cast to float32 explicitly to avoid type issues
        frames_array = frames_array.astype(np.float32)
        frames_array = frames_array / 255.0  # ✅ Fix Pylance error
        
        # Rearrange dimensions: (T, H, W, C) -> (C, T, H, W)
        frames_array = frames_array.transpose(3, 0, 1, 2)
        
        # Apply normalization (ImageNet stats)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1, 1)
        frames_array = (frames_array - mean) / std
        
        return frames_array
        
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return None

def validate_video(video_path: str) -> bool:
    """
    Check if video file is valid
    
    Args:
        video_path: Path to video file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    except:
        return False

def get_video_info(video_path: str) -> Optional[dict]:
    """
    Get video metadata
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video info or None
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        
        # Calculate duration
        if info['fps'] > 0:
            info['duration'] = info['total_frames'] / info['fps']
        else:
            info['duration'] = 0
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return None

def detect_temporal_inconsistencies(video_path: str) -> float:
    """
    Detect temporal inconsistencies in video (heuristic placeholder)
    
    Args:
        video_path: Path to video file
        
    Returns:
        float: Inconsistency score between 0.0 and 1.0
    """
    # TODO: Implement real optical flow analysis
    return 0.0