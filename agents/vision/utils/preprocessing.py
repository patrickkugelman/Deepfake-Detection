import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(image_path, target_size=(299, 299)):
    """
    Preprocess image for Xception input
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
    
    Returns:
        Tensor of shape (1, 3, H, W)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert to PIL for torchvision transforms
    img = Image.fromarray(img)
    
    # Normalization (ImageNet stats)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor: torch.Tensor = transform(img) # type: ignore
    
    # Add batch dimension
    return tensor.unsqueeze(0)

def extract_frames(video_path, num_frames=10):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
    
    Returns:
        List of frame arrays
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    return frames

def detect_faces(image):
    """
    Detect faces in image using MTCNN
    
    Args:
        image: numpy array (RGB)
    
    Returns:
        List of face bounding boxes [(x1, y1, x2, y2), ...]
    """
    from facenet_pytorch import MTCNN
    
    # Initialize detector
    mtcnn = MTCNN(keep_all=True, device='cpu')
    
    # Convert to PIL
    pil_image = Image.fromarray(image)
    
    # Detect faces
    # Detect faces
    # Pylance sees this as potentially returning 3 values, so we avoid direct unpacking
    detection = mtcnn.detect(pil_image)
    boxes = detection[0] if detection is not None else None
    
    if boxes is None:
        return []
    
    return boxes.tolist()

def crop_face(image, box, scale=1.3):
    """
    Crop face from image with padding
    
    Args:
        image: numpy array (RGB)
        box: (x1, y1, x2, y2)
        scale: Scale factor for padding
    
    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Calculate center and size
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    size = int(max(x2 - x1, y2 - y1) * scale)
    
    # Calculate crop coordinates
    x1_crop = max(0, cx - size // 2)
    y1_crop = max(0, cy - size // 2)
    x2_crop = min(image.shape[1], cx + size // 2)
    y2_crop = min(image.shape[0], cy + size // 2)
    
    # Crop
    face = image[y1_crop:y2_crop, x1_crop:x2_crop]
    
    return face