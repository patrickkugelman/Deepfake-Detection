"""
Metadata Inspection Tool
Inspects file metadata to inform routing decisions (e.g., audio tracks in video)
"""
import os
import logging
import subprocess
import tempfile
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger("MetadataTool")

def inspect_file_metadata(file_path: str, file_content: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Inspect file metadata without full analysis.
    Returns information about format, codecs, tracks, etc.
    """
    try:
        # If file_content is provided, write to temp file
        if file_content:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_path).suffix)
            temp_file.write(file_content)
            temp_file.close()
            actual_path = temp_file.name
        else:
            actual_path = file_path
        
        extension = Path(file_path).suffix.lower()
        result = {
            "file_path": file_path,
            "extension": extension,
            "has_audio": False,
            "has_video": False,
            "duration": None,
            "codec": None,
            "sample_rate": None,
            "channels": None,
            "fps": None,
            "resolution": None,
            "error": None
        }
        
        # Video file inspection
        if extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            result["has_video"] = True
            try:
                # Use ffprobe if available
                cmd = [
                    'ffprobe', '-v', 'error', '-show_entries',
                    'format=duration:stream=codec_name,codec_type,sample_rate,channels,width,height,r_frame_rate',
                    '-of', 'json', actual_path
                ]
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
                import json
                probe_data = json.loads(output)
                
                # Extract duration
                if 'format' in probe_data and 'duration' in probe_data['format']:
                    result["duration"] = float(probe_data['format']['duration'])
                
                # Check for audio/video streams
                if 'streams' in probe_data:
                    for stream in probe_data['streams']:
                        codec_type = stream.get('codec_type', '')
                        if codec_type == 'audio':
                            result["has_audio"] = True
                            result["sample_rate"] = stream.get('sample_rate')
                            result["channels"] = stream.get('channels')
                        elif codec_type == 'video':
                            result["codec"] = stream.get('codec_name')
                            result["fps"] = stream.get('r_frame_rate')
                            width = stream.get('width')
                            height = stream.get('height')
                            if width and height:
                                result["resolution"] = f"{width}x{height}"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # Fallback: basic file check
                result["has_audio"] = None  # Unknown
                logger.warning(f"ffprobe not available, using fallback for {file_path}")
        
        # Audio file inspection
        elif extension in ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg']:
            result["has_audio"] = True
            try:
                cmd = [
                    'ffprobe', '-v', 'error', '-show_entries',
                    'format=duration:stream=sample_rate,channels',
                    '-of', 'json', actual_path
                ]
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
                import json
                probe_data = json.loads(output)
                
                if 'format' in probe_data and 'duration' in probe_data['format']:
                    result["duration"] = float(probe_data['format']['duration'])
                
                if 'streams' in probe_data and len(probe_data['streams']) > 0:
                    stream = probe_data['streams'][0]
                    result["sample_rate"] = stream.get('sample_rate')
                    result["channels"] = stream.get('channels')
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning(f"ffprobe not available for audio inspection")
        
        # Image file - basic check
        elif extension in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
            result["has_video"] = False
            result["has_audio"] = False
        
        # Cleanup temp file
        if file_content and os.path.exists(actual_path):
            os.unlink(actual_path)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Metadata inspection error: {e}")
        return {
            "file_path": file_path,
            "error": str(e),
            "has_audio": None,
            "has_video": None
        }

def extract_video_info(file_path: str, file_content: Optional[bytes] = None) -> Dict[str, Any]:
    """Extract detailed video information"""
    metadata = inspect_file_metadata(file_path, file_content)
    return {
        "duration": metadata.get("duration"),
        "codec": metadata.get("codec"),
        "fps": metadata.get("fps"),
        "resolution": metadata.get("resolution"),
        "has_audio": metadata.get("has_audio"),
        "error": metadata.get("error")
    }

def extract_audio_info(file_path: str, file_content: Optional[bytes] = None) -> Dict[str, Any]:
    """Extract detailed audio information"""
    metadata = inspect_file_metadata(file_path, file_content)
    return {
        "duration": metadata.get("duration"),
        "sample_rate": metadata.get("sample_rate"),
        "channels": metadata.get("channels"),
        "error": metadata.get("error")
    }

