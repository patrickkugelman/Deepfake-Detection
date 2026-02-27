"""
Code Execution Tool
Sandboxed Python execution for custom analysis scripts
"""
import logging
import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, List, Optional
import resource

logger = logging.getLogger("CodeExecutor")

# Allowed imports whitelist
ALLOWED_IMPORTS = [
    "numpy", "cv2", "PIL", "PIL.Image", "json", "math", "statistics",
    "collections", "itertools", "functools", "operator", "datetime",
    "time", "random", "re", "string", "base64", "hashlib", "struct"
]

# Blocked imports (security)
BLOCKED_IMPORTS = [
    "os", "sys", "subprocess", "shutil", "socket", "urllib", "http",
    "ftplib", "smtplib", "pickle", "marshal", "eval", "exec", "compile",
    "__builtin__", "__builtins__", "builtins", "importlib", "imp"
]

def execute_python(
    code: str, 
    timeout: int = 10,
    imports_allowed: Optional[List[str]] = None,
    max_memory_mb: int = 512
) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment
    
    Security:
    - Timeout enforcement
    - Memory limits
    - Import restrictions
    - Output sanitization
    """
    if imports_allowed is None:
        imports_allowed = ALLOWED_IMPORTS
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    try:
        # Wrap code with import restrictions
        restricted_code = f"""
import sys

# Block dangerous imports
_blocked = {BLOCKED_IMPORTS}
_original_import = __builtins__.__import__

def _safe_import(name, *args, **kwargs):
    if name in _blocked:
        raise ImportError(f"Import of '{{name}}' is not allowed for security reasons")
    return _original_import(name, *args, **kwargs)

__builtins__.__import__ = _safe_import

# User code
{code}
"""
        
        temp_file.write(restricted_code)
        temp_file.close()
        
        # Set resource limits
        def set_limits():
            # Memory limit (RSS in bytes)
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, max_memory_mb * 1024 * 1024))
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        
        # Execute with timeout
        try:
            result = subprocess.run(
                [sys.executable, temp_file.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=set_limits if hasattr(os, 'fork') else None
            )
            
            output = result.stdout
            error = result.stderr
            
            # Sanitize output (remove file paths)
            output = output.replace(temp_file.name, "<temp_file>")
            error = error.replace(temp_file.name, "<temp_file>")
            
            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timeout after {timeout} seconds",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1
            }
    finally:
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def analyze_metadata_custom(file_path: str, script: str) -> Dict[str, Any]:
    """
    Execute custom metadata analysis script on file
    """
    code = f"""
import json
from pathlib import Path

file_path = "{file_path}"

{script}

# Result should be in 'result' variable
result = locals().get('result', {{"error": "No result variable found"}})
print(json.dumps(result))
"""
    return execute_python(code, timeout=15)

def custom_analysis(file_path: str, analysis_script: str) -> Dict[str, Any]:
    """
    Execute custom analysis script on file
    """
    code = f"""
import json
from pathlib import Path

file_path = "{file_path}"

{analysis_script}

# Result should be in 'result' variable
result = locals().get('result', {{"error": "No result variable found"}})
print(json.dumps(result))
"""
    return execute_python(code, timeout=30)

