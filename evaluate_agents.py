#!/usr/bin/env python3
"""
DeepGuard Pro - Agent Evaluation Script
=======================================
Benchmarks all running agents against local datasets using F1-Score, Accuracy, Precision, and Recall.

Usage:
    python evaluate_agents.py --agent all
    python evaluate_agents.py --agent vision
"""

import os
import argparse
import requests
import glob
import concurrent.futures
import time
from typing import List, Dict, Tuple, Any
from pathlib import Path

# Configuration
AGENTS = {
    "vision": {"port": 8002, "ext": ["jpg", "png", "jpeg", "webp"]},
    "audio": {"port": 8004, "ext": ["wav", "mp3", "flac"]},
    "video": {"port": 8003, "ext": ["mp4", "avi", "mov"]},
    "text": {"port": 8001, "ext": ["txt", "md"]},
    "bio": {"port": 8005, "ext": ["mp4", "avi", "mov"]}  # Uses video files
}

API_TIMEOUT = 30  # Seconds

def get_files(agent_name: str, label: str) -> List[str]:
    """Get list of files for a specific agent and label (Real/Fake)"""
    base_dir = Path(f"agents/{agent_name}/dataset/{label}")
    if not base_dir.exists():
         # Fallback for capitalization
        base_dir = Path(f"agents/{agent_name}/dataset/{label.lower()}")
        if not base_dir.exists():
            return []
    
    extensions = AGENTS[agent_name]["ext"]
    files = []
    for ext in extensions:
        files.extend(list(base_dir.glob(f"*.{ext}")))
        files.extend(list(base_dir.glob(f"*.{ext.upper()}")))
    
    return [str(f) for f in files]

def query_agent(agent_name: str, file_path: str, expected_label: int) -> Dict[str, Any]:
    """Send file to agent and compare verdict"""
    url = f"http://localhost:{AGENTS[agent_name]['port']}/analyze"
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            start_time = time.time()
            response = requests.post(url, files=files, timeout=API_TIMEOUT)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                verdict = result.get("verdict", "Inconclusive").lower()
                
                # Logic: Real=0, Fake=1
                # Text agent might return "AI-Generated" or "Human-Written"
                # Standardize to 0 (Real) and 1 (Fake)
                
                pred_label = -1
                if "real" in verdict or "human" in verdict or "authentic" in verdict:
                    pred_label = 0
                elif "fake" in verdict or "ai" in verdict or "synthetic" in verdict:
                    pred_label = 1
                
                return {
                    "file": os.path.basename(file_path),
                    "expected": expected_label,
                    "predicted": pred_label,
                    "verdict": verdict,
                    "confidence": result.get("fake_probability", 0) if pred_label == 1 else (100 - result.get("fake_probability", 0)),
                    "duration": duration,
                    "error": None
                }
            else:
                 return {
                    "file": os.path.basename(file_path),
                    "expected": expected_label,
                    "predicted": -1,
                    "error": f"HTTP {response.status_code}"
                }
                
    except Exception as e:
        return {
            "file": os.path.basename(file_path),
            "expected": expected_label,
            "predicted": -1,
            "error": str(e)
        }

def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate F1, Accuracy, Precision, Recall"""
    tp = 0 # Fake correctly predicted as Fake
    fp = 0 # Real incorrectly predicted as Fake
    tn = 0 # Real correctly predicted as Real
    fn = 0 # Fake incorrectly predicted as Real
    
    valid_results = [r for r in results if r["predicted"] != -1 and r["error"] is None]
    
    if not valid_results:
        return {}

    for r in valid_results:
        if r["expected"] == 1: # Ground Truth: Fake
            if r["predicted"] == 1:
                tp += 1
            else:
                fn += 1
        else: # Ground Truth: Real
            if r["predicted"] == 0:
                tn += 1
            else:
                fp += 1
                
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": len(valid_results),
        "errors": len(results) - len(valid_results)
    }

def evaluate_agent(name: str):
    print(f"\n🔍 Evaluating {name.upper()} Agent...")
    
    real_files = get_files(name, "Real")
    fake_files = get_files(name, "Fake")
    
    if not real_files and not fake_files:
        print(f"⚠️ No dataset found for {name}. Check agents/{name}/dataset/")
        return
        
    print(f"   Found {len(real_files)} Real, {len(fake_files)} Fake samples.")
    
    all_tasks = []
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for f in real_files:
            all_tasks.append(executor.submit(query_agent, name, f, 0))
        for f in fake_files:
            all_tasks.append(executor.submit(query_agent, name, f, 1))
            
        total = len(all_tasks)
        completed = 0
        for future in concurrent.futures.as_completed(all_tasks):
            results.append(future.result())
            completed += 1
            print(f"   Progress: {completed}/{total}", end="\r")
            
    print("\n")
    metrics = calculate_metrics(results)
    
    if metrics:
        print(f"📊 Results for {name.upper()}:")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   Matrix:    TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
        print(f"   Errors:    {metrics['errors']}")
        
        # Calculate avg latency
        valid_times = [r['duration'] for r in results if r['error'] is None]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        print(f"   Avg Time:  {avg_time:.2f}s")
    else:
        print("❌ No valid results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard Agents")
    parser.add_argument("--agent", type=str, default="all", help="Agent to evaluate (vision, audio, video, text, bio, all)")
    args = parser.parse_args()
    
    if args.agent == "all":
        for agent in AGENTS:
            evaluate_agent(agent)
    elif args.agent in AGENTS:
        evaluate_agent(args.agent)
    else:
        print(f"Unknown agent: {args.agent}")
