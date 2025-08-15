#!/usr/bin/env python3
"""Check alignment between .cursorrules, repo structure, and documentation."""

import os
import re
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_repo_structure():
    """Check repo structure matches .cursorrules."""
    print("\n=== Checking Repo Structure ===")
    
    required_dirs = [
        "src/inzwa/api",
        "src/inzwa/asr", 
        "src/inzwa/llm",
        "src/inzwa/tts",
        "src/inzwa/orch",  # Not orchestration!
        "src/inzwa/ops",   # Not config/telemetry!
        "src/inzwa/ui",
        "tests"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"{GREEN}✓{RESET} {dir_path} exists")
        else:
            print(f"{RED}✗{RESET} {dir_path} missing")
    
    # Check for disallowed directories
    disallowed = ["src/inzwa/orchestration", "src/inzwa/config", "src/inzwa/telemetry"]
    for dir_path in disallowed:
        if os.path.exists(dir_path):
            print(f"{RED}✗{RESET} {dir_path} should not exist (use ops/ instead)")

def check_settings():
    """Check settings file matches .cursorrules template."""
    print("\n=== Checking Settings ===")
    
    settings_file = "src/inzwa/ops/settings.py"
    if not os.path.exists(settings_file):
        print(f"{RED}✗{RESET} {settings_file} missing")
        return
    
    with open(settings_file) as f:
        content = f.read()
    
    # Check for required settings
    required = [
        'env_prefix="INZWA_"',
        "request_timeout_s",
        "max_text_chars",
        "max_audio_seconds",
        "backpressure_threshold"
    ]
    
    for item in required:
        if item in content:
            print(f"{GREEN}✓{RESET} {item} found")
        else:
            print(f"{RED}✗{RESET} {item} missing")

def check_function_lengths():
    """Check all functions are <50 lines."""
    print("\n=== Checking Function Lengths ===")
    
    issues = []
    for py_file in Path("src").rglob("*.py"):
        with open(py_file) as f:
            lines = f.readlines()
        
        in_function = False
        func_start = 0
        func_name = ""
        
        for i, line in enumerate(lines):
            if line.strip().startswith("def ") or line.strip().startswith("async def "):
                if in_function:
                    # Check previous function
                    func_lines = i - func_start
                    if func_lines > 50:
                        issues.append(f"{py_file}:{func_name} ({func_lines} lines)")
                
                in_function = True
                func_start = i
                func_name = line.strip().split("(")[0].replace("def ", "").replace("async ", "")
            
        # Check last function
        if in_function:
            func_lines = len(lines) - func_start
            if func_lines > 50:
                issues.append(f"{py_file}:{func_name} ({func_lines} lines)")
    
    if issues:
        for issue in issues[:5]:  # Show first 5
            print(f"{RED}✗{RESET} {issue}")
    else:
        print(f"{GREEN}✓{RESET} All functions <50 lines")

def check_dependencies():
    """Check dependencies are minimal per .cursorrules."""
    print("\n=== Checking Dependencies ===")
    
    pyproject = "pyproject.toml"
    if not os.path.exists(pyproject):
        print(f"{RED}✗{RESET} pyproject.toml missing")
        return
    
    with open(pyproject) as f:
        content = f.read()
    
    # Required dependencies
    required = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "prometheus-client"
    ]
    
    # Disallowed heavy deps (unless justified)
    disallowed = [
        "tensorflow",  # Too heavy
        "django",      # Wrong framework
        "flask",       # Wrong framework
    ]
    
    for dep in required:
        if dep in content:
            print(f"{GREEN}✓{RESET} {dep} present")
        else:
            print(f"{YELLOW}⚠{RESET} {dep} missing")
    
    for dep in disallowed:
        if dep in content:
            print(f"{RED}✗{RESET} {dep} should not be used (too heavy)")

def check_api_routes():
    """Check API routes match .cursorrules."""
    print("\n=== Checking API Routes ===")
    
    required_routes = [
        "/ws/audio",
        "/v1/tts", 
        "/v1/chat",
        "/healthz",
        "/readyz",
        "/v1/admin/warmup"
    ]
    
    disallowed_routes = [
        "/v1/asr"  # Should use WebSocket for ASR
    ]
    
    app_file = "src/inzwa/api/app.py"
    if os.path.exists(app_file):
        with open(app_file) as f:
            content = f.read()
        
        for route in required_routes:
            if route in content:
                print(f"{GREEN}✓{RESET} {route} implemented")
            else:
                print(f"{YELLOW}⚠{RESET} {route} not found")
        
        for route in disallowed_routes:
            if route in content:
                print(f"{RED}✗{RESET} {route} should not exist (use WebSocket)")

def check_performance_budgets():
    """Check performance budgets are documented."""
    print("\n=== Checking Performance Budgets ===")
    
    budgets = {
        "P50 TTFW": "500ms",
        "P95 round-trip": "1200ms",
        "ASR RTF": "0.2-0.5x",
        "LLM TTFB": "600-900ms CPU",
        "Token rate": ">=10-30 tok/s CPU"
    }
    
    for metric, target in budgets.items():
        print(f"{YELLOW}⚠{RESET} {metric}: {target} (needs testing)")

def main():
    print(f"\n{'='*50}")
    print("Inzwa Alignment Check (.cursorrules compliance)")
    print(f"{'='*50}")
    
    check_repo_structure()
    check_settings()
    check_function_lengths()
    check_dependencies()
    check_api_routes()
    check_performance_budgets()
    
    print(f"\n{'='*50}")
    print("Check complete. Fix any ✗ issues for full compliance.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
