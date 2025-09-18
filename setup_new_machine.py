#!/usr/bin/env python3
"""
Setup script for migrating Tacotron2 TTS project to new RTX 3060 machine
"""

import os
import sys
import json
import torch
import subprocess
from pathlib import Path

def check_cuda_installation():
    """Verify CUDA installation and GPU availability"""
    print("=== CUDA Installation Check ===")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA driver installed")
            print(result.stdout.split('\n')[0])  # Driver version line
        else:
            print("✗ NVIDIA driver not found")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi command not found")
        return False
    
    # Check PyTorch CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ PyTorch CUDA available")
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ VRAM: {gpu_memory:.1f} GB")
        
        if "RTX 3060" in gpu_name:
            print("✓ RTX 3060 detected")
        else:
            print(f"⚠ Expected RTX 3060, found: {gpu_name}")
        
        return True
    else:
        print("✗ PyTorch CUDA not available")
        return False

def verify_project_structure():
    """Verify project directory structure"""
    print("\n=== Project Structure Verification ===")
    
    required_dirs = [
        "tacotron2",
        "tacotron2/models",
        "tacotron2/training",
        "tacotron2/configs",
        "tacotron2/utils",
        "tacotron2/inference",
        "data"
    ]
    
    required_files = [
        "tacotron2/models/tacotron2.py",
        "tacotron2/training/train.py",
        "tacotron2/configs/config.py",
        "train_tacotron2.py",
        "requirements.txt"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - MISSING")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_good = False
    
    return all_good

def check_data_integrity():
    """Check preprocessed data integrity"""
    print("\n=== Data Integrity Check ===")
    
    preprocessed_path = "data/LJSpeech-1.1/preprocessed"
    metadata_file = os.path.join(preprocessed_path, "metadata.json")
    
    if not os.path.exists(preprocessed_path):
        print(f"✗ Preprocessed data directory not found: {preprocessed_path}")
        print("  Please transfer your preprocessed data to this location")
        return False
    
    if not os.path.exists(metadata_file):
        print(f"✗ Metadata file not found: {metadata_file}")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded: {len(metadata)} samples")
        
        # Check for some data files
        sample_count = min(5, len(metadata))
        missing_files = 0
        
        for i, item in enumerate(metadata[:sample_count]):
            mel_file = os.path.join(preprocessed_path, item['mel_file'])
            token_file = os.path.join(preprocessed_path, item['token_file'])
            
            if not os.path.exists(mel_file):
                missing_files += 1
            if not os.path.exists(token_file):
                missing_files += 1
        
        if missing_files == 0:
            print(f"✓ Sample data files verified ({sample_count} samples checked)")
        else:
            print(f"✗ {missing_files} data files missing in sample check")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading metadata: {e}")
        return False

def update_config_paths():
    """Update configuration paths for new machine"""
    print("\n=== Configuration Update ===")
    
    config_file = "tacotron2/configs/config.py"
    
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        return False
    
    # Get current working directory
    current_dir = os.path.abspath(".")
    
    # Read current config
    with open(config_file, 'r') as f:
        config_content = f.read()
    
    # Update paths
    new_data_path = os.path.join(current_dir, "data", "LJSpeech-1.1")
    new_preprocessed_path = os.path.join(new_data_path, "preprocessed")
    new_checkpoint_dir = os.path.join(current_dir, "tacotron2", "checkpoints")
    new_log_dir = os.path.join(current_dir, "tacotron2", "logs")
    
    # Replace paths in config
    config_content = config_content.replace(
        'DATA_PATH = r"D:\Project\data\LJSpeech-1.1"',
        f'DATA_PATH = r"{new_data_path}"'
    )
    config_content = config_content.replace(
        'PREPROCESSED_PATH = r"D:\Project\data\LJSpeech-1.1\preprocessed"',
        f'PREPROCESSED_PATH = r"{new_preprocessed_path}"'
    )
    config_content = config_content.replace(
        'CHECKPOINT_DIR = r"D:\Project\tacotron2\checkpoints"',
        f'CHECKPOINT_DIR = r"{new_checkpoint_dir}"'
    )
    config_content = config_content.replace(
        'LOG_DIR = r"D:\Project\tacotron2\logs"',
        f'LOG_DIR = r"{new_log_dir}"'
    )
    
    # Optimize for RTX 3060
    config_content = config_content.replace(
        'BATCH_SIZE = 8',
        'BATCH_SIZE = 6  # Optimized for RTX 3060'
    )
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✓ Configuration paths updated")
    print(f"  Data path: {new_data_path}")
    print(f"  Preprocessed: {new_preprocessed_path}")
    print(f"  Checkpoints: {new_checkpoint_dir}")
    print(f"  Logs: {new_log_dir}")
    print("✓ Batch size optimized for RTX 3060")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n=== Directory Creation ===")
    
    dirs_to_create = [
        "tacotron2/checkpoints",
        "tacotron2/logs",
        "outputs",
        "synthesis_tests"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {dir_path}/")

def test_model_loading():
    """Test if model can be loaded successfully"""
    print("\n=== Model Loading Test ===")
    
    try:
        sys.path.append('.')
        from tacotron2.models.tacotron2 import Tacotron2, HParams
        
        hparams = HParams()
        model = Tacotron2(hparams)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print("✓ Model moved to GPU")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully")
        print(f"✓ Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Tacotron2 TTS Project Setup for RTX 3060")
    print("=" * 50)
    
    checks = [
        ("CUDA Installation", check_cuda_installation),
        ("Project Structure", verify_project_structure),
        ("Data Integrity", check_data_integrity),
        ("Configuration Update", update_config_paths),
        ("Directory Creation", create_directories),
        ("Model Loading", test_model_loading)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"✗ {check_name} failed with error: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python train_tacotron2.py")
        print("2. Monitor: tensorboard --logdir tacotron2/logs")
        print("3. Check GPU: watch -n 1 nvidia-smi")
    else:
        print("❌ Setup incomplete. Please fix the failed checks above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
