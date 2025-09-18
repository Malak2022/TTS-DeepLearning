# Tacotron2 TTS Project Migration Guide

## Executive Summary

This document provides the technical procedures for migrating and resuming training of the Tacotron2 TTS project on a new RTX 3060 system. The guide covers environment setup, data transfer, configuration updates, and training resumption.

## System Requirements

### Hardware Specifications
- NVIDIA RTX 3060 (12GB VRAM)
- Minimum 16GB System RAM
- 50GB available storage
- CUDA-compatible motherboard

### Software Dependencies
- Python 3.8-3.10
- CUDA Toolkit 11.8+
- cuDNN 8.x
- Git version control

## Migration Procedure

### Phase 1: Environment Configuration

#### 1.1 CUDA Installation Verification
```bash
nvidia-smi
nvcc --version
```

#### 1.2 Python Environment Setup
```bash
python -m venv tacotron2_env
source tacotron2_env/bin/activate  # Linux/macOS
# tacotron2_env\Scripts\activate   # Windows

pip install --upgrade pip setuptools wheel
```

#### 1.3 PyTorch Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 1.4 Project Dependencies
```bash
pip install numpy pandas librosa matplotlib tqdm scipy unidecode jupyter tensorboard soundfile
```

### Phase 2: Project Migration

#### 2.1 Repository Clone
```bash
git clone https://github.com/Malak2022/TTS_deepLearningProject_Teknozor.git
cd TTS_deepLearningProject_Teknozor
```

#### 2.2 Data Transfer Protocol
Transfer the preprocessed LJSpeech data to the new system:
- Source: `data/LJSpeech-1.1/preprocessed/`
- Target: `data/LJSpeech-1.1/preprocessed/`
- Required files: `metadata.json`, `*.npy` mel spectrograms, `*.npy` token files

#### 2.3 Configuration Update
Update `tacotron2/configs/config.py` with new system paths:
```python
DATA_PATH = "/path/to/new/data/LJSpeech-1.1"
PREPROCESSED_PATH = "/path/to/new/data/LJSpeech-1.1/preprocessed"
CHECKPOINT_DIR = "/path/to/new/tacotron2/checkpoints"
LOG_DIR = "/path/to/new/tacotron2/logs"
```

### Phase 3: Training Resumption

#### 3.1 Checkpoint Transfer
If resuming from previous training:
```bash
# Copy existing checkpoint to new system
cp /old/path/tacotron2/checkpoints/*.pth tacotron2/checkpoints/
```

#### 3.2 Training Execution
```bash
python train_tacotron2.py
```

#### 3.3 Training Monitoring
```bash
tensorboard --logdir tacotron2/logs --port 6006
```

## RTX 3060 Optimization Parameters

### Memory Configuration
```python
# Recommended settings for RTX 3060
BATCH_SIZE = 6
GRAD_ACCUMULATION_STEPS = 2
MIXED_PRECISION = True
```

### Performance Tuning
```python
# Optimal hyperparameters
LEARNING_RATE = 1e-3
GRAD_CLIP_THRESH = 1.0
N_EPOCHS = 100
```

## Validation Procedures

### System Verification
```python
import torch
assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.get_device_properties(0).total_memory > 10e9, "Insufficient VRAM"
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Data Integrity Check
```python
import os, json
metadata_path = "data/LJSpeech-1.1/preprocessed/metadata.json"
assert os.path.exists(metadata_path), "Metadata file missing"
with open(metadata_path) as f:
    data = json.load(f)
print(f"Dataset size: {len(data)} samples")
```

### Model Verification
```python
from tacotron2.models.tacotron2 import Tacotron2, HParams
model = Tacotron2(HParams())
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
```

## Training Execution Scripts

### Primary Training Script
```bash
python train_tacotron2.py \
    --batch_size 6 \
    --learning_rate 1e-3 \
    --epochs 100 \
    --checkpoint_interval 1000 \
    --validation_interval 500
```

### Resume Training Script
```bash
python resume_training.py \
    --checkpoint tacotron2/checkpoints/latest_checkpoint.pth \
    --batch_size 6
```

### Synthesis Testing Script
```bash
python synthesize_text.py \
    --checkpoint tacotron2/checkpoints/best_checkpoint.pth \
    --text "Hello world, this is a test synthesis." \
    --output test_synthesis.wav
```

## Performance Monitoring

### GPU Utilization
```bash
watch -n 1 nvidia-smi
```

### Training Metrics
Monitor via TensorBoard:
- Training loss convergence
- Validation loss trends
- Attention alignment quality
- Learning rate scheduling

### Expected Performance Metrics
- Training speed: ~2.5 seconds/batch
- Memory utilization: ~9GB VRAM
- Convergence time: 15-20 hours

## Troubleshooting Protocol

### Memory Issues
```python
# Reduce batch size
BATCH_SIZE = 4
# Enable gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential
```

### Training Instability
```python
# Reduce learning rate
LEARNING_RATE = 5e-4
# Increase gradient clipping
GRAD_CLIP_THRESH = 0.5
```

### Data Loading Errors
```bash
# Verify data paths and permissions
ls -la data/LJSpeech-1.1/preprocessed/
python -c "import json; print(len(json.load(open('data/LJSpeech-1.1/preprocessed/metadata.json'))))"
```

## Quality Assurance

### Synthesis Quality Check
```python
python test_synthesis.py --checkpoint best_checkpoint.pth
```

### Model Evaluation
```python
python evaluate_model.py --checkpoint best_checkpoint.pth --test_set validation
```

## Deployment Completion

Upon successful migration:
1. Verify training resumption without errors
2. Confirm TensorBoard logging functionality
3. Validate synthesis output quality
4. Document final configuration parameters
5. Establish backup procedures for checkpoints

This migration procedure ensures seamless transition of the Tacotron2 TTS project to the RTX 3060 system with optimal performance configuration.
