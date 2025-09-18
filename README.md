# 🗣️ Text-to-Speech (TTS) Deep Learning Project

## 🎯 Project Overview

This project aims to build an end-to-end **Text-to-Speech (TTS)** system using deep learning techniques. The goal is to convert natural language text into human-like speech using a neural network architecture. This project is a part of a larger study in **speech synthesis** and applies state-of-the-art models to produce high-quality audio from input text.

## 🧠 Model Architecture

The project is primarily based on the **Tacotron 2** architecture, which consists of two main components:

- **Sequence-to-Sequence (Seq2Seq) with Attention**: Converts input text into a mel-spectrogram.
- **Vocoder (WaveGlow / Griffin-Lim / HiFi-GAN)**: Converts the mel-spectrogram into an audio waveform.

This pipeline allows for a fully neural TTS system that generates realistic and intelligible speech.

## 📚 Technologies & Libraries

- Python 3.x
- TensorFlow / PyTorch
- NumPy, Pandas
- Librosa (audio processing)
- Matplotlib / TensorBoard (visualization)
- Jupyter Notebooks
  
