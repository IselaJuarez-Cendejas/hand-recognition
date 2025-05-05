# Real-Time Hand Recognition & Landmark Detection System

## Overview
A real-time computer vision system for hand landmark detection and orientation classification, built with OpenCV, MediaPipe, and NumPy.  
Optimized for low-latency, memory-efficient execution.

## Key Features
- 🔎 **Landmark Detection**: 21-point hand tracking using MediaPipe.
- 🔄 **Hand Orientation**: Front/back, wrist angle, direction, and left/right classification.
- ⚡ **Real-Time Performance**: Low-latency processing for constrained hardware.
- 🤝 **Scalable Base**: Designed for gesture recognition and biometric systems.

## How It Works
1. Captures webcam feed.
2. Detects hand landmarks.
3. Analyzes geometry and classifies orientation.
4. Displays annotated video in real-time.

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy

## Use Cases
- 👮‍♂️ Gesture-based security authentication
- 🕹️ AR/VR hand tracking
- 🤖 Human-Computer Interaction (HCI)

## Run the System
```bash
python hand.py

