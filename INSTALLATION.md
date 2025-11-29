# Installation Guide

This guide provides detailed installation instructions for KOMMUNIK8 on different operating systems.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Camera**: Webcam or USB camera
- **Storage**: 500MB free space

### Recommended Requirements
- **OS**: Latest version of Windows/macOS/Linux
- **Python**: 3.9 or higher
- **RAM**: 8GB or more
- **Camera**: HD webcam (720p or higher)
- **GPU**: Optional, but recommended for model training

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. Check "Add Python to PATH" during installation
4. Verify installation:
   ```bash
   python --version
   ```

**macOS:**
```bash
# Using Homebrew
brew install python3

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Step 2: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd "Project Ult"

# Or download and extract the ZIP file
```

#### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 5: Verify Installation

```bash
python -c "import cv2, mediapipe, tensorflow, flask; print('Installation successful!')"
```

### Method 2: Using Conda

```bash
# Create conda environment
conda create -n kommunik8 python=3.9
conda activate kommunik8

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

#### Issue: `pip` command not found

**Solution:**
- Windows: Reinstall Python and check "Add to PATH"
- macOS/Linux: Install pip: `python3 -m ensurepip --upgrade`

#### Issue: OpenCV installation fails

**Solution:**
```bash
# Try installing with specific version
pip install opencv-python==4.5.0

# Or on Linux, install system dependencies first
sudo apt-get install libopencv-dev python3-opencv
```

#### Issue: MediaPipe installation fails

**Solution:**
```bash
# Update pip first
pip install --upgrade pip

# Install MediaPipe
pip install mediapipe
```

#### Issue: TensorFlow installation fails

**Solution:**
```bash
# For CPU-only version
pip install tensorflow==2.15.0

# For GPU support (requires CUDA)
pip install tensorflow-gpu==2.15.0
```

#### Issue: Camera not detected

**Solution:**
- Check camera permissions in system settings
- Verify camera is not being used by another application
- Try changing camera device index in code (default is 0)

#### Issue: Port 5000 already in use

**Solution:**
```bash
# Change port in app.py or server.py
socketio.run(app, debug=True, port=5001)  # Use different port
```

### Platform-Specific Notes

#### Windows
- May need to install Visual C++ Redistributable
- Camera permissions managed through Windows Settings
- Use PowerShell or Command Prompt (not Git Bash for some commands)

#### macOS
- May need to allow camera access in System Preferences > Security & Privacy
- If using Homebrew Python, ensure it's in PATH

#### Linux
- May need to install additional packages:
  ```bash
  sudo apt-get install python3-dev python3-pip libopencv-dev
  ```
- Camera access may require user to be in `video` group:
  ```bash
  sudo usermod -a -G video $USER
  ```

## Post-Installation

### First Run

1. Start the application:
   ```bash
   python app.py
   ```

2. Open browser to `http://localhost:5000`

3. Click "Start webcam" and allow camera access

4. Test gesture recognition by showing ASL letters

### Verifying Components

Check that all components work:

```python
# Test script
import cv2
import mediapipe as mp
import tensorflow as tf
from flask import Flask

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)
print("TensorFlow version:", tf.__version__)
print("Flask version:", Flask.__version__)

# Test camera
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera: OK")
    cap.release()
else:
    print("Camera: FAILED")
```

## Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

## Uninstallation

To remove KOMMUNIK8:

1. Deactivate virtual environment:
   ```bash
   deactivate
   ```

2. Delete the project directory

3. (Optional) Remove Python if not needed elsewhere

## Getting Help

If you encounter issues not covered here:

1. Check the [README.md](README.md) for common solutions
2. Review [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
3. Open an issue on GitHub with:
   - Your OS and Python version
   - Error messages
   - Steps to reproduce

## Next Steps

After successful installation:

1. Read the [README.md](README.md) for usage instructions
2. Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
3. Review [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute

Happy signing! 🎉

