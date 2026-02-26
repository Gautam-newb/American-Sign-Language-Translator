# KOMMUNIK8 - Sign Language Recognition System

KOMMUNIK8 is a real-time American Sign Language (ASL) recognition system that uses computer vision and machine learning to translate hand gestures into text. The system provides a web-based interface for live gesture recognition, allowing users to communicate through sign language with real-time transcription.

## Project Overview

KOMMUNIK8 leverages MediaPipe for hand tracking and TensorFlow Lite models for gesture classification. The system can recognize ASL letters (A-Z) and various hand gestures, providing a seamless communication interface for sign language users.

## Features

- **Real-time Hand Gesture Recognition**: Uses MediaPipe to track hand landmarks in real-time
- **ASL Letter Recognition**: Recognizes all 26 ASL letters (A-Z)
- **Gesture Stabilization**: Advanced filtering system to reduce false positives and stabilize detections
- **Web-based Interface**: Modern, responsive web UI with live video feed
- **Real-time Transcription**: Displays recognized letters in real-time via WebSocket
- **Text-to-Speech**: Built-in speech synthesis for output
- **Action History**: Tracks and replays user actions
- **Model Training Tools**: Scripts for training and improving gesture recognition models

## Tech Stack

### Backend
- **Python 3.8+**
- **Flask 3.0.0**: Web framework for serving the application
- **Flask-SocketIO 5.3.6**: Real-time bidirectional communication
- **OpenCV 4.5.0+**: Computer vision and video processing
- **MediaPipe 0.8.9+**: Hand tracking and landmark detection
- **TensorFlow 2.15.0**: Machine learning framework
- **NumPy 1.19.0+**: Numerical computations

### Frontend
- **HTML5/CSS3**: Modern web interface
- **JavaScript**: Client-side interactivity
- **Socket.IO Client**: Real-time communication with backend
- **Web Speech API**: Text-to-speech functionality

### Machine Learning
- **TensorFlow Lite**: Optimized models for inference
- **Scikit-learn**: Data preprocessing and model evaluation

## Folder Structure

```
Project Ult/
├── app.py                          # Main Flask application with gesture recognition
├── server.py                       # Alternative server implementation
├── frontend.html                   # Web interface
├── requirements.txt                # Python dependencies
├── sign_language_filter.py        # Gesture stabilization filter
├── train_model.py                 # Basic model training script
├── train_model_improved.py        # Enhanced model training with preprocessing
├── merge_keypoint_data.py         # Utility to merge training datasets
├── keypoint_classification.ipynb  # Jupyter notebook for model development
├── point_history_classification.ipynb  # Notebook for gesture history classification
│
├── model/                          # Machine learning models
│   ├── __init__.py
│   ├── keypoint_classifier/       # Hand gesture classifier
│   │   ├── keypoint_classifier.py
│   │   ├── keypoint_classifier.tflite
│   │   ├── keypoint_classifier.keras
│   │   ├── keypoint_classifier_label.csv
│   │   ├── keypoint.csv           # Training data
│   │   └── keypoint_scaler.pkl    # Data scaler
│   └── point_history_classifier/  # Gesture history classifier
│       ├── point_history_classifier.py
│       ├── point_history_classifier.tflite
│       ├── point_history_classifier_label.csv
│       └── point_history.csv
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   └── cvfpscalc.py               # FPS calculation utility
│
└── static/                         # Static assets
    └── images/
        └── Solid logo for KOMMUNIK8.jpg
```

## Setup & Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/camera device
- pip (Python package manager)

### Installation Steps

1. **Clone the repository** (or navigate to project directory):
   ```bash
   cd "C:\Project Ult"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import cv2, mediapipe, tensorflow, flask; print('All dependencies installed!')"
   ```

## How to Run

### Option 1: Using app.py (Recommended)

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Option 2: Using server.py

```bash
python server.py
```

### Access the Web Interface

1. Open your web browser
2. Navigate to `http://localhost:5000`
3. Click "Start webcam" to begin gesture recognition
4. Perform ASL gestures in front of your camera
5. Recognized letters will appear in the "Current word" field

## API Endpoints

### Web Routes

- `GET /` - Serves the main web interface (`frontend.html`)
- `GET /video_feed` - Streams video feed with gesture recognition overlay
- `POST /start_camera` - Initializes and starts the camera
- `POST /stop_camera` - Stops and releases the camera

### WebSocket Events

- `transcript` - Emitted when a gesture is recognized
  ```json
  {
    "type": "gesture",
    "text": "A"
  }
  ```

## Usage Examples

### Basic Usage

1. Start the server: `python app.py`
2. Open browser to `http://localhost:5000`
3. Click "Start webcam"
4. Perform ASL letters in front of camera
5. Letters appear in real-time in the interface

### Training a New Model

1. Collect training data using the data collection mode in `app.py` (press 'k' to toggle)
2. Save samples by pressing 's' while in data collection mode
3. Train the model:
   ```bash
   python train_model_improved.py
   ```
4. The trained model will be saved to `model/keypoint_classifier/keypoint_classifier.keras`

### Merging Training Datasets

```bash
python merge_keypoint_data.py
```

## Configuration

### Camera Settings

Modify camera settings in `app.py`:
```python
cap_device = 0  # Camera device index
cap_width = 960
cap_height = 540
```

### Gesture Filter Parameters

Adjust stabilization in `app.py`:
```python
sign_filter = SignLanguageFilter(
    window_size=30,      # Number of frames to average
    threshold=0.8,       # Confidence threshold
    cooldown_period=1.2  # Seconds between detections
)
```

## Model Training

### Training Data Format

Training data is stored in CSV format:
- First column: Class label (integer)
- Remaining 42 columns: Normalized landmark coordinates (21 points × 2 coordinates)

### Training Process

1. **Data Collection**: Use the built-in data collection mode
2. **Data Preprocessing**: Automatic normalization and scaling
3. **Model Architecture**: Deep neural network with dropout layers
4. **Training**: Uses early stopping and learning rate reduction
5. **Export**: Converts to TensorFlow Lite for efficient inference

## Contribution Guidelines

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Specify your license here - MIT, Apache 2.0, etc. Please update this section with your preferred license]

## Acknowledgments

- MediaPipe team for hand tracking technology
- TensorFlow team for machine learning framework
- OpenCV community for computer vision tools

## Contact

[Add your contact information or project maintainer details]

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

