# Architecture Documentation

This document describes the architecture and design of the KOMMUNIK8 sign language recognition system.

## System Overview

KOMMUNIK8 is a real-time sign language recognition system that processes video input from a webcam, detects hand gestures using computer vision, classifies them using machine learning models, and displays the results through a web interface.

## Architecture Diagram

```
┌─────────────────┐
│   Web Browser   │
│  (frontend.html)│
└────────┬────────┘
         │ HTTP/WebSocket
         │
┌────────▼────────────────────────┐
│      Flask Application          │
│  ┌──────────────────────────┐  │
│  │   Flask Routes           │  │
│  │   - / (index)            │  │
│  │   - /video_feed          │  │
│  │   - /start_camera        │  │
│  │   - /stop_camera         │  │
│  └──────────────────────────┘  │
│  ┌──────────────────────────┐  │
│  │   SocketIO Server        │  │
│  │   - Real-time events    │  │
│  └──────────────────────────┘  │
└────────┬───────────────────────┘
         │
┌────────▼────────────────────────┐
│   Video Processing Pipeline     │
│  ┌──────────────────────────┐  │
│  │   OpenCV Capture         │  │
│  │   - Frame capture        │  │
│  │   - Image preprocessing  │  │
│  └───────────┬──────────────┘  │
│              │                  │
│  ┌───────────▼──────────────┐  │
│  │   MediaPipe Hands        │  │
│  │   - Hand detection       │  │
│  │   - Landmark extraction  │  │
│  └───────────┬──────────────┘  │
│              │                  │
│  ┌───────────▼──────────────┐  │
│  │   Feature Extraction     │  │
│  │   - Normalization        │  │
│  │   - Coordinate transform │  │
│  └───────────┬──────────────┘  │
└──────────────┼──────────────────┘
               │
┌──────────────▼──────────────────┐
│   Machine Learning Pipeline     │
│  ┌──────────────────────────┐  │
│  │   KeyPoint Classifier    │  │
│  │   (TensorFlow Lite)      │  │
│  └───────────┬──────────────┘  │
│              │                  │
│  ┌───────────▼──────────────┐  │
│  │   Sign Language Filter   │  │
│  │   - Stabilization        │  │
│  │   - Noise reduction      │  │
│  └───────────┬──────────────┘  │
│              │                  │
│  ┌───────────▼──────────────┐  │
│  │   Point History Classifier│  │
│  │   (for dynamic gestures)  │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer

**File**: `frontend.html`

**Responsibilities**:
- Display video feed from backend
- Show recognized gestures in real-time
- Provide user controls (start/stop camera, speak, etc.)
- Maintain action history
- Handle WebSocket communication

**Technologies**:
- HTML5/CSS3 for UI
- JavaScript for interactivity
- Socket.IO client for real-time updates
- Web Speech API for text-to-speech

### 2. Backend Layer

#### 2.1 Flask Application

**Files**: `app.py`, `server.py`

**Responsibilities**:
- Serve web interface
- Handle HTTP routes
- Manage WebSocket connections
- Coordinate video processing pipeline
- Emit recognition results

**Key Routes**:
- `GET /`: Serve main HTML page
- `GET /video_feed`: Stream processed video frames
- `POST /start_camera`: Initialize camera
- `POST /stop_camera`: Release camera

#### 2.2 Video Processing

**Components**:
- **OpenCV**: Frame capture and image processing
- **MediaPipe Hands**: Hand detection and landmark extraction

**Process Flow**:
1. Capture frame from camera
2. Convert BGR to RGB for MediaPipe
3. Detect hands and extract 21 landmarks per hand
4. Calculate bounding box and normalize coordinates
5. Draw landmarks and annotations on frame
6. Encode frame as JPEG for streaming

### 3. Machine Learning Layer

#### 3.1 KeyPoint Classifier

**File**: `model/keypoint_classifier/keypoint_classifier.py`

**Purpose**: Classify static hand gestures (ASL letters)

**Input**: 
- 42 features (21 landmarks × 2 coordinates)
- Normalized and preprocessed

**Output**: 
- Class ID (0-29, representing different gestures)
- Corresponding label from CSV file

**Model Format**: TensorFlow Lite (.tflite)

**Architecture**:
- Input layer (42 features)
- Dense layers with ReLU activation
- Dropout for regularization
- Softmax output layer

#### 3.2 Point History Classifier

**File**: `model/point_history_classifier/point_history_classifier.py`

**Purpose**: Classify dynamic gestures based on movement history

**Input**: 
- Sequence of point coordinates over time
- 32 features (16 history points × 2 coordinates)

**Output**: 
- Dynamic gesture class ID

**Use Case**: Recognizing gestures that involve movement (e.g., pointing, waving)

#### 3.3 Sign Language Filter

**File**: `sign_language_filter.py`

**Purpose**: Stabilize gesture recognition and reduce false positives

**Algorithm**:
1. Maintains a sliding window of probability distributions
2. Averages probabilities over the window
3. Applies confidence threshold
4. Implements cooldown period between detections

**Parameters**:
- `window_size`: Number of frames to average (default: 10-30)
- `threshold`: Minimum confidence (default: 0.6-0.8)
- `cooldown_period`: Minimum time between detections (default: 0.5-1.2s)

### 4. Utility Modules

#### 4.1 FPS Calculator

**File**: `utils/cvfpscalc.py`

**Purpose**: Calculate and display frames per second

**Implementation**: Uses OpenCV's tick counter for accurate timing

## Data Flow

### Recognition Pipeline

1. **Frame Capture**: OpenCV captures frame from camera
2. **Hand Detection**: MediaPipe detects hands and extracts landmarks
3. **Preprocessing**: 
   - Convert to relative coordinates
   - Normalize to [0, 1] range
   - Flatten to feature vector
4. **Classification**: 
   - KeyPoint classifier predicts gesture
   - Filter stabilizes prediction
5. **Post-processing**:
   - Apply confidence threshold
   - Check cooldown period
   - Update output if conditions met
6. **Output**: 
   - Emit via WebSocket
   - Draw on video frame
   - Display in UI

### Training Pipeline

1. **Data Collection**: 
   - Use data collection mode in app.py
   - Save landmark data to CSV
2. **Preprocessing**:
   - Load CSV data
   - Normalize features
   - Split train/test sets
3. **Training**:
   - Build neural network
   - Train with early stopping
   - Validate on test set
4. **Export**:
   - Save as Keras model
   - Convert to TensorFlow Lite
   - Update label files

## Design Patterns

### 1. Generator Pattern

Video frames are streamed using Python generators:
```python
def generate_frames():
    while True:
        # Process frame
        yield frame_bytes
```

### 2. Singleton Pattern

Camera instance is managed as a global singleton to prevent multiple initializations.

### 3. Observer Pattern

WebSocket events notify frontend of recognition results in real-time.

## Performance Considerations

### Optimization Strategies

1. **TensorFlow Lite**: Lightweight models for fast inference
2. **Frame Skipping**: Process every Nth frame if needed
3. **Resolution**: Lower resolution for faster processing
4. **Threading**: Separate threads for video capture and processing

### Bottlenecks

1. **MediaPipe Processing**: Most computationally expensive
2. **Model Inference**: TensorFlow Lite is optimized but still requires CPU/GPU
3. **Network Latency**: WebSocket communication adds minimal delay

## Security Considerations

1. **Camera Access**: Requires user permission
2. **CORS**: Configured for local development (adjust for production)
3. **Input Validation**: Validate all user inputs
4. **Error Handling**: Graceful degradation on errors

## Future Enhancements

### Potential Improvements

1. **GPU Acceleration**: Use GPU for MediaPipe and model inference
2. **Multi-hand Support**: Process multiple hands simultaneously
3. **Gesture Sequences**: Recognize words and phrases
4. **Cloud Deployment**: Deploy to cloud for remote access
5. **Mobile App**: Native mobile application
6. **Offline Mode**: Work without internet connection
7. **Model Retraining**: Online learning from user corrections

## Dependencies

### Core Dependencies

- **Flask**: Web framework
- **Flask-SocketIO**: WebSocket support
- **OpenCV**: Computer vision
- **MediaPipe**: Hand tracking
- **TensorFlow**: Machine learning
- **NumPy**: Numerical operations

### Development Dependencies

- **Jupyter**: Notebook development
- **Scikit-learn**: Model evaluation
- **Joblib**: Model serialization

## File Organization

```
app.py                    # Main application entry point
server.py                 # Alternative server implementation
frontend.html            # Web interface
sign_language_filter.py  # Gesture stabilization
train_model*.py          # Model training scripts
model/                   # ML models and data
utils/                   # Utility functions
static/                  # Static assets
```

## Configuration

Key configuration points:
- Camera device index
- Frame resolution
- Model paths
- Filter parameters
- Server port

All configurable in code or via environment variables (future enhancement).

