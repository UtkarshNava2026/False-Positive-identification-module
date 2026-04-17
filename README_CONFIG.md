# Detection GUI - Configuration Guide

## Overview
The Detection GUI now supports configuration management and asynchronous model loading to provide a smooth user experience without UI freezing.

## Configuration File (config.json)

The application uses `config.json` to store and load settings. Edit this file to customize behavior without changing code.

### Configuration Structure

```json
{
  "model": {
    "pth_path": "path/to/your/model.pth",
    "exp_path": "path/to/your/experiment.yaml",
    "classes_path": "path/to/classes.txt",
    "device": "cpu"
  },
  "video": {
    "fps": 33
  },
  "ui": {
    "window_width": 1200,
    "window_height": 800,
    "display_confidence": true,
    "box_thickness": 2,
    "text_size": 0.5
  },
  "export": {
    "default_format": "YOLO"
  }
}
```

### Configuration Options

#### Model Configuration
- **pth_path**: Path to your PyTorch model file (.pth)
- **exp_path**: Path to your experiment configuration file (.yaml, .json, or custom format)
- **classes_path**: Path to your class names file (one class name per line)
- **device**: Device to use for inference ('cpu' or 'cuda')

#### Video Configuration
- **fps**: Frames per second for video playback (default: 33)

#### UI Configuration
- **window_width**: Default window width in pixels (default: 1200)
- **window_height**: Default window height in pixels (default: 800)
- **display_confidence**: Show confidence scores on detections (default: true)
- **box_thickness**: Bounding box thickness (default: 2)
- **text_size**: Text size for labels (default: 0.5)

#### Export Configuration
- **default_format**: Default export format - 'YOLO', 'VOC', or 'COCO' (default: 'YOLO')

## Features

### Asynchronous Model Loading
- Model loading now happens in a separate thread
- UI remains responsive during model loading
- Progress indicator shows loading status
- Button is disabled until loading completes

### Automatic Model Loading
- If model paths are configured in `config.json`, the model will be automatically loaded when the application starts
- This saves time if you're working with the same model repeatedly

### Configuration Persistence
- When you load a model via the dialog, the paths are automatically saved to `config.json`
- Next time you run the application, it will try to load the same model automatically

## How to Use

### Method 1: Edit config.json Directly
1. Open `config.json` in a text editor
2. Fill in the paths to your model, experiment config, and classes file
3. Set other options as needed
4. Save the file
5. Run the application - it will automatically load the model

### Method 2: Use the GUI
1. Click "Load Model (.pth + exp + classes)"
2. Select your model files via the dialog boxes
3. The paths are saved to `config.json` automatically
4. Model loads asynchronously without freezing the UI
5. Next time you run the app, it will automatically load the same model

## Example config.json

```json
{
  "model": {
    "pth_path": "models/yolov5s.pth",
    "exp_path": "configs/exp_config.yaml",
    "classes_path": "data/classes.txt",
    "device": "cuda"
  },
  "video": {
    "fps": 30
  },
  "ui": {
    "window_width": 1400,
    "window_height": 900,
    "display_confidence": true,
    "box_thickness": 2,
    "text_size": 0.6
  },
  "export": {
    "default_format": "COCO"
  }
}
```

## Threading Architecture

### VideoThread
- Handles video/RTSP stream reading
- Runs inference in a loop
- Emits frames for display

### ModelLoaderThread
- Loads the detection model in background
- Prevents UI freezing during model initialization
- Emits signals when loading completes

### Main Thread (UI Thread)
- Remains responsive for user interactions
- Displays progress during model loading
- Updates display with new frames from video thread

## Benefits

1. **Smooth UI**: No freezing while loading large models
2. **Flexible Configuration**: Control everything via config.json
3. **Persistent Settings**: Model paths automatically saved
4. **Modular Design**: Easy to extend and modify
5. **Clean Code**: Separation of concerns with specialized threads
