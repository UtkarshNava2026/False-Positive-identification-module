"""
Example script demonstrating object tracking and anomaly analysis.
Run this to test the tracking system on a video file.
"""
import sys
import cv2
import argparse
from pathlib import Path

# Add the parent directory to path to import fpa_agent
sys.path.insert(0, str(Path(__file__).parent))

from fpa_agent import DetectionModel, TrackingAnalyzer


def track_video(video_path: str, pth_path: str, exp_path: str, classes_path: str, 
                output_path: str = None, display: bool = True):
    """
    Track objects in a video and analyze for false positives/missed detections.
    
    Args:
        video_path: Path to input video file
        pth_path: Path to YOLOX model weights
        exp_path: Path to experiment config
        classes_path: Path to class names file
        output_path: Optional path to save annotated video
        display: Whether to display video during processing
    """
    
    print(f"Loading model from {pth_path}...")
    model = DetectionModel(
        pth_path=pth_path,
        exp_path=exp_path,
        classes_path=classes_path,
        device='cpu',
        enable_tracking=True
    )
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output video will be saved to: {output_path}")
    
    # Process video
    analyzer = TrackingAnalyzer()
    frame_count = 0
    
    print("\nProcessing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}/{total_frames} ({100*frame_count//total_frames}%)")
        
        # Run detection with tracking
        detections = model.predict(frame)
        
        # Draw detections with track IDs
        display_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            label = det['label']
            track_id = det.get('track_id', '?')
            
            # Format label
            display_label = f"ID:{track_id} {label} {conf:.2f}"
            
            # Draw box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label_size = cv2.getTextSize(display_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 4), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(display_frame, display_label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save frame if output specified
        if out:
            out.write(display_frame)
        
        # Display frame
        if display:
            cv2.imshow('Object Tracking', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if out:
        out.release()
    if display:
        cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames")
    
    # Analyze results
    print("\nAnalyzing tracking results...")
    anomalies = model.get_anomalies()
    analysis = analyzer.analyze_detections(anomalies)
    
    # Print report
    analyzer.print_analysis(analysis)
    
    # Export report
    report_path = Path(video_path).stem + "_tracking_report.json"
    analyzer.export_report(report_path)
    print(f"\nReport saved to: {report_path}")
    
    # Print top anomalies
    print("\n" + "="*60)
    print("TOP FALSE POSITIVES")
    print("="*60)
    for fp in analysis['false_positives'].get('person', [])[:5]:
        print(f"  Track {fp['track_id']}: avg_conf={fp['avg_confidence']:.3f}, "
              f"std={fp['confidence_std']:.3f}, anomaly_score={fp['anomaly_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Track objects in video and analyze for false positives")
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--model', default='mondelich.pth', help='Path to YOLOX model weights')
    parser.add_argument('--exp', default='exp.py', help='Path to experiment config')
    parser.add_argument('--classes', default='class.txt', help='Path to class names file')
    parser.add_argument('--output', default=None, help='Path to save annotated video')
    parser.add_argument('--no-display', action='store_true', help='Do not display video during processing')
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not Path(args.classes).exists():
        print(f"Error: Classes file not found: {args.classes}")
        return
    
    track_video(
        video_path=args.video,
        pth_path=args.model,
        exp_path=args.exp,
        classes_path=args.classes,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
