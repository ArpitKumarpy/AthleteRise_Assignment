import cv2
import numpy as np
import mediapipe as mp
import json
import os
import math
import yt_dlp
import argparse

def download_video(url):
    print(f"Downloading video from: {url}")
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': 'input_video.%(ext)s',
        'overwrites': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def calculate_angle(p1, p2, p3):
#angle between 3 points
    try:
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    except:
        return 0.0

def calculate_spine_lean(hip, shoulder):
    #spine lean angle from vertical
    try:
        dy = shoulder[1] - hip[1]
        dx = shoulder[0] - hip[0]
        return np.degrees(np.arctan2(abs(dx), abs(dy)))
    except:
        return 0.0

def process_and_analyze_video(video_path, output_dir="output"):

    print(f"Processing video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = os.path.join(output_dir, "annotated_video.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    metrics_history = []
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        annotated_frame = frame.copy()
        
        metrics = {'frame_number': frame_number, 'timestamp': frame_number / fps}

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE] # Needed for foot direction

            
            def to_px(landmark):

                if landmark.visibility < 0.5:
                    return (0, 0) # Return a default or handle as unseen
                return (int(landmark.x * width), int(landmark.y * height))
            
            # 1. Biomechanical Metrics
            metrics['front_elbow_angle'] = calculate_angle(to_px(r_shoulder), to_px(r_elbow), to_px(r_wrist))
            metrics['spine_lean'] = calculate_spine_lean(to_px(r_hip), to_px(r_shoulder))
            metrics['head_knee_alignment'] = abs(to_px(nose)[0] - to_px(r_knee)[0])
            
            # This is a very basic approximation for foot direction
            current_ankle_px = to_px(r_ankle)
            if frame_number > 0 and 'last_ankle' in metrics_history[-1] and metrics_history[-1]['last_ankle'] != (0,0):
                last_ankle_px = metrics_history[-1]['last_ankle']
                # Angle relative to a horizontal line
                metrics['front_foot_direction'] = calculate_angle(current_ankle_px, last_ankle_px, (last_ankle_px[0] + 100, last_ankle_px[1]))
            else:
                metrics['front_foot_direction'] = 0.0 # Default if no previous frame or ankle not visible
            metrics['last_ankle'] = current_ankle_px # Store current ankle for next frame's calculation


            #OVerlay
            text_y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_color = (255, 255, 255) # White color for text

            cv2.putText(annotated_frame, f"Elbow: {metrics['front_elbow_angle']:.1f} deg", (20, text_y_offset), font, font_scale, text_color, thickness)
            text_y_offset += 30
            cv2.putText(annotated_frame, f"Spine Lean: {metrics['spine_lean']:.1f} deg", (20, text_y_offset), font, font_scale, text_color, thickness)
            text_y_offset += 30
            cv2.putText(annotated_frame, f"Head-Knee: {metrics['head_knee_alignment']:.1f}px", (20, text_y_offset), font, font_scale, text_color, thickness)
            text_y_offset += 30
            cv2.putText(annotated_frame, f"Foot Dir: {metrics['front_foot_direction']:.1f} deg", (20, text_y_offset), font, font_scale, text_color, thickness)
            
            # Short feedback cues
            feedback_y = text_y_offset + 30
            # Elbow feedback
            if 90 <= metrics['front_elbow_angle'] <= 140:
                cv2.putText(annotated_frame, "Good elbow elevation", (20, feedback_y), font, font_scale - 0.1, (0, 255, 0), thickness - 1)
            else:
                cv2.putText(annotated_frame, "Bad, Adjust elbow angle", (20, feedback_y), font, font_scale - 0.1, (0, 0, 255), thickness - 1)
            
            feedback_y += 25
            # Head-knee feedbac
            if metrics['head_knee_alignment'] < 50: 
                cv2.putText(annotated_frame, "Head over front knee", (20, feedback_y), font, font_scale - 0.1, (0, 255, 0), thickness - 1)
            else:
                cv2.putText(annotated_frame, "Bad, Head not over front knee", (20, feedback_y), font, font_scale - 0.1, (0, 0, 255), thickness - 1)

        out.write(annotated_frame)
        metrics_history.append(metrics)
        frame_number += 1
    
    cap.release()
    out.release()
    pose.close()
    
    print(f"Annotated video saved to: {output_video_path}")
    return metrics_history, output_video_path

def evaluate_and_save_report(metrics_history, output_dir="output"):
    #Evaluates the shot and saves a simple JSON report
    if not metrics_history:
        print("No metrics data to evaluate.")
        return None
    
    # Filter out frames where pose detection might have failed or landmarks were not visible
    valid_metrics = [m for m in metrics_history if 'front_elbow_angle' in m and m['front_elbow_angle'] > 0]
    if not valid_metrics:
        print("No valid pose data found for evaluation.")
        return None
        
    avg_elbow = np.mean([m['front_elbow_angle'] for m in valid_metrics])
    avg_spine_lean = np.mean([m['spine_lean'] for m in valid_metrics])
    avg_head_knee = np.mean([m['head_knee_alignment'] for m in valid_metrics])
    avg_foot_direction = np.mean([m['front_foot_direction'] for m in valid_metrics if m['front_foot_direction'] != 0.0])
    
    #(scores 1-10)
    scores = {}
    feedback = {}

    #Footwork Score & Feedback
    # A smaller head-knee distance is better, closer to 0. A foot direction around 90 degrees (straight) might be ideal.
    footwork_score = max(1, min(10, 10 - avg_head_knee / 8 - abs(avg_foot_direction - 90) / 10))
    scores["footwork"] = round(footwork_score, 1)
    feedback["footwork"] = []
    if footwork_score >= 7.5:
        feedback["footwork"].append("Excellent foot placement, allowing for strong drives.")
        feedback["footwork"].append("Maintain this aggressive stride towards the ball.")
    elif footwork_score >= 5.0:
        feedback["footwork"].append("Work on more consistent foot placement.")
        feedback["footwork"].append("Focus on getting closer to the line of the ball to maximize power.")
    else:
        feedback["footwork"].append("Significant improvement needed in foot positioning.")
        feedback["footwork"].append("Practice moving your front foot more decisively towards the ball's line.")
        feedback["footwork"].append("Ensure your foot lands stably to generate power.")

    # Head Position Score & Feedback
    head_score = max(1, min(10, 10 - avg_head_knee / 8))
    scores["head_position"] = round(head_score, 1)
    feedback["head_position"] = []
    if head_score >= 8.0:
        feedback["head_position"].append("Excellent head stillness, keeping your eyes locked on the ball.")
        feedback["head_position"].append("This stable base is crucial for precision.")
    elif head_score >= 5.0:
        feedback["head_position"].append("Aim to keep your head more still and aligned over your front knee.")
        feedback["head_position"].append("Ensure your head is steady throughout the shot to watch the ball closely.")
    else:
        feedback["head_position"].append("Your head position needs significant attention.")
        feedback["head_position"].append("Focus on keeping your head still and directly above your front knee to maintain balance and vision.")

    # Swing Control Score & Feedback
    # Ideal range 90-140 degrees, target 115 mid-point
    swing_score = max(1, min(10, 10 - abs(avg_elbow - 115) / 5))
    scores["swing_control"] = round(swing_score, 1)
    feedback["swing_control"] = []
    if swing_score >= 7.5:
        feedback["swing_control"].append("Great swing control and a powerful arc.")
        feedback["swing_control"].append("Your elbow elevation is consistent, enabling good power transfer.")
    elif swing_score >= 5.0:
        feedback["swing_control"].append("Work on maintaining a more consistent swing plane.")
        feedback["swing_control"].append("Ensure your front elbow stays high and you swing through the line of the ball.")
    else:
        feedback["swing_control"].append("Review your swing path for consistency and power generation.")
        feedback["swing_control"].append("Focus on a higher front elbow and a fuller swing through the ball.")

    # Balance Score & Feedback
    balance_score = max(1, min(10, 10 - avg_spine_lean / 2)) # max_lean = 20, so /2 ensures range
    scores["balance"] = round(balance_score, 1)
    feedback["balance"] = []
    if balance_score >= 7.5:
        feedback["balance"].append("Excellent balance maintained throughout the shot.")
        feedback["balance"].append("This stability allows for maximum power and control.")
    elif balance_score >= 5.0:
        feedback["balance"].append("Work on maintaining better balance, especially after contact.")
        feedback["balance"].append("Keep your weight on the balls of your feet to stay agile.")
    else:
        feedback["balance"].append("Significant balance issues observed.")
        feedback["balance"].append("Practice staying upright and stable through your shot to avoid falling over.")
        feedback["balance"].append("Strengthen your core and leg muscles to improve stability.")

    #Follow-through Score & Feedback
    follow_through_score = 5 # Base score
    if valid_metrics:
        final_frame_metrics = valid_metrics[-1]
        if final_frame_metrics['front_elbow_angle'] > 90: # Elbow up during follow-through
            follow_through_score += 2
        if final_frame_metrics['spine_lean'] < 15: # Good posture during follow-through
            follow_through_score += 1
    scores["follow_through"] = round(min(10, follow_through_score), 1)
    feedback["follow_through"] = []
    if follow_through_score >= 7.5:
        feedback["follow_through"].append("Strong and complete follow-through.")
        feedback["follow_through"].append("You are extending fully towards your target.")
    elif follow_through_score >= 5.0:
        feedback["follow_through"].append("Aim for a more complete follow-through.")
        feedback["follow_through"].append("Ensure full extension of the arms after hitting the ball.")
    else:
        feedback["follow_through"].append("Your follow-through needs considerable work.")
        feedback["follow_through"].append("Focus on extending your hands and bat fully towards where you want the ball to go.")

    overall_score = round(np.mean(list(scores.values())), 1)

    report = {
        "overall_score": overall_score,
        "scores": scores,
        "feedback": feedback, 
        "metrics_summary": {
            "avg_elbow_angle": round(avg_elbow, 1),
            "avg_spine_lean": round(avg_spine_lean, 1),
            "avg_head_knee_distance": round(avg_head_knee, 1),
            "avg_front_foot_direction": round(avg_foot_direction, 1),
            "total_frames_analyzed": len(valid_metrics)
        }
    }
    
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Evaluation report saved to: {report_path}")
    return report_path

def main():
    
    parser = argparse.ArgumentParser(description='Simplified Cricket Cover Drive Analysis')
    parser.add_argument('--url', default='https://youtube.com/shorts/vSX3IRxGnNY',
                        help='YouTube video URL to analyze')
    parser.add_argument('--input', help='Local video file path (alternative to URL)')
    
    args = parser.parse_args()
    
    if args.input:
        video_path = args.input
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return
    else:
        video_path = download_video(args.url)
        if not video_path:
            return
    
    metrics, annotated_video = process_and_analyze_video(video_path)
    if metrics:
        report_file = evaluate_and_save_report(metrics)
        print(f"\nAnalysis Complete! Annotated video at {annotated_video} and report at {report_file}.")

if __name__ == "__main__":
    main()
