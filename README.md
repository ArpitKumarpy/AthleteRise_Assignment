# AthleteRise_Assignment
Asssignment submission: Cricket video analyser using computer vision

This is a beginner-friendly Python script that helps you analyze a cricketer's cover drive technique using video. It performs real-time pose estimation, calculates key biomechanical metrics, and provides a detailed feedback report.

---

## ✨ Features

* **Real-time Biomechanical Analysis:** Tracks and logs important aspects of the cover drive:
    * **Front Elbow Angle:** Measures the angle of the batting arm.
    * **Spine Lean:** Checks how much the batter's body is leaning.
    * **Head-over-Knee Alignment:** Ensures the head stays stable over the front knee.
    * **Front Foot Direction:** A basic estimate of the front foot's angle.
* **Live Visual Feedback:**
    * Draws the **pose skeleton** directly on the video.
    * Displays **real-time metric readouts** (e.g., "Elbow: 110 deg", "Spine Lean: 15 deg").
    * Gives **instant cues** like "Good elbow elevation" or "Head not over front knee."
* **Comprehensive Shot Evaluation:** After processing the video, it generates a JSON report with:
    * An **overall score** (1-10) for the shot.
    * Individual scores for **Footwork, Head Position, Swing Control, Balance, and Follow-through.**
    * **Detailed, scenario-based feedback** for each category to help improve technique.

---

## 🚀 Installation Guide

Follow these simple steps to get the script up and running on your computer.

### Step 1: Install Python < 3.11 (for mediapipe dependancy)

If you don't have Python installed, download it from the official website:
[https://www.python.org/downloads/](https://www.python.org/downloads/)
**Make sure to check "Add Python to PATH" during installation.**

### Step 2: Install Required Libraries

Open your terminal or command prompt and run the following command to install all the necessary libraries:

```bash
pip install opencv-python mediapipe numpy yt-dlp
```

## 🏃‍♀️ How to Run the Analysis
You can analyze videos directly from YouTube or use a video file saved on your computer.

**Option 1: Analyze a YouTube Video (Recommended)
To analyze a YouTube video, simply run the script with the --url argument followed by the video link:

```Bash

python cricket_drive_analysis_realtime.py --url https://youtube.com/shorts/vSX3IRxGnNY (or whatever url)
```


**Option 2: Analyze a Local Video File
If you have a video file (e.g., input_video.mp4) on your computer, use the --input argument:

```Bash

python cricket_analyzer.py --input path/to/your/video/my_cricket_shot.mp4
```
(Remember to replace path/to/your/video/my_cricket_shot.mp4 with the actual path to your video file.)

##Understanding the Output

After the script finishes processing, it will:

  Save an Annotated Video: A new video file named annotated_video.mp4 will be created in the output/ folder. This video will show the pose skeleton and live        
  metrics on top of the original footage.

  Generate an Evaluation Report: A JSON file named evaluation_report.json will also be saved in the output/ folder. This file contains:

  Your overall score.

  Breakdown scores for each technique aspect.

  Actionable feedback points to help you improve your cover drive!

**You can open the evaluation_report.json file with any text editor to read the detailed analysis.
  
##Feel free to experiment with different videos and see how your technique measures up! Happy analyzing! 🎉


