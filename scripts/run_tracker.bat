@echo off
cd /d "%~dp0.."
echo ======================================================================
echo Identity-Preserving Multi-Object Tracking Pipeline
echo ======================================================================
echo Running tracking on real crowded pedestrian sequence...
python run_tracker.py --input data/video_sequences/real_people_walking.mp4 --output outputs/tracked_videos/real_crowd_walking_tracked.mp4
echo.
echo Tracking completed successfully! Output saved to:
echo outputs/tracked_videos/real_crowd_walking_tracked.mp4
pause
