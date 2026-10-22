@echo off
cd /d "%~dp0.."
echo ======================================================================
echo Running MOT Benchmark Evaluation...
echo ======================================================================
python evaluate_benchmarks.py
echo.
echo ======================================================================
echo Running Component Ablation Study...
echo ======================================================================
python scripts/run_ablation_study.py
echo.
echo ======================================================================
echo Running Crowd Density Stress Analysis...
echo ======================================================================
python scripts/run_stress_analysis.py
pause
