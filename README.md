# Anime Wallpaper Extraction Bot

## Overview
This Python script automatically extracts the most visually appealing frames from an anime movie, creating high-quality wallpapers using advanced image processing techniques.

## Features
- Analyzes video frames based on multiple visual quality metrics
- Detects and prioritizes frames with:
  - Sharp, clear images
  - Vibrant color palettes
  - Balanced compositions
  - Character presence
- Filters out frames with text overlays
- Automatically saves top-rated wallpaper frames

## Requirements
- Python 3.7+
- OpenCV
- pytesseract
- Tesseract OCR

## Quick Start
1. Install dependencies:
   ```
   pip install opencv-python pytesseract
   ```

2. Run the script:
   ```python
   bot = AnimeWallpaperBot('path/to/your/anime_movie.mp4')
   bot.extract_wallpapers(num_wallpapers=10)
   ```

## How It Works
The bot uses computational metrics to evaluate frame "beauty":
- Sharpness assessment
- Color variance analysis
- Edge density calculation
- Symmetry evaluation
- Character detection

## Customization
Easily adjust:
- Number of wallpapers
- Feature weights
- Detection thresholds

## Limitations
- Results depend on specific anime art style
- Requires manual tuning for best results
