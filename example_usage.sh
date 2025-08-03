#!/bin/bash

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Basic usage - transcribe and split audiobook
echo -e "\n1. Basic usage (auto-detect chapters and split):"
echo "python audiobook_chapter_splitter.py audiobook.mp3"

# Use tiny model for faster processing on CPU
echo -e "\n2. Use tiny model (fastest, good for testing):"
echo "python audiobook_chapter_splitter.py audiobook.mp3 --model tiny"

# Only find chapters without splitting (useful for testing)
echo -e "\n3. Find chapters only (no splitting):"
echo "python audiobook_chapter_splitter.py audiobook.mp3 --no-split"

# Custom output directory
echo -e "\n4. Custom output directory:"
echo "python audiobook_chapter_splitter.py audiobook.mp3 --output-dir my_chapters"

# Process a sample (first 30 minutes) for testing
echo -e "\n5. Create a sample for testing (using ffmpeg):"
echo "ffmpeg -i audiobook.mp3 -t 1800 -c copy audiobook_sample.mp3"
echo "python audiobook_chapter_splitter.py audiobook_sample.mp3 --model tiny"