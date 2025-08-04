# Audiobook Chapter Splitter - Project Context

## Overview

This project uses OpenAI's Whisper to automatically transcribe audiobooks and split them into chapters based on detected chapter markers.

## Key Features

- Automatic chapter detection from audio transcription
- CPU-friendly with optional GPU acceleration
- Lossless audio splitting at chapter boundaries
- Multiple Whisper model sizes for speed/accuracy tradeoff

## Project Structure

```
sounds-chaptersplit/
├── audiobook_chapter_splitter.py  # Main script
├── requirements.txt               # Python dependencies
├── Makefile                       # Build automation
├── README.md                      # User documentation
└── example_usage.sh              # Usage examples
```

## Quick Start

```bash
# Install dependencies
make install

# Test with sample
export AUDIOBOOK=/path/to/audiobook.mp3
make test

# Split full audiobook
make split-fast  # Fast mode with tiny model
make split       # Better accuracy with base model
```

## Technical Details

- Uses Whisper's word-level timestamps for precise chapter detection
- Processes entire audio file once for better context and accuracy
- Splits audio using ffmpeg without re-encoding (preserves quality)
- Supports various chapter patterns: "Chapter One", "Part 1", etc.

## Model Recommendations

- **CPU only**: Use `tiny` or `base` models
- **GPU available**: Can use `small`, `medium`, or `large` for better accuracy
- **Testing**: Always test with `--no-split` flag first to verify chapter detection

## Common Tasks

- Modify chapter patterns: Edit `chapter_patterns` in `audiobook_chapter_splitter.py`
- Process multiple books: Use bash loop with the Python script
- Custom output directory: Use `--output-dir` flag

## Dependencies

- Python 3.8+
- openai-whisper
- ffmpeg (system dependency)
- torch (CPU or CUDA version)
