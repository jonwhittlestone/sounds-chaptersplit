# sounds-chaptersplit

> Audiobook Chapter Splitter

Automatically split audiobooks into chapters using OpenAI's Whisper speech recognition. This tool transcribes your audiobook, detects chapter markers, and splits the audio file at the correct timestamps.

## Features

- 🎯 Automatic chapter detection (Chapter One, Chapter Two, Part 1, etc.)
- 🚀 CPU-friendly with optional GPU acceleration
- 📝 Full transcription with timestamps
- ✂️ Lossless audio splitting (no quality loss)
- 🔍 Multiple model sizes for speed vs accuracy tradeoff

## Requirements

- Python 3.8+
- ffmpeg (for audio processing)
- 4-8GB RAM (depending on model size)
- Optional: NVIDIA GPU for faster processing

## Installation

### Quick Install with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or download this repository
cd sounds-chaptersplit

# Create virtual environment and install dependencies
make uv-install
```

### Alternative: Standard virtualenv

```bash
# Clone or download this repository
cd sounds-chaptersplit

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
```

### Manual Installation

```bash
# Using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# OR using standard pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install ffmpeg (required for all methods)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

## Usage

### Quick Start with Makefile

```bash
# Set your audiobook path
export AUDIOBOOK=/path/to/your/audiobook.mp3

# Test with a 30-minute sample first
make test

# Split the full audiobook
make split

# Use tiny model for faster processing (CPU-friendly)
make split-fast
```

### Direct Python Usage

```bash
# Basic usage (auto-detect and split)
python audiobook_chapter_splitter.py audiobook.mp3

# Fast mode with tiny model (best for CPU)
python audiobook_chapter_splitter.py audiobook.mp3 --model tiny

# Test without splitting
python audiobook_chapter_splitter.py audiobook.mp3 --no-split

# Custom output directory
python audiobook_chapter_splitter.py audiobook.mp3 --output-dir my_chapters
```

## Model Selection

Choose model based on your hardware and accuracy needs:

| Model  | Speed      | Accuracy | RAM   | Recommended For          |
| ------ | ---------- | -------- | ----- | ------------------------ |
| tiny   | ⚡⚡⚡⚡⚡ | ★★★      | ~1GB  | CPU, quick tests         |
| base   | ⚡⚡⚡⚡   | ★★★★     | ~1GB  | CPU, good balance        |
| small  | ⚡⚡⚡     | ★★★★     | ~2GB  | CPU/GPU, better accuracy |
| medium | ⚡⚡       | ★★★★★    | ~5GB  | GPU recommended          |
| large  | ⚡         | ★★★★★    | ~10GB | GPU required             |

## Output

The tool creates:

```
chapters/
├── 01_Chapter-One.mp3
├── 02_Chapter-Two.mp3
├── 03_Chapter-Three.mp3
└── ...

chapters.json  # Chapter information with timestamps
```

## Performance Tips

### For CPU-Only Systems

1. **Use smaller models**: Start with `tiny` or `base`

   ```bash
   make split-fast  # Uses tiny model
   ```

2. **Process samples first**: Test with shorter segments

   ```bash
   make test  # Tests with 30-minute sample
   ```

3. **Close other applications**: Whisper uses significant RAM and CPU

### For GPU Systems

1. **Verify CUDA is available**:

   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   ```

2. **Use larger models** for better accuracy:
   ```bash
   python audiobook_chapter_splitter.py audiobook.mp3 --model medium
   ```

## Troubleshooting

### No chapters detected

The script looks for common patterns like "Chapter One", "Chapter 1", "Part 1", etc. If your audiobook uses different markers:

1. Check the transcription:

   ```bash
   python audiobook_chapter_splitter.py audiobook.mp3 --no-split
   cat full_transcription.txt  # Review the transcription
   ```

2. Modify chapter patterns in `audiobook_chapter_splitter.py`:
   ```python
   chapter_patterns = [
       r'chapter\s+(\d+)',
       r'episode\s+(\d+)',  # Add custom patterns
   ]
   ```

### Out of memory errors

- Use a smaller model (`tiny` or `base`)
- Process a sample instead of the full file
- Close other applications
- Consider processing on a system with more RAM

### Slow processing

- Use `tiny` model for 10-50x speedup
- Enable GPU if available
- Process shorter samples for testing

## Example Workflow

```bash
# 1. Set up with uv (recommended)
export AUDIOBOOK=~/Downloads/my_audiobook.mp3
make uv-install

# 2. Activate virtual environment
source .venv/bin/activate  # uv creates .venv by default

# 3. Quick test with sample
make test
# Review chapters.json to verify detection

# 4. Process full audiobook
make split-fast  # Fast mode with tiny model
# OR
make split       # Better accuracy with base model

# 5. Check results
ls chapters/
cat chapters.json
```

## Advanced Usage

### Custom Chapter Patterns

Edit `audiobook_chapter_splitter.py` to add custom patterns:

```python
chapter_patterns = [
    r'chapter\s+(\d+)',
    r'chapter\s+(\w+)',
    r'track\s+(\d+)',      # For music albums
    r'episode\s+(\d+)',    # For podcasts
    r'lesson\s+(\d+)',     # For educational content
]
```

### Batch Processing

```bash
for book in *.mp3; do
    python audiobook_chapter_splitter.py "$book" \
        --model tiny \
        --output-dir "chapters_${book%.mp3}"
done
```

## License

MIT

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [ffmpeg](https://ffmpeg.org/) for audio processing
