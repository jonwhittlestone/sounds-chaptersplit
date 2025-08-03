.PHONY: help install uv-install uv-sync test sample split clean

# Default target
help:
	@echo "Audiobook Chapter Splitter - Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "  make uv-install   - Create venv and install deps with uv (recommended)"
	@echo "  make install      - Install Python dependencies (standard pip)"
	@echo ""
	@echo "  Usage:"
	@echo "  make test         - Test with a sample (first 30 min) without splitting"
	@echo "  make sample       - Create a 30-minute sample from audiobook"
	@echo "  make split        - Split full audiobook into chapters (streaming)"
	@echo "  make split-fast   - Split using tiny model (faster on CPU, streaming)"
	@echo "  make clean        - Remove generated files and directories"
	@echo ""
	@echo "Usage:"
	@echo "  1. Set AUDIOBOOK variable: export AUDIOBOOK=path/to/your/audiobook.mp3"
	@echo "  2. Run: make uv-install (or make install)"
	@echo "  3. Activate venv: source .venv/bin/activate"
	@echo "  4. Run: make test (to test with sample)"
	@echo "  5. Run: make split (to process full audiobook)"

# Check if AUDIOBOOK variable is set
check-audiobook:
ifndef AUDIOBOOK
	$(error AUDIOBOOK is not set. Please run: export AUDIOBOOK=path/to/your/audiobook.mp3)
endif

# Install dependencies with standard pip
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo ""
	@echo "Checking ffmpeg installation..."
	@which ffmpeg > /dev/null || (echo "Error: ffmpeg not found. Please install ffmpeg:" && echo "  Ubuntu/Debian: sudo apt-get install ffmpeg" && echo "  macOS: brew install ffmpeg" && echo "  Windows: Download from https://ffmpeg.org/download.html" && exit 1)
	@echo "✓ All dependencies installed"

# Install with uv (recommended)
uv-install:
	@echo "Checking for uv installation..."
	@which uv > /dev/null || (echo "Error: uv not found. Install it with:" && echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)
	@echo "Creating virtual environment with uv (Python 3.12)..."
	uv venv --python 3.12
	@echo "Installing Python dependencies..."
	uv pip install -r requirements.txt
	@echo ""
	@echo "Checking ffmpeg installation..."
	@which ffmpeg > /dev/null || (echo "Error: ffmpeg not found. Please install ffmpeg:" && echo "  Ubuntu/Debian: sudo apt-get install ffmpeg" && echo "  macOS: brew install ffmpeg" && echo "  Windows: Download from https://ffmpeg.org/download.html" && exit 1)
	@echo "✓ All dependencies installed"
	@echo ""
	@echo "Activate the virtual environment with:"
	@echo "  source .venv/bin/activate"

# Sync dependencies with uv
uv-sync:
	@echo "Syncing dependencies with uv..."
	uv pip sync requirements.txt
	@echo "✓ Dependencies synced"

# Create a 30-minute sample for testing
sample: check-audiobook
	@echo "Creating 30-minute sample from $(AUDIOBOOK)..."
	@mkdir -p samples
	ffmpeg -i "$(AUDIOBOOK)" -t 1800 -c copy samples/audiobook_sample.mp3 -y
	@echo "✓ Sample created: samples/audiobook_sample.mp3"

# Test with sample (no splitting)
test: check-audiobook sample
	@echo "Testing chapter detection on sample with streaming processing..."
	python audiobook_chapter_splitter_streaming.py samples/audiobook_sample.mp3 --model tiny --no-split
	@echo ""
	@echo "✓ Test complete. Check chapters.json for detected chapters"

# Split full audiobook with base model
split: check-audiobook
	@echo "Processing full audiobook with base model (streaming)..."
	@echo "This may take a while depending on file size..."
	python audiobook_chapter_splitter_streaming.py "$(AUDIOBOOK)" --model base --chunk-duration 60
	@echo "✓ Audiobook split into chapters/"

# Split with tiny model (faster on CPU)
split-fast: check-audiobook
	@echo "Processing audiobook with tiny model (fast mode, streaming)..."
	python audiobook_chapter_splitter_streaming.py "$(AUDIOBOOK)" --model tiny --chunk-duration 90
	@echo "✓ Audiobook split into chapters/"

# Split with custom output directory
split-to:
ifndef OUTPUT_DIR
	$(error OUTPUT_DIR is not set. Please run: make split-to OUTPUT_DIR=my_chapters)
endif
	@echo "Processing audiobook to $(OUTPUT_DIR) with streaming..."
	python audiobook_chapter_splitter_streaming.py "$(AUDIOBOOK)" --output-dir "$(OUTPUT_DIR)" --chunk-duration 60
	@echo "✓ Audiobook split into $(OUTPUT_DIR)/"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf chapters/
	rm -rf samples/
	rm -f chapters.json
	rm -f full_transcription.txt
	@echo "✓ Cleaned"

# Advanced targets for different models (streaming)
split-small: check-audiobook
	python audiobook_chapter_splitter_streaming.py "$(AUDIOBOOK)" --model small --chunk-duration 60

split-medium: check-audiobook
	python audiobook_chapter_splitter_streaming.py "$(AUDIOBOOK)" --model medium --chunk-duration 45

split-large: check-audiobook
	python audiobook_chapter_splitter_streaming.py "$(AUDIOBOOK)" --model large --chunk-duration 30