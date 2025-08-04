#!/usr/bin/env python3
"""
Audiobook Chapter Splitter using OpenAI Whisper with streaming processing
Transcribes an audiobook and splits it into chapters based on detected chapter markers.
Uses streaming to handle large files without memory issues.
"""

import whisper
import re
import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Tuple


def get_audio_duration(audio_file: str) -> float:
    """Get duration of audio file in seconds."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
    ]
    duration = float(subprocess.check_output(cmd).decode().strip())
    return duration


def extract_audio_chunk(audio_file: str, start_time: float, duration: float, output_file: str):
    """Extract a chunk of audio using ffmpeg."""
    cmd = [
        'ffmpeg', '-ss', str(start_time), '-i', audio_file,
        '-t', str(duration), '-ar', '16000', '-ac', '1',
        '-y', output_file
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def transcribe_audio_streaming(audio_file: str, model_size: str = "base", chunk_duration: int = 60) -> Dict:
    """
    Transcribe audio file using Whisper by processing it in chunks via ffmpeg.
    
    Args:
        audio_file: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        chunk_duration: Duration of each chunk in seconds
    
    Returns:
        Transcription result with segments and timestamps
    """
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    
    # Get total duration
    total_duration = get_audio_duration(audio_file)
    total_chunks = int((total_duration + chunk_duration - 1) / (chunk_duration - 2))  # Account for overlap
    print(f"Audio duration: {format_time(total_duration)} ({total_duration:.1f} seconds)")
    print(f"Will process in approximately {total_chunks} chunks of {chunk_duration}s each")
    
    print(f"\nTranscribing '{audio_file}'...")
    print("This may take a while depending on file size and CPU/GPU...")
    print("\nProgress:")
    
    all_segments = []
    current_time = 0
    chunk_overlap = 2  # seconds of overlap
    chunk_count = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        while current_time < total_duration:
            chunk_count += 1
            # Calculate chunk duration
            actual_duration = min(chunk_duration, total_duration - current_time)
            
            # Progress bar visualization
            progress = (current_time / total_duration) * 100
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"Chunk {chunk_count:3d}/{total_chunks}: [{bar}] {progress:5.1f}% | Time: {format_time(current_time)}/{format_time(total_duration)}")
            
            # Extract chunk to temporary file
            chunk_file = os.path.join(tmpdir, f"chunk_{current_time}.wav")
            extract_audio_chunk(audio_file, current_time, actual_duration, chunk_file)
            
            # Transcribe chunk
            try:
                chunk_result = model.transcribe(
                    chunk_file,
                    word_timestamps=True,
                    verbose=False,
                    language='en',
                    fp16=False  # Disable FP16 for CPU
                )
                
                # Adjust timestamps
                for segment in chunk_result['segments']:
                    segment['start'] += current_time
                    segment['end'] += current_time
                    if 'words' in segment:
                        for word in segment['words']:
                            word['start'] += current_time
                            word['end'] += current_time
                    all_segments.append(segment)
                
                # Show sample text from this chunk (first 50 chars)
                if chunk_result['segments']:
                    sample_text = chunk_result['segments'][0]['text'][:50].strip()
                    if sample_text:
                        print(f"  └─ Sample: \"{sample_text}...\"")
                
            except Exception as e:
                print(f"  └─ Error processing chunk at {current_time}s: {e}")
            finally:
                # Clean up chunk file
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            
            # Move to next chunk (with overlap)
            current_time += chunk_duration - chunk_overlap
    
    # Final progress bar at 100%
    bar = '█' * bar_length
    print(f"Chunk {chunk_count:3d}/{total_chunks}: [{bar}] 100.0% | Time: {format_time(total_duration)}/{format_time(total_duration)}")
    print("\n✓ Transcription complete!")
    
    # Combine results
    result = {
        'text': ' '.join(seg['text'] for seg in all_segments),
        'segments': all_segments,
        'language': 'en'
    }
    
    return result


def find_chapter_markers(transcription: Dict) -> List[Tuple[str, float]]:
    """
    Find chapter markers in the transcription.
    
    Args:
        transcription: Whisper transcription result
    
    Returns:
        List of (chapter_name, timestamp) tuples
    """
    chapters = []
    
    # Common chapter patterns
    chapter_patterns = [
        r'chapter\s+(\d+)',
        r'chapter\s+(\w+)',
        r'part\s+(\d+)',
        r'book\s+(\d+)',
        r'section\s+(\d+)',
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in chapter_patterns)
    
    for segment in transcription['segments']:
        text = segment['text'].lower().strip()
        
        # Check for chapter markers
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                chapter_name = match.group(0).title()
                timestamp = segment['start']
                
                # Avoid duplicate chapters within 30 seconds
                if not chapters or abs(timestamp - chapters[-1][1]) > 30:
                    chapters.append((chapter_name, timestamp))
                    print(f"Found: {chapter_name} at {format_time(timestamp)}")
                break
    
    return chapters


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def split_audio_file(audio_file: str, chapters: List[Tuple[str, float]], output_dir: str = None):
    """
    Split audio file into chapters using ffmpeg.
    
    Args:
        audio_file: Path to the source audio file
        chapters: List of (chapter_name, timestamp) tuples
        output_dir: Directory to save chapter files (default: chapters subdirectory next to audio file)
    """
    # If no output directory specified, create chapters subdirectory next to audio file
    if output_dir is None:
        audio_path = Path(audio_file).resolve()
        output_dir = audio_path.parent / "chapters"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get audio duration
    duration = get_audio_duration(audio_file)
    
    print(f"\nSplitting audio into {len(chapters)} chapters...")
    
    for i, (chapter_name, start_time) in enumerate(chapters):
        # Determine end time (start of next chapter or end of file)
        end_time = chapters[i + 1][1] if i + 1 < len(chapters) else duration
        
        # Clean filename
        safe_filename = re.sub(r'[^\w\s-]', '', chapter_name)
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
        output_file = str(output_dir / f"{i+1:02d}_{safe_filename}.mp3")
        
        # Split using ffmpeg
        # Check if we need to convert formats
        input_ext = os.path.splitext(audio_file)[1].lower()
        output_ext = os.path.splitext(output_file)[1].lower()
        
        if input_ext in ['.m4b', '.m4a'] and output_ext == '.mp3':
            # Need to convert from m4b/m4a to mp3
            cmd = [
                'ffmpeg', '-i', audio_file,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:a', 'libmp3lame',  # Convert to mp3
                '-q:a', '2',  # Good quality (VBR ~190 kbps)
                '-y',  # Overwrite output files
                output_file
            ]
        else:
            # Can copy codec directly
            cmd = [
                'ffmpeg', '-i', audio_file,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c', 'copy',  # Copy codec (no re-encoding, faster)
                '-y',  # Overwrite output files
                output_file
            ]
        
        print(f"Creating: {output_file} [{format_time(start_time)} - {format_time(end_time)}]")
        subprocess.run(cmd, capture_output=True, check=True)
    
    print(f"\n✓ Successfully created {len(chapters)} chapter files in '{output_dir}/'")


def save_chapter_info(chapters: List[Tuple[str, float]], output_file: str = None, audio_file: str = None, duration: float = None):
    """Save chapter information to JSON file.
    
    Args:
        chapters: List of (chapter_name, timestamp) tuples
        output_file: Path to save JSON file (default: chapters.json next to audio file)
        audio_file: Path to the audio file (used to determine default output location)
        duration: Total duration of the audio file in seconds
    """
    # If no output file specified and audio file provided, save next to audio file
    if output_file is None and audio_file is not None:
        audio_path = Path(audio_file).resolve()
        output_file = str(audio_path.parent / "chapters" / "chapters.json")
    elif output_file is None:
        output_file = "chapters.json"
    
    chapter_data = {
        "total_duration": duration if duration is not None else None,
        "total_duration_formatted": format_time(duration) if duration is not None else None,
        "total_chapters": len(chapters),
        "chapters": [
            {
                "index": i + 1,
                "name": name,
                "timestamp": timestamp,
                "time_formatted": format_time(timestamp)
            }
            for i, (name, timestamp) in enumerate(chapters)
        ]
    }
    
    # Ensure parent directory exists
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(chapter_data, f, indent=2)
    
    print(f"Chapter information saved to '{output_file}'")


def main():
    """Main function to run the chapter splitter."""
    import argparse
    import time
    
    # Track start time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Split audiobook into chapters using Whisper (streaming version)')
    parser.add_argument('audio_file', help='Path to the audiobook audio file')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for chapter files (default: chapters subdirectory next to audio file)')
    parser.add_argument('--no-split', action='store_true',
                       help='Only transcribe and find chapters, do not split audio')
    parser.add_argument('--chunk-duration', type=int, default=60,
                       help='Duration of audio chunks in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found")
        return
    
    # Get audio duration for reporting
    duration = get_audio_duration(args.audio_file)
    
    # Transcribe audio
    transcription = transcribe_audio_streaming(args.audio_file, args.model, args.chunk_duration)
    
    # Find chapter markers
    chapters = find_chapter_markers(transcription)
    
    if not chapters:
        print("\nNo chapter markers found in transcription.")
        print("You may need to adjust the chapter patterns or review the transcription.")
        print(f"Total audio duration: {format_time(duration)}")
        
        # Save full transcription for review
        with open('full_transcription.txt', 'w') as f:
            f.write(transcription['text'])
        print("Full transcription saved to 'full_transcription.txt' for review.")
    else:
        print(f"\nFound {len(chapters)} chapters in {format_time(duration)} of audio:")
        for i, (name, timestamp) in enumerate(chapters, 1):
            print(f"  {i}. {name} at {format_time(timestamp)}")
        
        # Save chapter information with duration
        save_chapter_info(chapters, audio_file=args.audio_file, duration=duration)
        
        # Split audio if requested
        if not args.no_split:
            split_audio_file(args.audio_file, chapters, args.output_dir)
        else:
            print("\nSkipping audio splitting (--no-split flag used)")
        
        print(f"\nTotal audio duration: {format_time(duration)}")
    
    # Calculate and display total execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {format_time(execution_time)}")


if __name__ == "__main__":
    main()