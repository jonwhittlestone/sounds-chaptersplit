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
    print(f"Audio duration: {total_duration:.1f} seconds")
    
    print(f"Transcribing '{audio_file}' in {chunk_duration}s chunks...")
    print("This may take a while depending on file size and CPU/GPU...")
    
    all_segments = []
    current_time = 0
    chunk_overlap = 2  # seconds of overlap
    
    with tempfile.TemporaryDirectory() as tmpdir:
        while current_time < total_duration:
            # Calculate chunk duration
            actual_duration = min(chunk_duration, total_duration - current_time)
            
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
                
            except Exception as e:
                print(f"Error processing chunk at {current_time}s: {e}")
            finally:
                # Clean up chunk file
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            
            # Progress
            progress = (current_time / total_duration) * 100
            print(f"  Processed: {progress:.1f}%", end='\r')
            
            # Move to next chunk (with overlap)
            current_time += chunk_duration - chunk_overlap
    
    print("\n  Transcription complete!")
    
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


def split_audio_file(audio_file: str, chapters: List[Tuple[str, float]], output_dir: str = "chapters"):
    """
    Split audio file into chapters using ffmpeg.
    
    Args:
        audio_file: Path to the source audio file
        chapters: List of (chapter_name, timestamp) tuples
        output_dir: Directory to save chapter files
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get audio duration
    duration = get_audio_duration(audio_file)
    
    print(f"\nSplitting audio into {len(chapters)} chapters...")
    
    for i, (chapter_name, start_time) in enumerate(chapters):
        # Determine end time (start of next chapter or end of file)
        end_time = chapters[i + 1][1] if i + 1 < len(chapters) else duration
        
        # Clean filename
        safe_filename = re.sub(r'[^\w\s-]', '', chapter_name)
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
        output_file = f"{output_dir}/{i+1:02d}_{safe_filename}.mp3"
        
        # Split using ffmpeg
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


def save_chapter_info(chapters: List[Tuple[str, float]], output_file: str = "chapters.json"):
    """Save chapter information to JSON file."""
    chapter_data = [
        {
            "index": i + 1,
            "name": name,
            "timestamp": timestamp,
            "time_formatted": format_time(timestamp)
        }
        for i, (name, timestamp) in enumerate(chapters)
    ]
    
    with open(output_file, 'w') as f:
        json.dump(chapter_data, f, indent=2)
    
    print(f"Chapter information saved to '{output_file}'")


def main():
    """Main function to run the chapter splitter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split audiobook into chapters using Whisper (streaming version)')
    parser.add_argument('audio_file', help='Path to the audiobook audio file')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--output-dir', default='chapters',
                       help='Output directory for chapter files (default: chapters)')
    parser.add_argument('--no-split', action='store_true',
                       help='Only transcribe and find chapters, do not split audio')
    parser.add_argument('--chunk-duration', type=int, default=60,
                       help='Duration of audio chunks in seconds (default: 60)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found")
        return
    
    # Transcribe audio
    transcription = transcribe_audio_streaming(args.audio_file, args.model, args.chunk_duration)
    
    # Find chapter markers
    chapters = find_chapter_markers(transcription)
    
    if not chapters:
        print("\nNo chapter markers found in transcription.")
        print("You may need to adjust the chapter patterns or review the transcription.")
        
        # Save full transcription for review
        with open('full_transcription.txt', 'w') as f:
            f.write(transcription['text'])
        print("Full transcription saved to 'full_transcription.txt' for review.")
    else:
        print(f"\nFound {len(chapters)} chapters:")
        for i, (name, timestamp) in enumerate(chapters, 1):
            print(f"  {i}. {name} at {format_time(timestamp)}")
        
        # Save chapter information
        save_chapter_info(chapters)
        
        # Split audio if requested
        if not args.no_split:
            split_audio_file(args.audio_file, chapters, args.output_dir)
        else:
            print("\nSkipping audio splitting (--no-split flag used)")


if __name__ == "__main__":
    main()