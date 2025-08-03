#!/usr/bin/env python3
"""
Audiobook Chapter Splitter using OpenAI Whisper
Transcribes an audiobook and splits it into chapters based on detected chapter markers.
"""

import whisper
import re
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple


def transcribe_audio(audio_file: str, model_size: str = "base", chunk_length: int = 30) -> Dict:
    """
    Transcribe audio file using Whisper with word-level timestamps.
    Processes audio in chunks to reduce memory usage.
    
    Args:
        audio_file: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        chunk_length: Length of audio chunks in seconds (default: 30)
    
    Returns:
        Transcription result with segments and timestamps
    """
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing '{audio_file}' in chunks...")
    print("This may take a while depending on file size and CPU/GPU...")
    
    # Load audio and process in chunks to reduce memory usage
    import numpy as np
    
    # Load audio with Whisper's load_audio function
    audio = whisper.load_audio(audio_file)
    sample_rate = 16000  # Whisper uses 16kHz
    
    # Calculate chunk size in samples
    chunk_samples = chunk_length * sample_rate
    total_samples = len(audio)
    
    # Process audio in chunks
    all_segments = []
    offset = 0
    
    print(f"Processing audio in {chunk_length}s chunks...")
    
    while offset < total_samples:
        # Get chunk with overlap to avoid cutting words
        chunk_end = min(offset + chunk_samples, total_samples)
        chunk = audio[offset:chunk_end]
        
        # Pad if chunk is too short
        if len(chunk) < sample_rate:  # Less than 1 second
            chunk = np.pad(chunk, (0, sample_rate - len(chunk)))
        
        # Transcribe chunk
        chunk_result = model.transcribe(
            chunk,
            word_timestamps=True,
            verbose=False,
            language='en'  # Specify language to speed up processing
        )
        
        # Adjust timestamps based on offset
        time_offset = offset / sample_rate
        for segment in chunk_result['segments']:
            segment['start'] += time_offset
            segment['end'] += time_offset
            if 'words' in segment:
                for word in segment['words']:
                    word['start'] += time_offset
                    word['end'] += time_offset
            all_segments.append(segment)
        
        # Progress indicator
        progress = (offset / total_samples) * 100
        print(f"  Processed: {progress:.1f}%", end='\r')
        
        # Move to next chunk (with small overlap to avoid cutting words)
        overlap = 2 * sample_rate  # 2 second overlap
        offset += chunk_samples - overlap
    
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
    
    # Add end timestamp
    # Get audio duration using ffprobe
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
    ]
    duration = float(subprocess.check_output(cmd).decode().strip())
    
    print(f"\nSplitting audio into {len(chapters)} chapters...")
    
    for i, (chapter_name, start_time) in enumerate(chapters):
        # Determine end time (start of next chapter or end of file)
        end_time = chapters[i + 1][1] if i + 1 < len(chapters) else duration
        
        # Clean filename
        safe_filename = re.sub(r'[^\w\s-]', '', chapter_name)
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
        output_file = str(output_dir / f"{i+1:02d}_{safe_filename}.mp3")
        
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


def save_chapter_info(chapters: List[Tuple[str, float]], output_file: str = None, audio_file: str = None):
    """Save chapter information to JSON file.
    
    Args:
        chapters: List of (chapter_name, timestamp) tuples
        output_file: Path to save JSON file (default: chapters.json next to audio file)
        audio_file: Path to the audio file (used to determine default output location)
    """
    # If no output file specified and audio file provided, save next to audio file
    if output_file is None and audio_file is not None:
        audio_path = Path(audio_file).resolve()
        output_file = str(audio_path.parent / "chapters" / "chapters.json")
    elif output_file is None:
        output_file = "chapters.json"
    
    chapter_data = [
        {
            "index": i + 1,
            "name": name,
            "timestamp": timestamp,
            "time_formatted": format_time(timestamp)
        }
        for i, (name, timestamp) in enumerate(chapters)
    ]
    
    # Ensure parent directory exists
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w') as f:
        json.dump(chapter_data, f, indent=2)
    
    print(f"Chapter information saved to '{output_file}'")


def main():
    """Main function to run the chapter splitter."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Split audiobook into chapters using Whisper')
    parser.add_argument('audio_file', help='Path to the audiobook audio file')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for chapter files (default: chapters subdirectory next to audio file)')
    parser.add_argument('--no-split', action='store_true',
                       help='Only transcribe and find chapters, do not split audio')
    parser.add_argument('--chunk-length', type=int, default=30,
                       help='Length of audio chunks in seconds for processing (default: 30)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file '{args.audio_file}' not found")
        return
    
    # Transcribe audio
    transcription = transcribe_audio(args.audio_file, args.model, args.chunk_length)
    
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
        save_chapter_info(chapters, audio_file=args.audio_file)
        
        # Split audio if requested
        if not args.no_split:
            split_audio_file(args.audio_file, chapters, args.output_dir)
        else:
            print("\nSkipping audio splitting (--no-split flag used)")


if __name__ == "__main__":
    main()