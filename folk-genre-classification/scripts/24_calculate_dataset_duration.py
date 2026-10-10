import os
import csv
import librosa
from pathlib import Path
from datetime import timedelta
import time
import hashlib
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_AUDIO_DIR = PROJECT_ROOT / "datasets" / "raw_audio"

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as afile:
        buf = afile.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(65536)
    return hasher.hexdigest()

def main():
    if not RAW_AUDIO_DIR.exists():
        print(f"Error: {RAW_AUDIO_DIR} does not exist.")
        return

    csv_data = []
    genre_stats = {}
    
    total_songs = 0
    total_duration_sec = 0.0
    longest_song = {"name": "", "duration": 0.0}
    shortest_song = {"name": "", "duration": float('inf')}
    
    corrupted_files = []
    hashes = {}
    duplicates = []
    
    print(f"Scanning {RAW_AUDIO_DIR}...")
    
    start_time = time.time()
    
    for genre_dir in RAW_AUDIO_DIR.iterdir():
        if not genre_dir.is_dir():
            continue
            
        genre = genre_dir.name
        if genre not in genre_stats:
            genre_stats[genre] = {"count": 0, "total_duration": 0.0}
            
        for file_path in genre_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
                file_size = file_path.stat().st_size
                
                # Check for duplicates using hash
                file_hash = get_file_hash(file_path)
                if file_hash in hashes:
                    duplicates.append((file_path.name, hashes[file_hash]))
                else:
                    hashes[file_hash] = file_path.name
                
                try:
                    duration = librosa.get_duration(path=str(file_path))
                    
                    csv_data.append({
                        "File Name": file_path.name,
                        "Genre": genre,
                        "Duration (seconds)": duration,
                        "Duration (HH:MM:SS)": format_duration(duration),
                        "File Size": file_size
                    })
                    
                    total_songs += 1
                    total_duration_sec += duration
                    genre_stats[genre]["count"] += 1
                    genre_stats[genre]["total_duration"] += duration
                    
                    if duration > longest_song["duration"]:
                        longest_song = {"name": file_path.name, "duration": duration}
                        
                    if duration < shortest_song["duration"]:
                        shortest_song = {"name": file_path.name, "duration": duration}
                        
                except Exception as e:
                    corrupted_files.append(file_path.name)
                    print(f"Error reading {file_path.name}: {e}")
                    
    print(f"Scan complete in {time.time() - start_time:.2f} seconds.")
    
    csv_path = PROJECT_ROOT / "dataset_duration.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["File Name", "Genre", "Duration (seconds)", "Duration (HH:MM:SS)", "File Size"])
        writer.writeheader()
        writer.writerows(csv_data)
        
    print(f"Saved CSV to {csv_path}")
    
    summary = {
        "total_songs": total_songs,
        "total_duration_sec": total_duration_sec,
        "total_duration_hhmmss": format_duration(total_duration_sec),
        "average_duration_sec": total_duration_sec / total_songs if total_songs > 0 else 0,
        "average_duration_hhmmss": format_duration(total_duration_sec / total_songs if total_songs > 0 else 0),
        "longest_song": longest_song,
        "shortest_song": shortest_song,
        "corrupted_files": corrupted_files,
        "duplicates": duplicates,
        "genre_stats": genre_stats
    }
    
    summary_path = PROJECT_ROOT / "duration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Saved JSON summary to {summary_path}")

if __name__ == "__main__":
    main()
