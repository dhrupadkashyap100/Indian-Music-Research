import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import json
import pandas as pd

# Root folder of the SARAGA dataset
SARAGA_ROOT = str(BASE_DIR / "reference_dataset" / "Saraga" / "saraga1.5_hindustani")

# Output CSV file
OUTPUT_CSV = str(BASE_DIR / "reference_dataset" / "metadata" / "saraga_metadata.csv")

metadata_list = []

print("Extracting metadata...\n")

non_raga_keywords = ['lyrics', 'introduction', 'introductory', 'taal']
rec_id_counter = 1

def normalize_raga(name, fallback):
    if not name or str(name).strip() == "":
        name = fallback
    name = str(name).replace('Raag ', '').strip()
    
    mapping = {
        'Bhoopali': 'Bhoop',
        'Sohani': 'Sohini',
        'Bhimpalasi': 'Bhimpalas',
        'Bhairavi Bhajan': 'Bhairabi',
        'Bhairavi Dadra': 'Bhairabi',
        'Bhairavi Thumri': 'Bhairabi',
        'Irani Bhairavi Thumri': 'Bhairabi',
        'Aahir Bhairon': 'Ahir bhairav',
        'Raag Ahir Bhairon': 'Ahir bhairav',
        'Bilaskhani Todi': 'Bilaskhani todi',
        'Raga Lalit - Khayal': 'Lalat',
        'Raga Shree - Khayal': 'Shree',
        'Raageshree': 'Rageshri',
        'Miyan malhar': 'Miya malhar',
        'Gaud malhar': 'Gaud Malhar'
    }
    # Capitalize first letter to standardize
    normalized = mapping.get(name, name)
    normalized = normalized[0].upper() + normalized[1:] if normalized else normalized
    if normalized == "Gaud malhar": normalized = "Gaud Malhar"
    return normalized

for root, dirs, files in sorted(os.walk(SARAGA_ROOT)):

    mp3_file = None
    json_file = None
    tonic_file = None

    # Find the required files
    for file in files:
        if file.endswith(".mp3") or file.endswith(".mp3.mp3"):
            mp3_file = file
        elif file.endswith(".json"):
            json_file = file
        elif file.endswith(".ctonic.txt"):
            tonic_file = file

    # Skip folders that don't contain all three files
    if not (mp3_file and json_file and tonic_file):
        continue
        
    # Check if it's a non-musical file based on mp3_file or folder name
    folder_name = os.path.basename(root)
    is_speech = any(k in mp3_file.lower() or k in folder_name.lower() for k in non_raga_keywords)
    if is_speech:
        print(f"Skipping non-musical recording: {folder_name}")
        continue

    # Read JSON file
    json_path = os.path.join(root, json_file)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Read tonic value
    tonic_path = os.path.join(root, tonic_file)

    with open(tonic_path, "r") as f:
        tonic = f.read().strip()

    # Extract artist name
    artist = ""
    if "artists" in data and len(data["artists"]) > 0:
        artist = data["artists"][0]["artist"]["name"]

    # Extract raga name
    raga = ""
    if "raags" in data and len(data["raags"]) > 0:
        raga = data["raags"][0]["common_name"]
        
    if not raga or str(raga).strip() == "":
        print(f"Skipping recording due to missing raga metadata: {folder_name}")
        continue
        
    # Use canonical raga name
    canonical_raga = normalize_raga(raga, None)

    unique_id = f"REC_{rec_id_counter:04d}"
    rec_id_counter += 1

    # Save metadata
    metadata_list.append({
        "unique_id": unique_id,
        "audio_file": mp3_file,
        "canonical_raga": canonical_raga,
        "artist": artist,
        "tonic": tonic,
        "folder": root
    })

# Convert to DataFrame
df = pd.DataFrame(metadata_list)

# Save CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("\nDone! Metadata saved to:")
print(OUTPUT_CSV)

print("\nFirst five rows:")
print(df.head())

print("\nTotal valid musical recordings:", len(df))
print("Unique ragas:", df["canonical_raga"].nunique())

print("\nList of ragas:")
print(sorted(df["canonical_raga"].unique()))