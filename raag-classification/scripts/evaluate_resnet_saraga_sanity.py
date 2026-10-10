
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor

def normalize_raag_name(name):
    # Remove spaces and convert to lower case for comparison
    return str(name).replace(" ", "").replace("_", "").lower()

def map_saraga_to_model(saraga_raag, model_raags):
    normalized_saraga = normalize_raag_name(saraga_raag)
    
    # Direct match after normalization
    for m_raag in model_raags:
        if normalized_saraga == normalize_raag_name(m_raag):
            return m_raag
            
    # Some common aliases if needed
    aliases = {
        "bhimpalasi": "Bheempalasi",
        "ahribhairav": "AheerBhairav",
        "ahirobhairav": "AheerBhairav",
        "darbari": "DarbariKanada",
        "alhaiyabilawal": "AlhaiyaBilawal",
        "puriakalyan": "PuriyaKalyan",
        "puriadhanashree": "PuriyaDhanashri",
        "puriadhanashri": "PuriyaDhanashri",
        "shuddhakalyan": "Kalyan", # Wait, no Shuddha Kalyan in 50? 
        "marubihaag": "MaruBihag"
    }
    
    if normalized_saraga in aliases:
        if aliases[normalized_saraga] in model_raags:
            return aliases[normalized_saraga]
            
    return None

def main():
    metadata_path = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
    df = pd.read_csv(metadata_path)
    
    model_name = "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small"
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    model_raags = list(config.label2id.keys())
    print(f"Model supports {len(model_raags)} Raags.")
    
    # Add mapped column
    df['mapped_raag'] = df['canonical_raga'].apply(lambda x: map_saraga_to_model(x, model_raags))
    
    supported_df = df[df['mapped_raag'].notna()]
    unsupported_df = df[df['mapped_raag'].isna()]
    
    print(f"Total SARAGA files: {len(df)}")
    print(f"Supported files: {len(supported_df)}")
    print(f"Unsupported files: {len(unsupported_df)}")
    print("\nUnsupported Raags found in SARAGA:")
    print(unsupported_df['canonical_raga'].value_counts())
    
    if len(supported_df) == 0:
        print("No supported files found to evaluate!")
        return
        
    print("\nInitializing model and processor...")
    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    expected_sr = processor.sampling_rate
    
    # Sanity check on 3-5 files
    sanity_files = supported_df.head(5)
    print("\n" + "="*50)
    print("SANITY CHECK (First 5 files)")
    print("="*50)
    
    for idx, row in sanity_files.iterrows():
        audio_path = os.path.join(row['folder'], row['audio_file'])
        ground_truth = row['mapped_raag']
        
        print(f"\nFile: {row['audio_file']}")
        print(f"Ground-truth Raag: {ground_truth}")
        
        try:
            waveform, sr = librosa.load(audio_path, sr=expected_sr)
            
            # The model expects at least 40000 samples. Let's take the first 30 seconds if too long, or at least 40000
            if len(waveform) < 40000:
                print(f"  Audio too short ({len(waveform)} samples). Skipping.")
                continue
                
            # For testing, we might want to evaluate on 10s chunks or just the first 10s to be fast.
            # Let's take samples from 30s to 60s if possible, else first available
            if len(waveform) > expected_sr * 60:
                waveform = waveform[expected_sr*30 : expected_sr*60]
            
            inputs = processor(waveform, sampling_rate=expected_sr)
            if isinstance(inputs, dict):
                inputs = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            else:
                inputs = {"input_values": torch.tensor(inputs)}
                
            for k, v in inputs.items():
                if v.dim() == 1:
                    inputs[k] = v.unsqueeze(0).unsqueeze(0)
                elif v.dim() == 2:
                    inputs[k] = v.unsqueeze(0)
                    
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            top_probs, top_indices = torch.topk(probs, 5)
            
            print(f"Predicted Top-5 Raags:")
            for i in range(5):
                pred_raag = config.id2label[top_indices[i].item()]
                prob_val = top_probs[i].item() * 100
                print(f"  {i+1}. {pred_raag} ({prob_val:.2f}%)")
                
        except Exception as e:
            print(f"  Error processing: {e}")

if __name__ == '__main__':
    main()
