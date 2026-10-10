
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import librosa
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor

def test_recording(file_path, gt_raag, model, processor, config, segment_len=5.0):
    print(f"\n" + "="*60)
    print(f"DIAGNOSTIC: {os.path.basename(file_path)} | Ground Truth: {gt_raag}")
    print("="*60)
    
    # 1. Inspect original audio properties
    orig_waveform, orig_sr = librosa.load(file_path, sr=None, mono=False)
    
    channels = 1 if orig_waveform.ndim == 1 else orig_waveform.shape[0]
    duration = orig_waveform.shape[-1] / orig_sr
    dtype = orig_waveform.dtype
    amp_min, amp_max = orig_waveform.min(), orig_waveform.max()
    
    print(f"--- Original Audio Properties ---")
    print(f"Sample Rate: {orig_sr} Hz")
    print(f"Channels: {channels}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Dtype: {dtype}")
    print(f"Amplitude Range: [{amp_min:.4f}, {amp_max:.4f}]")
    
    # Reload with expected SR for model
    expected_sr = processor.sampling_rate
    waveform, _ = librosa.load(file_path, sr=expected_sr, mono=True)
    
    # Split into segments
    segment_samples = int(segment_len * expected_sr)
    num_segments = len(waveform) // segment_samples
    
    print(f"\n--- Model Evaluation ({segment_len}s segments) ---")
    print(f"Expected SR: {expected_sr} Hz, Segment Samples: {segment_samples}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    segment_results = []
    
    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        seg = waveform[start_idx:end_idx]
        
        # Processor handles mono->stereo, normalization
        inputs = processor(seg, sampling_rate=expected_sr)
        
        # Fix shape to include batch dim: (1, 2, L)
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if v.dim() == 1:
                inputs[k] = v.unsqueeze(0).unsqueeze(0)
            elif v.dim() == 2:
                inputs[k] = v.unsqueeze(0)
                
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, 5)
        
        preds = []
        for j in range(5):
            preds.append((config.id2label[top_indices[j].item()], top_probs[j].item()))
            
        segment_results.append({
            'start_time': start_idx / expected_sr,
            'end_time': end_idx / expected_sr,
            'top1_raag': preds[0][0],
            'top1_prob': preds[0][1],
            'top5': preds
        })
        
    # Aggregate
    if not segment_results:
        print("No full segments found.")
        return
        
    top1_preds = [s['top1_raag'] for s in segment_results]
    pred_counts = Counter(top1_preds)
    majority_raag = pred_counts.most_common(1)[0][0]
    
    top1_probs = [s['top1_prob'] for s in segment_results]
    mean_conf = np.mean(top1_probs)
    median_conf = np.median(top1_probs)
    
    correct_segs = sum(1 for p in top1_preds if p == gt_raag)
    top5_correct_segs = sum(1 for s in segment_results if gt_raag in [x[0] for x in s['top5']])
    
    print(f"Total Segments: {num_segments}")
    print(f"Top-1 Segment Accuracy: {correct_segs}/{num_segments} ({correct_segs/num_segments*100:.1f}%)")
    print(f"Top-5 Segment Accuracy: {top5_correct_segs}/{num_segments} ({top5_correct_segs/num_segments*100:.1f}%)")
    print(f"Mean Top-1 Confidence: {mean_conf*100:.1f}%")
    print(f"Median Top-1 Confidence: {median_conf*100:.1f}%")
    
    print("\nMost Frequently Predicted Raags (Top-1 across segments):")
    for r, c in pred_counts.most_common(3):
        print(f"  {r}: {c} segments")
        
    print(f"\nSong-Level Majority Vote Prediction: {majority_raag}")
    if majority_raag == gt_raag:
        print(">>> MATCHES Ground Truth!")
    else:
        print(">>> FAILS to match Ground Truth.")
        
    return segment_results

def main():
    model_name = "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small"
    print(f"Loading config, processor, model for {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # We will hardcode the 5 test files based on the previous output
    base_dir = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\Saraga\saraga1.5_hindustani")
    test_files = [
        (r"Anaahata by Milind Malshe\Raag Bairagi\Bairagi.mp3.mp3", "Bairagi"),
        (r"Raag Malkauns\Malkauns.mp3.mp3", "Malkauns"), # Wait, exact path might vary, let's load from CSV
    ]
    
    # Let's dynamically find them using the CSV
    metadata_path = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
    df = pd.read_csv(metadata_path)
    
    target_raags = ["Bairagi", "Malkauns", "Todi", "Multani", "MaruBihag"]
    
    def normalize_raag_name(name):
        return str(name).replace(" ", "").replace("_", "").lower()
        
    def get_mapped(r):
        n = normalize_raag_name(r)
        if n == "bairagi": return "Bairagi"
        if n == "malkauns": return "Malkauns"
        if n == "todi": return "Todi"
        if n == "multani": return "Multani"
        if n == "marubihag": return "MaruBihag"
        return None
        
    df['diag_raag'] = df['canonical_raga'].apply(get_mapped)
    subset = df[df['diag_raag'].notna()].drop_duplicates(subset=['diag_raag'])
    
    for _, row in subset.iterrows():
        audio_path = os.path.join(row['folder'], row['audio_file'])
        gt_raag = row['diag_raag']
        if os.path.exists(audio_path):
            test_recording(audio_path, gt_raag, model, processor, config, segment_len=5.0)
        else:
            print(f"File not found: {audio_path}")

if __name__ == '__main__':
    main()
