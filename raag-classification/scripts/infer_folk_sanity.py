
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import glob
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor

def main():
    model_name = "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small"
    print(f"Loading config, processor, model for {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    raw_audio_dir = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\raw_audio")
    
    # Get 3-5 files across different genres
    genres = os.listdir(raw_audio_dir)
    test_files = []
    for g in genres[:3]: # take 3 genres
        g_path = os.path.join(raw_audio_dir, g)
        if os.path.isdir(g_path):
            files = [f for f in os.listdir(g_path) if f.endswith('.wav') or f.endswith('.mp3')]
            if files:
                test_files.append(os.path.join(g_path, files[0]))
                
    expected_sr = processor.sampling_rate
    segment_len = 5.0
    segment_samples = int(segment_len * expected_sr)
    
    print("\n" + "="*50)
    print("SANITY CHECK ON FOLK AUDIO")
    print("="*50)
    
    for audio_path in test_files:
        filename = os.path.basename(audio_path)
        genre = os.path.basename(os.path.dirname(audio_path))
        
        print(f"\nGenre: {genre} | File: {filename}")
        
        try:
            orig_waveform, orig_sr = librosa.load(audio_path, sr=None, mono=False)
            channels = 1 if orig_waveform.ndim == 1 else orig_waveform.shape[0]
            duration = orig_waveform.shape[-1] / orig_sr
            print(f"Original Audio: {orig_sr} Hz, {channels} channels, {duration:.2f}s")
            
            # Re-load or resample
            waveform, _ = librosa.load(audio_path, sr=expected_sr, mono=True)
            
            num_segments = len(waveform) // segment_samples
            print(f"Number of 5s segments: {num_segments}")
            
            if num_segments == 0:
                print("Audio too short.")
                continue
                
            segment_preds = []
            
            # We can do a batch here
            for i in range(num_segments):
                seg = waveform[i*segment_samples : (i+1)*segment_samples]
                inputs = processor(seg, sampling_rate=expected_sr)
                
                # fix tensor
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
                    
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                top_probs, top_indices = torch.topk(probs, 5)
                
                preds = [(config.id2label[top_indices[j].item()], top_probs[j].item()) for j in range(5)]
                segment_preds.append(preds[0][0])
                
                if i < 3 or i == num_segments - 1: # just print first 3 and last
                    print(f"  Seg {i+1}: 1.{preds[0][0]}({preds[0][1]*100:.1f}%) 2.{preds[1][0]}({preds[1][1]*100:.1f}%) 3.{preds[2][0]}({preds[2][1]*100:.1f}%)")
                elif i == 3:
                    print("  ...")
            
            from collections import Counter
            counts = Counter(segment_preds)
            song_majority = counts.most_common(1)[0]
            print(f"Song-Level Prediction: {song_majority[0]} ({song_majority[1]}/{num_segments} segments)")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    main()
