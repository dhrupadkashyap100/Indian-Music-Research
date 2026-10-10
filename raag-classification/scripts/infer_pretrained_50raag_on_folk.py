
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import glob
import torch
import librosa
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor
from tqdm import tqdm

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
    out_dir = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\results\pretrained_50raag_folk_inference")
    os.makedirs(out_dir, exist_ok=True)
    
    # Gather files
    audio_files = []
    for root, _, files in os.walk(raw_audio_dir):
        for f in files:
            if f.endswith('.wav') or f.endswith('.mp3'):
                audio_files.append(os.path.join(root, f))
                
    expected_sr = processor.sampling_rate
    segment_len = 5.0
    segment_samples = int(segment_len * expected_sr)
    batch_size = 64
    
    segment_records = []
    song_records = []
    
    files_processed = 0
    files_skipped = 0
    skip_reasons = Counter()
    
    num_classes = len(config.id2label)
    
    for audio_path in tqdm(audio_files, desc="Processing Folk Dataset"):
        filename = os.path.basename(audio_path)
        genre = os.path.basename(os.path.dirname(audio_path))
        
        try:
            # Load and resample
            waveform, _ = librosa.load(audio_path, sr=expected_sr, mono=True)
            
            num_segments = len(waveform) // segment_samples
            if num_segments == 0:
                files_skipped += 1
                skip_reasons['Audio too short'] += 1
                continue
                
            # Create batches
            segments = []
            for i in range(num_segments):
                segments.append(waveform[i*segment_samples : (i+1)*segment_samples])
                
            all_probs = []
            
            with torch.inference_mode():
                for i in range(0, num_segments, batch_size):
                    batch_segs = segments[i:i+batch_size]
                    
                    # Process batch
                    # Note: custom processor expects 1D array per call. Let's process one by one to get tensors, then stack
                    batch_tensors = []
                    for seg in batch_segs:
                        inputs = processor(seg, sampling_rate=expected_sr)
                        inp_val = inputs['input_values']
                        if not isinstance(inp_val, torch.Tensor):
                            inp_val = torch.tensor(inp_val)
                        if inp_val.dim() == 1:
                            inp_val = inp_val.unsqueeze(0) # (1, L) -> (2, L) handled by processor? 
                            # Wait, the custom processor returns (2, 40000).
                        # Let's trust it returns shape (2, L)
                        batch_tensors.append(inp_val)
                        
                    batch_tensor = torch.stack(batch_tensors).to(device) # (B, 2, L)
                    
                    outputs = model(input_values=batch_tensor)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
                    
            all_probs = np.concatenate(all_probs, axis=0) # (num_segments, num_classes)
            
            # Segment level processing
            song_top1_preds = []
            song_top1_confs = []
            
            for i in range(num_segments):
                p = all_probs[i]
                top5_idx = np.argsort(p)[::-1][:5]
                
                seg_rec = {
                    'genre': genre,
                    'original_filename': filename,
                    'full_audio_path': audio_path,
                    'segment_index': i,
                    'segment_start_time': i * segment_len,
                    'segment_end_time': (i+1) * segment_len,
                    'top1_predicted_raag': config.id2label[top5_idx[0]],
                    'top1_probability': p[top5_idx[0]],
                    'top2_raag': config.id2label[top5_idx[1]],
                    'top2_probability': p[top5_idx[1]],
                    'top3_raag': config.id2label[top5_idx[2]],
                    'top3_probability': p[top5_idx[2]],
                    'top4_raag': config.id2label[top5_idx[3]],
                    'top4_probability': p[top5_idx[3]],
                    'top5_raag': config.id2label[top5_idx[4]],
                    'top5_probability': p[top5_idx[4]]
                }
                
                # Optional: preserve complete 50-class prob
                for class_id in range(num_classes):
                    seg_rec[f'prob_{config.id2label[class_id]}'] = p[class_id]
                    
                segment_records.append(seg_rec)
                song_top1_preds.append(seg_rec['top1_predicted_raag'])
                song_top1_confs.append(seg_rec['top1_probability'])
                
            # Song level aggregation
            mean_probs = np.mean(all_probs, axis=0)
            song_top5_idx = np.argsort(mean_probs)[::-1][:5]
            
            pred_counts = Counter(song_top1_preds)
            majority_raag = pred_counts.most_common(1)[0][0]
            majority_perc = pred_counts.most_common(1)[0][1] / num_segments
            
            consistency = majority_perc
            mean_conf = np.mean(song_top1_confs)
            max_conf = np.max(song_top1_confs)
            
            song_rec = {
                'genre': genre,
                'filename': filename,
                'number_of_segments': num_segments,
                'majority_vote_raag': majority_raag,
                'majority_vote_percentage': majority_perc,
                'mean_top1_confidence': mean_conf,
                'max_top1_confidence': max_conf,
                'prediction_consistency': consistency,
                'song_top1_raag': config.id2label[song_top5_idx[0]],
                'song_top1_probability': mean_probs[song_top5_idx[0]],
                'song_top2_raag': config.id2label[song_top5_idx[1]],
                'song_top2_probability': mean_probs[song_top5_idx[1]],
                'song_top3_raag': config.id2label[song_top5_idx[2]],
                'song_top3_probability': mean_probs[song_top5_idx[2]],
                'song_top4_raag': config.id2label[song_top5_idx[3]],
                'song_top4_probability': mean_probs[song_top5_idx[3]],
                'song_top5_raag': config.id2label[song_top5_idx[4]],
                'song_top5_probability': mean_probs[song_top5_idx[4]]
            }
            
            # Preserve complete mean prob
            for class_id in range(num_classes):
                song_rec[f'mean_prob_{config.id2label[class_id]}'] = mean_probs[class_id]
                
            song_records.append(song_rec)
            files_processed += 1
            
        except Exception as e:
            files_skipped += 1
            skip_reasons[str(e)] += 1
            print(f"Error processing {filename}: {e}")

    print("\nSaving detailed predictions...")
    seg_df = pd.DataFrame(segment_records)
    song_df = pd.DataFrame(song_records)
    
    seg_df.to_csv(os.path.join(out_dir, "segment_predictions.csv"), index=False)
    song_df.to_csv(os.path.join(out_dir, "song_level_predictions.csv"), index=False)
    
    # ---------------- SUMMARY METRICS ----------------
    
    # 1. Total segment distribution
    raag_dist = seg_df['top1_predicted_raag'].value_counts().reset_index()
    raag_dist.columns = ['raag', 'number_of_segments']
    raag_dist['percentage_of_segments'] = (raag_dist['number_of_segments'] / len(seg_df)) * 100
    raag_dist.to_csv(os.path.join(out_dir, "raag_prediction_distribution.csv"), index=False)
    
    # 2. Genre-specific distribution
    genre_raag = seg_df.groupby(['genre', 'top1_predicted_raag']).size().reset_index(name='count')
    genre_totals = genre_raag.groupby('genre')['count'].transform('sum')
    genre_raag['percentage'] = (genre_raag['count'] / genre_totals) * 100
    genre_raag.sort_values(['genre', 'count'], ascending=[True, False], inplace=True)
    genre_raag.to_csv(os.path.join(out_dir, "genre_raag_distribution.csv"), index=False)
    
    print("\n" + "="*50)
    print("QUALITY CHECKS / REPORT")
    print("="*50)
    print(f"Total files discovered: {len(audio_files)}")
    print(f"Successfully processed: {files_processed}")
    print(f"Skipped files: {files_skipped}")
    for reason, count in skip_reasons.items():
        print(f"  - {reason}: {count}")
        
    print(f"\nTotal segments evaluated: {len(seg_df)}")
    
    print(f"\nOverall Average Top-1 Confidence: {seg_df['top1_probability'].mean()*100:.2f}%")
    print(f"Overall Median Top-1 Confidence: {seg_df['top1_probability'].median()*100:.2f}%")
    
    highest = seg_df.loc[seg_df['top1_probability'].idxmax()]
    print(f"Highest confidence prediction: {highest['top1_predicted_raag']} ({highest['top1_probability']*100:.2f}%) in {highest['original_filename']}")
    
    lowest = seg_df.loc[seg_df['top1_probability'].idxmin()]
    print(f"Lowest confidence prediction: {lowest['top1_predicted_raag']} ({lowest['top1_probability']*100:.2f}%) in {lowest['original_filename']}")
    
    print("\nTop-5 Predicted Raags Overall:")
    print(raag_dist.head(5))
    
    print("\nTop Predicted Raag by Genre:")
    for g in genre_raag['genre'].unique():
        top_for_genre = genre_raag[genre_raag['genre'] == g].iloc[0]
        print(f"  {g}: {top_for_genre['top1_predicted_raag']} ({top_for_genre['percentage']:.1f}%)")
        
if __name__ == '__main__':
    main()
