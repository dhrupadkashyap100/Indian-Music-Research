
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor
from tqdm import tqdm
import matplotlib.pyplot as plt

def normalize_raag_name(name):
    if not isinstance(name, str):
        return ""
    return str(name).replace(" ", "").replace("_", "").lower()

def main():
    metadata_path = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
    out_dir = os.path.join(PROJECT_ROOT, r"scripts\saraga_reference_consistency")
    os.makedirs(out_dir, exist_ok=True)
    
    target_classes = ["Shree", "Jog", "Todi", "Marwa", "Bihag", "Bairagi", "Multani", "Dhani", "Malkauns", "Hameer"]
    target_classes_lower = {c.lower(): c for c in target_classes}
    
    df = pd.read_csv(metadata_path)
    
    selected_recordings = []
    
    for idx, row in df.iterrows():
        audio_path = os.path.join(row['folder'], row['audio_file'])
        raag_label = str(row['canonical_raga'])
        norm_label = normalize_raag_name(raag_label)
        
        matched_class = None
        for t in target_classes:
            if raag_label.replace(" ", "") == t or norm_label == t.lower():
                matched_class = t
                break
                
        if matched_class and os.path.exists(audio_path):
            selected_recordings.append({
                'filename': row['audio_file'],
                'full_audio_path': audio_path,
                'true_raag': matched_class
            })
            
    print(f"Found {len(selected_recordings)} valid recordings matching target classes.")
    
    model_name = "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small"
    print(f"Loading config, processor, model for {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    expected_sr = processor.sampling_rate
    segment_len = 5.0
    segment_samples = int(segment_len * expected_sr)
    batch_size = 64
    num_classes = len(config.id2label)
    
    segment_records = []
    
    for rec in tqdm(selected_recordings, desc="Processing SARAGA Recordings"):
        audio_path = rec['full_audio_path']
        filename = rec['filename']
        true_raag = rec['true_raag']
        
        try:
            waveform, _ = librosa.load(audio_path, sr=expected_sr, mono=True)
            num_segments = len(waveform) // segment_samples
            
            if num_segments == 0:
                continue
                
            segments = []
            for i in range(num_segments):
                segments.append(waveform[i*segment_samples : (i+1)*segment_samples])
                
            all_probs = []
            
            with torch.inference_mode():
                for i in range(0, num_segments, batch_size):
                    batch_segs = segments[i:i+batch_size]
                    batch_tensors = []
                    for seg in batch_segs:
                        inputs = processor(seg, sampling_rate=expected_sr)
                        inp_val = inputs['input_values']
                        if not isinstance(inp_val, torch.Tensor):
                            inp_val = torch.tensor(inp_val)
                        if inp_val.dim() == 1:
                            inp_val = inp_val.unsqueeze(0)
                        batch_tensors.append(inp_val)
                        
                    batch_tensor = torch.stack(batch_tensors).to(device)
                    outputs = model(input_values=batch_tensor)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
                    
            all_probs = np.concatenate(all_probs, axis=0)
            
            for i in range(num_segments):
                p = all_probs[i]
                sorted_idx = np.argsort(p)[::-1]
                
                top5_raags = [config.id2label[sorted_idx[j]] for j in range(5)]
                top5_probs = [p[sorted_idx[j]] for j in range(5)]
                
                # find true raag rank and prob
                true_raag_id = config.label2id.get(true_raag)
                true_prob = p[true_raag_id] if true_raag_id is not None else 0.0
                true_rank = np.where(sorted_idx == true_raag_id)[0][0] + 1 if true_raag_id is not None else -1
                
                segment_records.append({
                    'recording': filename,
                    'true_raag': true_raag,
                    'segment_index': i,
                    'start_seconds': i * segment_len,
                    'end_seconds': (i+1) * segment_len,
                    'predicted_raag_top1': top5_raags[0],
                    'top1_probability': top5_probs[0],
                    'true_raag_probability': true_prob,
                    'true_raag_rank': true_rank,
                    'top2_raag': top5_raags[1],
                    'top2_probability': top5_probs[1],
                    'top3_raag': top5_raags[2],
                    'top3_probability': top5_probs[2],
                    'top4_raag': top5_raags[3],
                    'top4_probability': top5_probs[3],
                    'top5_raag': top5_raags[4],
                    'top5_probability': top5_probs[4]
                })
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    seg_df = pd.DataFrame(segment_records)
    seg_df.to_csv(os.path.join(out_dir, "segment_predictions.csv"), index=False)
    
    if len(seg_df) == 0:
        print("No segments processed. Exiting.")
        return

    # Recording-level metrics
    rec_records = []
    for rec_name, group in seg_df.groupby('recording'):
        true_raag = group['true_raag'].iloc[0]
        num_segs = len(group)
        
        preds = group['predicted_raag_top1'].tolist()
        maj_pred = Counter(preds).most_common(1)[0][0]
        
        true_top1 = (group['true_raag_rank'] == 1).sum()
        true_top3 = (group['true_raag_rank'] <= 3).sum()
        true_top5 = (group['true_raag_rank'] <= 5).sum()
        
        rec_records.append({
            'recording': rec_name,
            'true_raag': true_raag,
            'evaluated_segments': num_segs,
            'majority_vote_predicted_raag': maj_pred,
            'percent_predicting_true_raag': (true_top1 / num_segs) * 100,
            'top1_hit_rate': true_top1 / num_segs,
            'top3_hit_rate': true_top3 / num_segs,
            'top5_hit_rate': true_top5 / num_segs,
            'mean_true_raag_probability': group['true_raag_probability'].mean(),
            'median_true_raag_probability': group['true_raag_probability'].median()
        })
        
    rec_df = pd.DataFrame(rec_records)
    rec_df.to_csv(os.path.join(out_dir, "recording_level_results.csv"), index=False)
    
    # Class-level metrics & Confusion
    class_records = []
    confusion_records = []
    
    for raag in target_classes:
        group = seg_df[seg_df['true_raag'] == raag]
        num_recs = len(group['recording'].unique())
        num_segs = len(group)
        
        if num_segs == 0:
            continue
            
        true_top1 = (group['true_raag_rank'] == 1).sum()
        true_top3 = (group['true_raag_rank'] <= 3).sum()
        true_top5 = (group['true_raag_rank'] <= 5).sum()
        
        most_common_pred = Counter(group['predicted_raag_top1']).most_common(1)[0]
        pred_concentration = (most_common_pred[1] / num_segs) * 100
        
        class_records.append({
            'true_raag': raag,
            'number_of_recordings': num_recs,
            'number_of_valid_segments': num_segs,
            'top1_accuracy': true_top1 / num_segs,
            'top3_accuracy': true_top3 / num_segs,
            'top5_accuracy': true_top5 / num_segs,
            'mean_probability': group['true_raag_probability'].mean(),
            'median_probability': group['true_raag_probability'].median(),
            'mean_top1_confidence': group['top1_probability'].mean(),
            'true_raag_top1_rate': true_top1 / num_segs,
            'true_raag_top5_rate': true_top5 / num_segs,
            'prediction_concentration': pred_concentration
        })
        
        confusion_records.append({
            'True Raag': raag,
            'Most common prediction': most_common_pred[0],
            'Percentage': pred_concentration,
            'True Raag Top-1 %': (true_top1 / num_segs) * 100,
            'True Raag Top-5 %': (true_top5 / num_segs) * 100
        })
        
    class_df = pd.DataFrame(class_records)
    class_df.to_csv(os.path.join(out_dir, "class_level_results.csv"), index=False)
    
    conf_df = pd.DataFrame(confusion_records)
    conf_df.to_csv(os.path.join(out_dir, "confusion_summary.csv"), index=False)
    
    # Prediction distribution
    dist = seg_df['predicted_raag_top1'].value_counts().reset_index()
    dist.columns = ['predicted_raag', 'count']
    dist['percentage'] = (dist['count'] / len(seg_df)) * 100
    dist.to_csv(os.path.join(out_dir, "prediction_distribution.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(class_df['true_raag'], class_df['top5_accuracy'] * 100, color='skyblue')
    plt.title('Top-5 Hit Rate by Reference Raag')
    plt.ylabel('Top-5 Hit Rate (%)')
    plt.xlabel('True Raag')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top5_hit_rate.png"))
    
    # Text Report
    report = []
    report.append("SARAGA REFERENCE CONSISTENCY REPORT")
    report.append("===================================")
    
    # Classification
    consistent = []
    partially_consistent = []
    inconsistent = []
    
    for idx, row in class_df.iterrows():
        top5 = row['top5_accuracy']
        if top5 >= 0.5:
            consistent.append(row['true_raag'])
            cat = "CONSISTENT"
        elif top5 >= 0.15:
            partially_consistent.append(row['true_raag'])
            cat = "PARTIALLY CONSISTENT"
        else:
            inconsistent.append(row['true_raag'])
            cat = "INCONSISTENT"
            
        report.append(f"Raag: {row['true_raag']} -> {cat} (Top-5: {top5*100:.1f}%)")
        
    bias_most_common = dist.iloc[0]['predicted_raag']
    bias_perc = dist.iloc[0]['percentage']
    
    report.append("\nFINAL DECISION SUPPORT")
    report.append("======================")
    
    report.append("1. Does the pretrained model show enough consistency on SARAGA to justify using it as a reference model for the Folk dataset?")
    if len(consistent) > len(target_classes) / 2:
        report.append("   Yes, it shows reasonable consistency on the majority of reference classes.")
    else:
        report.append("   No, it is highly inconsistent. It repeatedly fails to recognize its own training classes on high-quality classical audio.")
        
    report.append(f"\n2. Which SARAGA Raags does the model recognize most consistently?")
    report.append(f"   {', '.join(consistent) if consistent else 'None significantly'}.")
    
    report.append(f"\n3. Which Raags are consistently confused?")
    report.append(f"   {', '.join(inconsistent) if inconsistent else 'None'}.")
    
    report.append(f"\n4. Is the model collapsing toward a small number of output classes?")
    report.append(f"   Yes, it shows bias. For example, {bias_most_common} represents {bias_perc:.1f}% of all predictions across all reference classes.")
    
    report.append("\n5. Based on this experiment, should we:")
    report.append("   A. Proceed with the pretrained model for Folk candidate-label generation,")
    report.append("   B. Use it only as a weak reference,")
    report.append("   C. Abandon it completely and move to another approach?")
    
    if len(consistent) > len(target_classes) / 2:
        report.append("   Recommendation: B. Use it only as a weak reference.")
    else:
        report.append("   Recommendation: C. Abandon it completely and move to another approach.")
        
    with open(os.path.join(out_dir, "saraga_reference_consistency_report.txt"), "w") as f:
        f.write("\n".join(report))
        
    # JSON
    with open(os.path.join(out_dir, "saraga_reference_consistency_summary.json"), "w") as f:
        json.dump({
            'consistent_classes': consistent,
            'partially_consistent_classes': partially_consistent,
            'inconsistent_classes': inconsistent,
            'overall_top_prediction': bias_most_common,
            'overall_top_prediction_percentage': bias_perc,
            'recommendation': "Abandon completely" if len(consistent) <= len(target_classes)/2 else "Weak reference"
        }, f, indent=4)
        
    print("Files Generated:")
    for file in os.listdir(out_dir):
        print(f"  {os.path.join(out_dir, file)}")

if __name__ == '__main__':
    main()
