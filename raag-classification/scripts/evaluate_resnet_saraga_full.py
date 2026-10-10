
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
    return str(name).replace(" ", "").replace("_", "").lower()

def map_saraga_to_model(saraga_raag, model_raags):
    normalized_saraga = normalize_raag_name(saraga_raag)
    
    for m_raag in model_raags:
        if normalized_saraga == normalize_raag_name(m_raag):
            return m_raag
            
    aliases = {
        "bhimpalasi": "Bheempalasi",
        "ahribhairav": "AheerBhairav",
        "ahirobhairav": "AheerBhairav",
        "darbari": "DarbariKanada",
        "alhaiyabilawal": "AlhaiyaBilawal",
        "puriakalyan": "PuriyaKalyan",
        "puriadhanashree": "PuriyaDhanashri",
        "puriadhanashri": "PuriyaDhanashri",
        "marubihaag": "MaruBihag",
        "kalyan": "Yaman",
        "yaman": "Yaman",
        "bageshri": "Bageshree",
        "bhupali": "Bhoopali",
        "shuddhakalyan": "Kalyan" 
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
    
    # ---------------- SANITY CHECK ----------------
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
            if len(waveform) < 40000:
                print(f"  Audio too short ({len(waveform)} samples).")
                continue
            
            # evaluate on first 30s
            eval_wave = waveform[:expected_sr*30]
            
            inputs = processor(eval_wave, sampling_rate=expected_sr)
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
            
            print(f"Predicted Top-5 Raags:")
            for i in range(5):
                print(f"  {i+1}. {config.id2label[top_indices[i].item()]} ({top_probs[i].item()*100:.2f}%)")
                
        except Exception as e:
            print(f"  Error: {e}")

    # ---------------- FULL EVALUATION ----------------
    print("\n" + "="*50)
    print("FULL EVALUATION")
    print("="*50)
    
    results = []
    y_true = []
    y_pred = []
    
    for idx, row in supported_df.iterrows():
        audio_path = os.path.join(row['folder'], row['audio_file'])
        ground_truth = row['mapped_raag']
        
        try:
            waveform, sr = librosa.load(audio_path, sr=expected_sr)
            if len(waveform) < 40000:
                continue
                
            # Chunking and averaging or just first 30s?
            # Let's just use the first 30s for standard evaluation across all files to be fast and consistent
            eval_wave = waveform[:expected_sr*30]
            
            inputs = processor(eval_wave, sampling_rate=expected_sr)
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
            
            pred_1 = config.id2label[top_indices[0].item()]
            
            y_true.append(ground_truth)
            y_pred.append(pred_1)
            
            results.append({
                "filename": row['audio_file'],
                "ground_truth_raag": ground_truth,
                "predicted_raag": pred_1,
                "top1_probability": top_probs[0].item(),
                "top2_raag": config.id2label[top_indices[1].item()],
                "top2_probability": top_probs[1].item(),
                "top3_raag": config.id2label[top_indices[2].item()],
                "top3_probability": top_probs[2].item(),
                "top4_raag": config.id2label[top_indices[3].item()],
                "top4_probability": top_probs[3].item(),
                "top5_raag": config.id2label[top_indices[4].item()],
                "top5_probability": top_probs[4].item(),
                "correct_top1": ground_truth == pred_1
            })
            
        except Exception as e:
            print(f"Error processing {row['audio_file']}: {e}")

    # Metrics
    if len(y_true) > 0:
        acc_top1 = accuracy_score(y_true, y_pred)
        
        # Calculate Top-5 accuracy manually
        top5_correct = 0
        for r in results:
            if r['ground_truth_raag'] in [r['predicted_raag'], r['top2_raag'], r['top3_raag'], r['top4_raag'], r['top5_raag']]:
                top5_correct += 1
        acc_top5 = top5_correct / len(results)
        
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"\nNumber of evaluated audio files: {len(results)}")
        print(f"Number of supported Raag classes evaluated: {len(set(y_true))}")
        print(f"Top-1 Accuracy: {acc_top1:.4f}")
        print(f"Top-5 Accuracy: {acc_top5:.4f}")
        print(f"Macro Precision: {macro_prec:.4f}")
        print(f"Macro Recall: {macro_rec:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        # Conf Matrix
        print("\nConfusion Matrix:")
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("Classes:", labels)
        print(cm)
        
        # Save CSV
        res_df = pd.DataFrame(results)
        out_csv = os.path.join(PROJECT_ROOT, r"scripts\saraga_evaluation_results.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"\nDetailed evaluation saved to: {out_csv}")
    else:
        print("No files were successfully evaluated.")

if __name__ == '__main__':
    main()
