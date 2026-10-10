
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import librosa
import numpy as np
import pandas as pd
import requests
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
import time

def main():
    model_name = "ntua-slp/CultureMERT-95M"
    print(f"Investigating model: {model_name}")
    print("="*50)
    
    # 1. Fetch Model Card
    print("Fetching Model Card from Hugging Face API...")
    try:
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            tags = data.get('tags', [])
            print(f"Tags: {tags}")
        
        # get readme
        readme_url = f"https://huggingface.co/{model_name}/raw/main/README.md"
        readme_resp = requests.get(readme_url)
        if readme_resp.status_code == 200:
            with open("culturemert_readme.md", "w", encoding="utf-8") as f:
                f.write(readme_resp.text)
            print("Model card downloaded as culturemert_readme.md")
    except Exception as e:
        print(f"Error fetching model card: {e}")
        
    # 2. Load Config and Extractor
    print("\nLoading Config and Feature Extractor...")
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"Architecture: {config.architectures}")
        print(f"Hidden size (Embedding dim): {getattr(config, 'hidden_size', 'Unknown')}")
        print(f"Sampling rate: {getattr(processor, 'sampling_rate', 'Unknown')}")
        
    except Exception as e:
        print(f"Error loading config/processor: {e}")
        return
        
    # 3. Load Model
    print("\nLoading Model...")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameter count: {params:,}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # 4. Find Audio Files
    folk_file = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\raw_audio\Baul\BAUL_001.wav")
    saraga_meta = os.path.join(PROJECT_ROOT, r"Reference_Model\Raaga_Info\reference_dataset\metadata\saraga_metadata.csv")
    
    saraga_file = None
    try:
        df = pd.read_csv(saraga_meta)
        saraga_file = os.path.join(df.iloc[0]['folder'], df.iloc[0]['audio_file'])
    except:
        pass
        
    if not os.path.exists(folk_file):
        # try another folk file
        import glob
        folk_files = glob.glob(os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\raw_audio\*\*.wav"))
        if folk_files:
            folk_file = folk_files[0]
            
    print(f"\nTesting Inference on:")
    print(f"Folk: {folk_file}")
    print(f"SARAGA: {saraga_file}")
    
    # Inference function
    def get_embedding(audio_path, max_dur=5.0):
        if not audio_path or not os.path.exists(audio_path):
            return None
        
        sr = getattr(processor, 'sampling_rate', 16000)
        wav, _ = librosa.load(audio_path, sr=sr, duration=max_dur, mono=True)
        
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.inference_mode():
            outputs = model(**inputs)
            
        inf_time = time.time() - start
        mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        # Last hidden state
        last_hidden = outputs.last_hidden_state
        shape = last_hidden.shape
        
        # Mean pooling
        pooled = last_hidden.mean(dim=1).cpu().numpy()[0]
        
        print(f"\nStats for {os.path.basename(audio_path)}:")
        print(f"  Hidden state shape: {shape}")
        print(f"  Inference time: {inf_time:.4f}s")
        print(f"  GPU Memory usage: {mem:.2f} MB")
        
        return pooled
        
    emb_folk = get_embedding(folk_file)
    emb_saraga = get_embedding(saraga_file)
    
    # 5. Cosine Similarity
    if emb_folk is not None and emb_saraga is not None:
        from numpy.linalg import norm
        sim = np.dot(emb_folk, emb_saraga) / (norm(emb_folk) * norm(emb_saraga))
        print(f"\nCosine Similarity (Folk vs SARAGA): {sim:.4f}")
        
if __name__ == '__main__':
    main()
