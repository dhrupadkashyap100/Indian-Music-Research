
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import librosa
import numpy as np
from transformers import AutoConfig, AutoModel, AutoFeatureExtractor, AutoImageProcessor, pipeline

def test_resnet_model():
    model_name = "neerajaabhyankar/resnet-finetuned-1-hindustani-raag-small"
    print(f"Loading config for: {model_name}")
    
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    print("\n" + "="*50)
    print("MODEL CONFIGURATION")
    print("="*50)
    print(f"Architecture: {config.architectures}")
    num_labels = len(config.id2label)
    print(f"Number of output classes: {num_labels}")
    
    print("\n" + "="*50)
    print("ID TO RAAG MAPPING")
    for k, v in config.id2label.items():
        print(f"{k}: {v}")

    # We will count parameters later when loading clf_model

    # Try loading processor
    processor = None
    try:
        processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        print(f"\nLoaded Feature Extractor: {type(processor)}")
    except:
        pass
        
    if not processor:
        try:
            processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            print(f"\nLoaded Image Processor: {type(processor)}")
        except Exception as e:
            print(f"Failed to load processor: {e}")
    
    if processor:
        print("\nPROCESSOR CONFIGURATION")
        print(processor)

    print("\n" + "="*50)
    print("INFERENCE ON TEST AUDIO")
    print("="*50)
    test_audio = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\augmented\Baul\BAUL_001_SEG000_pitch.wav")
    
    try:
        from transformers import AutoModelForAudioClassification, AutoModelForImageClassification
        
        if config.architectures and "ResNetForImageClassification" in config.architectures[0]:
            model_cls = AutoModelForImageClassification
            is_image = True
        else:
            # Maybe it's registered as audio classification
            model_cls = AutoModelForAudioClassification
            is_image = False
            
        clf_model = model_cls.from_pretrained(model_name, trust_remote_code=True)
        total_params = sum(p.numel() for p in clf_model.parameters())
        print(f"\nTOTAL PARAMETERS: {total_params:,}")
        
        # Load audio
        waveform, sr = librosa.load(test_audio, sr=None)
        
        if hasattr(processor, 'sampling_rate'):
            expected_sr = processor.sampling_rate
            waveform, sr = librosa.load(test_audio, sr=expected_sr)
            
        print(f"Loaded audio: shape {waveform.shape}, sr {sr}")

        if is_image:
            print("Model expects an image. Converting audio to mel-spectrogram image...")
            S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            S_normalized = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
            S_normalized = (S_normalized * 255).astype(np.uint8)
            
            from PIL import Image
            img = Image.fromarray(S_normalized).convert('RGB')
            
            inputs = processor(img, return_tensors="pt")
            print(f"Input pixel_values shape: {inputs['pixel_values'].shape}")
        else:
            print("Model expects audio waveform.")
            try:
                inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
            except TypeError:
                # Custom processor might not support return_tensors="pt"
                out = processor(waveform, sampling_rate=sr)
                if isinstance(out, dict):
                    inputs = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in out.items()}
                else:
                    inputs = {"input_values": torch.tensor(out)}
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clf_model = clf_model.to(device)
        
        # Add batch dimension if missing
        for k, v in inputs.items():
            if v.dim() == 1:
                inputs[k] = v.unsqueeze(0).unsqueeze(0) # 1, 1, 80000
            elif v.dim() == 2:
                inputs[k] = v.unsqueeze(0) # 1, 2, 80000
                
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print("Inputs shape:")
        for k, v in inputs.items():
            print(f"  {k}: {v.shape}")
        
        # If the model expects a certain shape like (batch, channels, time) for 1D, or (batch, channels, freq, time) for 2D.
        # Let's inspect the first layer of the model
        first_layer = list(clf_model.modules())[1]
        print(f"First layer: {first_layer}")
        
        with torch.no_grad():
            outputs = clf_model(**inputs)
            
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        k = min(10, len(probabilities))
        top_probs, top_indices = torch.topk(probabilities, k)

        print(f"\nTop-{k} Predictions:")
        print(f"{'Rank':<5} | {'Raag':<25} | {'Probability'}")
        print("-" * 50)
        for i in range(k):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            raag = clf_model.config.id2label[idx]
            print(f"{i+1:<5} | {raag:<25} | {prob*100:.2f}%")
            
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_resnet_model()
