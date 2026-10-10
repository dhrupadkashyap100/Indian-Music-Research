
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

def test_model():
    model_name = "neerajaabhyankar/distilhubert-finetuned-hindustani-raag-small"
    print(f"Loading model: {model_name}")

    try:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Print model architecture
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    print(model)

    # 5. Print total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*50)
    print(f"TOTAL PARAMETERS: {total_params:,}")

    # 6. Print number of output classes
    num_classes = len(model.config.id2label)
    print("\n" + "="*50)
    print(f"NUMBER OF OUTPUT CLASSES: {num_classes}")

    # 7. Print ID -> Raag mapping
    print("\n" + "="*50)
    print("ID TO RAAG MAPPING")
    for k, v in model.config.id2label.items():
        print(f"{k}: {v}")

    # 8. Print label -> ID mapping
    print("\n" + "="*50)
    print("RAAG TO ID MAPPING")
    for k, v in model.config.label2id.items():
        print(f"{k}: {v}")

    # 9. Print sampling rate
    expected_sr = extractor.sampling_rate
    print("\n" + "="*50)
    print(f"EXPECTED SAMPLING RATE: {expected_sr} Hz")

    # 10. Print useful input configuration
    print("\n" + "="*50)
    print("PROCESSOR CONFIGURATION")
    print(extractor)

    # Task 4 - Verify one audio inference
    test_audio = os.path.join(PROJECT_ROOT, r"Folk_Music_Classifier\datasets\augmented\Baul\BAUL_001_SEG000_pitch.wav")
    
    print("\n" + "="*50)
    print(f"RUNNING INFERENCE ON: {test_audio}")
    print("="*50)

    try:
        # Load audio
        waveform, sr = librosa.load(test_audio, sr=expected_sr)
        
        if len(waveform) > expected_sr * 30:
            print("Truncating audio to 30 seconds for test to avoid OOM.")
            waveform = waveform[:expected_sr * 30]

        inputs = extractor(waveform, sampling_rate=expected_sr, return_tensors="pt")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Top k
        k = min(5, len(probabilities))
        top_probs, top_indices = torch.topk(probabilities, k)

        print(f"\nTop-{k} Predictions:")
        print(f"{'Rank':<5} | {'Raag':<25} | {'Probability'}")
        print("-" * 50)
        for i in range(k):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            raag = model.config.id2label[idx]
            print(f"{i+1:<5} | {raag:<25} | {prob*100:.2f}%")

        from huggingface_hub import constants
        print("\n" + "="*50)
        print(f"Hugging Face Cache Dir: {constants.HF_HUB_CACHE}")

    except Exception as e:
        print(f"Inference error: {e}")

if __name__ == '__main__':
    test_model()
