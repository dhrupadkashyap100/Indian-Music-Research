---
license: cc-by-nc-4.0
tags:
- audio
- music
metrics:
- roc_auc
- average_precision
- f1
model_type: audio-classification
pipeline_tag: audio-classification
library_name: transformers
base_model:
- m-a-p/MERT-v1-95M
---

# CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning
📑 [**Read the full paper (presented at ISMIR 2025)**](https://arxiv.org/abs/2506.17818)

**CultureMERT-95M** is a multi-culturally adapted 95M-parameter music foundation model based on [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M). It is developed through a two-stage **continual pre-training (CPT)** strategy on 650 hours of culturally diverse audio spanning Greek, Turkish, and Indian musical traditions. The model significantly improves representation quality for **"non-Western" music**, achieving an average improvement of **+4.9%** across ROC-AUC and mAP on culturally diverse non-Western music tagging tasks, surpassing prior state-of-the-art, while maintaining strong performance on Western-centric benchmarks such as MagnaTagATune and FMA-medium.

<img src="method_figure.png" width="500"/>

🔀 Alternative variant available: [**CultureMERT-TA-95M**](https://huggingface.co/ntua-slp/CultureMERT-TA-95M) is a merged model constructed via task arithmetic, combining separately adapted models—each continually pre-trained (using the same two-stage CPT strategy) on data from a single musical tradition.


# 🧠 Model Details

- **Architecture**: 12-layer Transformer encoder (768-dim) with a 7-layer 1D CNN frontend  
- **Input**: Raw mono audio at 24kHz
- **Training Context Length**: 5 seconds
- **Pretraining Objective**: MLM-style multi-task masked prediction of discrete [EnCodec](https://huggingface.co/facebook/encodec_24khz) acoustic tokens and continuous constant-Q transform (CQT) spectrogram reconstruction at a 75 Hz feature rate

---

# 🌍 Training Data

| Dataset         | Music Tradition             | Hours Used |
|-----------------|-----------------------------|------------|
| [*Lyra*](https://github.com/pxaris/lyra-dataset)            | Greek traditional/folk      | 50h        |
| [*Turkish-makam*](https://dunya.compmusic.upf.edu/makam/)   | Turkish/Ottoman classical   | 200h       |
| [*Hindustani*](https://dunya.compmusic.upf.edu/hindustani/) | North Indian classical      | 200h       |
| [*Carnatic*](https://dunya.compmusic.upf.edu/carnatic/)     | South Indian classical      | 200h       |

* To further stabilize adaptation and mitigate catastrophic forgetting, we include *20% Western data* (i.e., 20 hours from [*Music4All*](https://sites.google.com/view/contact4music4all)) during Stage 1 of continual pre-training.

> 🔒 The datasets used were obtained under research-use agreements and are not redistributed.

---

# 📊 Evaluation

We evaluate **CultureMERT-95M** via probing on both Western and non-Western music auto-tagging tasks. All results are averaged over five random seeds. Metrics used:

- **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve) 
- **mAP** (Mean Average Precision)  
- **Micro-F1** and **Macro-F1**


Evaluation follows the [MARBLE](https://github.com/a43992899/MARBLE) protocol under constrained settings. We use standardized train/test splits from [ccml](https://github.com/pxaris/ccml) for continual pre-training and probing-based evaluation.


## Evaluation Datasets and Metadata Used (Top-k Tags)

- **Non-Western traditions**:  
  - **Turkish-makam**: Top-30 tags, covering *makam*, *usul*, and *instruments*
  - **Hindustani**: Top-20 tags, primarily reflecting *raga*, *tala*, *instruments*, and *forms*
  - **Carnatic**: Top-20 tags, primarily reflecting *raga*, *tala*, *instruments*, and *forms*
  - **Lyra**: Top-30 tags, related to *genre*, regional (*place*), and *instruments* metadata
- **Western benchmarks**:  
  - **MagnaTagATune (MTAT)**: Top-50 tags, spanning *genre*, *instruments*, and *mood*  
  - **FMA-medium**: Top-20 hierarchical *genre* tags

Top-k tags were selected based on tag frequency distributions.


The evaluation results are shown in the following tables:

## ROC-AUC / mAP

|                        |  Turkish-makam  |   Hindustani   |   Carnatic   |    Lyra    |    FMA-medium    |    MTAT    |  **Avg.**  |
|------------------------|:-----------------:|:----------------:|:--------------:|:------------:|:-----------:|:------------:|:------------:|
| **MERT-v1-95M**        | 83.2% / 53.3%   | 82.4% / 52.9% | 74.9% / 39.7% | 85.7% / 56.5% | 90.7% / 48.1% | **89.6%** / 35.9% | 66.1% |
| **CultureMERT-95M**    | **89.6%** / **60.6%** | **88.2%** / **63.5%** | **79.2%** / **43.1%** | **86.9%** / **56.7%** | 90.7% / 48.1% | 89.4% / 35.9% | **69.3%** |


## Micro-F1 / Macro-F1

|                    | Turkish-makam | Hindustani | Carnatic | Lyra | FMA-medium | MTAT | **Avg.** |
|--------------------|:----------------:|:------------:|:----------:|:------:|:------:|:------:|:----------:|
| **MERT-v1-95M**        | 73.0% / 38.9%    | 71.1% / 33.2% | 80.1% / 30.0% | 72.4% / 42.6% | 57.0% / **36.9%** | **35.7%** / 21.2% | 49.3% |
| **CultureMERT-95M**    | **77.4%** / **45.8%** | **77.8%** / **50.4%** | **82.7%** / **32.5%** | **73.1%** / **43.1%** | **58.3%** / 36.6% | 35.6% / **22.9%** | **52.9%** |


📈 **CultureMERT-95M** outperforms the original **MERT-v1-95M** by an average of **+4.4%** in ROC-AUC across non-Western traditions, with consistent improvements of **+5.4%** in mAP, **+3.6%** in Micro-F1, and **+6.8%** in Macro-F1. On Western datasets, it maintains performance with only a **−0.05%** average drop across ROC-AUC and mAP, and even improves by **+0.65%** in Micro/Macro-F1, resulting in an overall average gain of **+0.3%** across all metrics and datasets (MTAT and FMA-medium).

---

# 🔧 Model Usage

```python
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset

# Load model weights and preprocessor config
model = AutoModel.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)

# Load example audio
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True).sort("id")
audio_array = dataset[0]["audio"]["array"]
sampling_rate = dataset.features["audio"].sampling_rate

# Resample if needed
resample_rate = processor.sampling_rate
if resample_rate != sampling_rate:
    print(f'Setting sample rate from {sampling_rate} to {resample_rate}')
    resampler = T.Resample(sampling_rate, resample_rate)
else:
    resampler = None

# Audio file is decoded on the fly
if resampler is None:
	input_audio = dataset[0]["audio"]["array"]
else:
  input_audio = resampler(torch.from_numpy(dataset[0]["audio"]["array"]).to(dtype=resampler.kernel.dtype))

# Extract hidden states
inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# Representations: 13 layers (CNN feature extractor + 12 Transformer)
# NOTE: each layer performs differently in different downstream tasks - you should choose empirically
all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [13 layers, Time steps, 768 feature_dim]

# For utterance-level classification tasks, you can simply reduce the representation in time
time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
print(time_reduced_hidden_states.shape) # [13, 768]

# You can even use a learnable weighted average representation over all layers
aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
print(weighted_avg_hidden_states.shape) # [768]
```

---

# Ethical Considerations

This model is released under a non-commercial CC BY-NC 4.0 license and is intended for research purposes. While it is designed to address cultural bias in MIR, its training data and pretraining paradigm may still reflect cultural and dataset-specific biases. The model should not be used in commercial or generative applications without explicit consideration of cultural representation, proper attribution, and consent from relevant communities or dataset curators.


# 📚 Citation

```shell
@misc{kanatas2025culturemertcontinualpretrainingcrosscultural,
      title={CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning}, 
      author={Angelos-Nikolaos Kanatas and Charilaos Papaioannou and Alexandros Potamianos},
      year={2025},
      eprint={2506.17818},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2506.17818}, 
}
```

---