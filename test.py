from transformers import pipeline
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pipe = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    framework="pt",
    device=device,
    batch_size=64,
)
