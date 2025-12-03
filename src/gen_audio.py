from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
import soundfile as sf
import torch

MODEL_DIR = "musicgen-ambient"

model = MusicgenForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype="auto"
)
processor = MusicgenProcessor.from_pretrained(MODEL_DIR)

prompt = "evolving atmospheric pads, deep ambient drone, slow movement"

# If your finetuning does NOT use text: comment out text=prompt
inputs = processor(
    text=prompt,
    return_tensors="pt"
).to("cuda")

print("Generating audio...")

audio_values = model.generate(
    **inputs,
    do_sample=True,
    temperature=1.0,
    top_p=0.9,
    max_new_tokens=1400   # ~1.5 minutes
)

audio = audio_values[0].cpu().numpy()
sf.write("generated_ambient.wav", audio, 32000)

print("Saved to generated_ambient.wav")
