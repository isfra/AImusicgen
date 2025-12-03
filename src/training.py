from transformers import MusicgenForConditionalGeneration, MusicgenProcessor, TrainingArguments, Trainer
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# ------------------------
# 1. Load Base Model
# ------------------------
model_name = "facebook/musicgen-medium"
model = MusicgenForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

processor = MusicgenProcessor.from_pretrained(model_name)

# ------------------------
# 2. Apply LoRA (recommended)
# ------------------------
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ------------------------
# 3. Load Dataset
# ------------------------
ds = load_from_disk("ambient_dataset")

def preprocess(batch):
    audio = batch["audio"]["array"]
    processed = processor(
        audio,
        sampling_rate=32000,
        text=None,    # pure style finetuning
        return_tensors="pt"
    )
    return processed

ds = ds.map(preprocess, remove_columns=["audio"])

# ------------------------
# 4. Training Configuration
# ------------------------
training_args = TrainingArguments(
    output_dir="musicgen-ambient",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_steps=2000,
    fp16=True,
    save_steps=500,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
)

# ------------------------
# 5. Train
# ------------------------
trainer.train()

# ------------------------
# 6. Save Finetuned Model
# ------------------------
model.save_pretrained("musicgen-ambient")
processor.save_pretrained("musicgen-ambient")

print("Training complete!")
