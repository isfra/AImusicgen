from datasets import Dataset, Audio
import os

DATA_DIR = "dataset_32k"

files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".flac")]

data = {"audio": [os.path.join(DATA_DIR, f) for f in files]}

ds = Dataset.from_dict(data)
ds = ds.cast_column("audio", Audio(sampling_rate=32000))

ds.save_to_disk("ambient_dataset")
print("Dataset saved locally to ambient_dataset/")
