import os
import ffmpeg

INPUT_DIR = "dataset_raw"
OUTPUT_DIR = "dataset_32k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if file.lower().endswith(".flac"):
        inp = os.path.join(INPUT_DIR, file)
        out = os.path.join(OUTPUT_DIR, file)

        print("Converting:", file)
        (
            ffmpeg
            .input(inp)
            .output(out, ar=32000, ac=2)
            .overwrite_output()
            .run(quiet=True)
        )

print("Conversion done!")
