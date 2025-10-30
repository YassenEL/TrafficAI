import os, random, shutil, pathlib

random.seed(42)

RAW_FRAMES = pathlib.Path("dataset/raw_frames")
TRAIN_IMG = pathlib.Path("dataset/train/images")
VAL_IMG = pathlib.Path("dataset/val/images")
TRAIN_IMG.mkdir(parents=True, exist_ok=True)
VAL_IMG.mkdir(parents=True, exist_ok=True)

images = [p for p in RAW_FRAMES.glob("*.*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
random.shuffle(images)

split_ratio = 0.8
split_index = int(len(images) * split_ratio)

for i, img_path in enumerate(images):
    dest = TRAIN_IMG if i < split_index else VAL_IMG
    shutil.copy(str(img_path), str(dest / img_path.name))

print(f"Moved {split_index} to train and {len(images)-split_index} to val.")
