import os
import shutil
import random

def split_data(source_dir, dest_dir, split_ratio=0.8):
    for label in ['stroke', 'nostroke']:
        files = os.listdir(os.path.join(source_dir, label))
        random.shuffle(files)

        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for phase, file_list in zip(['train', 'test'], [train_files, test_files]):
            out_dir = os.path.join(dest_dir, phase, label)
            os.makedirs(out_dir, exist_ok=True)
            for f in file_list:
                shutil.copy(
                    os.path.join(source_dir, label, f),
                    os.path.join(out_dir, f)
                )

split_data('data/raw', 'data', split_ratio=0.8)
