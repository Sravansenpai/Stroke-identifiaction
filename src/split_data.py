import os
import shutil
import random

def train_test_split(data_dir, train_dir, test_dir, test_size=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    classes = os.listdir(data_dir)
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        
        files = os.listdir(os.path.join(data_dir, cls))
        random.shuffle(files)
        split_idx = int(len(files) * (1 - test_size))
        
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        for f in train_files:
            shutil.copy(os.path.join(data_dir, cls, f), os.path.join(train_dir, cls))
        for f in test_files:
            shutil.copy(os.path.join(data_dir, cls, f), os.path.join(test_dir, cls))

if __name__ == "__main__":
    # Use relative path for data directory
    data_dir = os.path.join('..', 'data', 'raw')

    train_dir = '../data/train'
    test_dir = '../data/test'
    train_test_split(data_dir, train_dir, test_dir)
    print("Train-test split done!")