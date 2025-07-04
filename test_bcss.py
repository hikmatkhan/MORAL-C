import argparse
import os
import time
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from PIL import Image
from sklearn.metrics import confusion_matrix
from PIL import Image, UnidentifiedImageError

from datasets.bcss import BCSSDataset

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataloader(root_dir, batch_size=64, shuffle=False, num_workers=4):
    

    dataset = BCSSDataset(
        root_dir=root_dir
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def evaluate_model(model, dataloader):
    confusion_mat = np.zeros((2, 2), dtype=int)  # 2x2 Confusion Matrix for binary classification
    total_skipped = 0
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

        for data in progress_bar:
            # Ensure batch contains only valid data
            if data is None:
                total_skipped += 1
                continue  # Skip empty batch
            
            try:
                x, labels = data  # Unpacking like your second function
                x = x.to(cuda_device, memory_format=torch.channels_last)
                labels = labels.unsqueeze(1).float().to(cuda_device)  # Ensure labels are float
                
                # Model prediction
                outputs = torch.sigmoid(model(x))
                preds = (outputs > 0.5).cpu()  # Convert predictions to binary

                # Update confusion matrix
                confusion_mat += confusion_matrix(labels.cpu(), preds, labels=[0, 1])

                # Compute real-time accuracy
                acc = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
                progress_bar.set_postfix({"Accuracy": f"{100 * acc:.2f}%"})

            except Exception as e:
                print(f"[ERROR] Skipping batch due to error: {e}")
                total_skipped += 1
                continue  # Skip problematic batch

    # Final Accuracy Calculation
    acc = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)

    print("\nFinal Evaluation Results:")
    print(f'Final Accuracy: {100 * acc:.2f}%')
    print(pd.DataFrame(confusion_mat, index=['NoTumour', 'Tumour'], columns=['NoTumour', 'Tumour']))
    print(f"[INFO] Total skipped batches: {total_skipped}/{total_samples}")

def run_eval(checkpoint_path, image_folder):
    print(f"Evaluating model: {checkpoint_path}\n", flush=True)

    model = resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=cuda_device)['model'])
    model.to(cuda_device)
    model.eval()

    dataloader = get_dataloader(image_folder)

    start_time = time.time()
    evaluate_model(model, dataloader)
    print(f"Evaluation completed in {timedelta(seconds=time.time() - start_time)}\n{'=' * 70}\n")


def main(ckpts_root_path: Path, image_folder: str):
    models = [f'{ckpts_root_path}/{i}' for i in os.listdir(ckpts_root_path) if i.endswith('.pth')]

    print(f'\n\n{"=" * 80}\nEvaluating models on New Dataset.\n{"=" * 80}\n\n')
    print(f'Running evaluations on {len(models)} models...')
    for model_path in models:
        run_eval(model_path, image_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation on a new dataset.')
    parser.add_argument('--ckpts_path', default='./ckpts/Minloss1/', type=Path, help='Path to the checkpoint files.')
    parser.add_argument('--image_folder', default='./preprocess/BCSS/BCSS/', type=str, help='Path to the folder containing images.')
    args = parser.parse_args()

    main(args.ckpts_path, args.image_folder)
