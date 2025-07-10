import argparse
import os
import re
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet50

from train import get_dataloader
from imagenet_c import corrupt
from PIL import Image
from tqdm import tqdm
import sys
cuda_device = torch.device('cuda', 0)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main(ckpts_root_path: Path, data_dir):
    models = [f'{ckpts_root_path}/{i}' for i in os.listdir(ckpts_root_path) if i.endswith('.pth')]

    print(f'\n\n{"=" * 80}\nEvaluating models on CAMELYON17.\n{"=" * 80}\n\n')
    print(f'Gotta run evals on {len(models)} models!')
    noise_list = ["shot_noise","gaussian_noise","impulse_noise","defocus_blur", "jpeg_compression", "motion_blur", "snow", "elastic_transform"]
    
    print("Running for all noise type: ", noise_list )
    for x in noise_list:
        print("---------------------------------------------------------------------------------------------")
        print(f'Running Evaluation for Noise: ', x)
        for m in models:
            run_eval(m, data_dir, x)


def run_eval(checkpoint_path, data_dir, x):
    config = {'DISTANCE_START_EPOCH': 1, 'TOTAL_EPOCHS': 0}
    print(f'Evaluating {checkpoint_path}', flush=True)
    hospital = int(re.search(r"(\d+)", Path(checkpoint_path).stem).group(1))
    model = resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0')['model'])
    model.to(cuda_device)
    model.eval()

    h0_dataloader = get_dataloader(data_dir, hospital=0, split=('val' if hospital == 0 else 'all'), batch_size=128,
                                   config=config, num_workers=4)
    h1_dataloader = get_dataloader(data_dir, hospital=1, split=('val' if hospital == 1 else 'all'), batch_size=128,
                                   config=config, num_workers=4)
    h2_dataloader = get_dataloader(data_dir, hospital=2, split=('val' if hospital == 2 else 'all'), batch_size=128,
                                   config=config, num_workers=4)
    h3_dataloader = get_dataloader(data_dir, hospital=3, split=('val' if hospital == 3 else 'all'), batch_size=128,
                                   config=config, num_workers=4)
    h4_dataloader = get_dataloader(data_dir, hospital=4, split=('val' if hospital == 4 else 'all'), batch_size=128,
                                   config=config, num_workers=4)

    print(f'Evaluating model on h0 on split {h0_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h0_dataloader, x)
    print(f'Evaluating on h0 took {timedelta(seconds=time.time() - start_time)}\n{"="*70}\n', flush=True)

    print(f'Evaluating model on h1 on split {h1_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h1_dataloader, x)
    print(f'Evaluating on h1 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)

    print(f'Evaluating model on h2 on split {h2_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h2_dataloader, x)
    print(f'Evaluating on h2 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)

    print(f'Evaluating model on h3 on split {h3_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h3_dataloader, x)
    print(f'Evaluating on h3 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)

    print(f'Evaluating model on h4 on split {h4_dataloader.dataset.split}', flush=True)
    start_time = time.time()
    evaluate_model(model, h4_dataloader, x)
    print(f'Evaluating on h4 took {timedelta(seconds=time.time() - start_time)}\n{"=" * 70}\n', flush=True)


def evaluate_model(model, dataloader, c_name="None"):
    print("Noise = ", c_name)
    confusion_mat = np.zeros((2, 2), dtype=int)  # Initialize confusion matrix
    total_skipped = 0
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        disable_progress = not sys.stdout.isatty()
        progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch", disable=disable_progress)  #  Add progress bar

        for batch_idx, data in enumerate(progress_bar):
            try:
                (x, _), labels, _ = data

                # Ensure correct shape: Convert to NumPy & Permute (CHW → HWC)
                x_np = x.permute(0, 2, 3, 1).cpu().numpy()  # (N, C, H, W) → (N, H, W, C)

                
                if x_np.max() <= 1.0:  
                    x_np = (x_np * 255).astype(np.uint8)

                x_np = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in x_np])

                #  Apply corruption
                x_np_corrupted = np.array([
                    corrupt(img, severity=2, corruption_name=c_name) for img in x_np
                ])

                
                x_np_corrupted = np.clip(x_np_corrupted, 0, 255).astype(np.uint8)

                
                x = torch.tensor(x_np_corrupted).permute(0, 3, 1, 2).to(cuda_device, dtype=torch.float32) / 255.0  

                labels = labels.unsqueeze(1).float()
                outputs = torch.sigmoid(model(x))
                preds = outputs > 0.5
                confusion_mat += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0, 1])

                #  Compute real-time accuracy
                acc = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
                progress_bar.set_postfix({"Accuracy": f"{100 * acc:.2f}%"})

            except Exception as e:
                print(f"[ERROR] Skipping batch {batch_idx} due to error: {e}")
                total_skipped += 1
                continue  # Skip problematic batch

    #  Final Accuracy Calculation
    acc = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)

    print("\nFinal Evaluation Results:")
    print(f'Final Accuracy: {100 * acc:.2f}%')
    print(pd.DataFrame(confusion_mat, index=['No Tumor', 'Tumor'], columns=['No Tumor', 'Tumor']))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval on all baseline models or other models.')
    parser.add_argument('--ckpts_path', default=r"E:\Dataset_Hikmat\SFL\ckpts\resnet50\a1", type=Path, help='Path to the checkpoint files.')
    parser.add_argument('--data_dir', default=r"E:\Dataset_Hikmat\SFL\dataset", type=str, help='Path to the CAMELYON17 data directory.')
    args = parser.parse_args()

    main(args.ckpts_path, args.data_dir)
