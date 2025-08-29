import argparse
import importlib
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from datasets.camelyon17 import Camelyon17DatasetWithMasks

cuda_device = torch.device('cuda', 0)
features = {}


def get_features(name):
    def hook(model, x, output):
        features[name] = output
    return hook


def get_dataloader(root_dir, hospital, split='train', batch_size=64, transform=None,
                   shuffle=False, config=None, num_workers=10):
    get_mask = not is_baseline_run(config)
    geometric_aug = split == 'train'
    affine_aug = False  # Not used, but included for future use

    dataset = Camelyon17DatasetWithMasks(
        root_dir=root_dir,
        hospital=hospital,
        split=split,
        transform=transform,
        get_mask=get_mask,
        geometric_aug=geometric_aug,
        affine_aug=affine_aug
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def import_config(config_path):
    spec = importlib.util.spec_from_file_location('config', config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def is_baseline_run(config):
    return config['DISTANCE_START_EPOCH'] > config['TOTAL_EPOCHS']


def train(config_path, train_hospital, data_dir):
    try:
        config = import_config(config_path)
    except FileNotFoundError:
        print(f'Error: File not found at {config_path}')
        sys.exit(1)

    config_name = Path(config_path).stem
    print(f'{config_name} Config:\n{config.config}')
    print(inspect.getsource(config.get_optimizer_and_scheduler))
    print(inspect.getsource(config.get_transforms))

    # Model setup
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    torch.nn.init.constant_(model.fc.weight, 1e-4)
    torch.nn.init.constant_(model.fc.bias, 0)
    model.layer4.register_forward_hook(get_features('layer4'))
    model.to(cuda_device)

    # Data loaders
    transform = config.get_transforms()
    dataloader = get_dataloader(data_dir, train_hospital, 'train', 64, transform, True, config.config)
    val_dataloader = get_dataloader(data_dir, train_hospital, 'val', 128, config=config.config)

    optimizer, scheduler = config.get_optimizer_and_scheduler(model)

    # --- SWA SETUP ---
    swa_start = config.config.get('SWA_START', 10)
    swa_lr = config.config.get('SWA_LR', 0.05)
    swa_freq = config.config.get('SWA_FREQ', 1)
    use_swa = config.config.get('USE_SWA', True)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    best_loss = float('inf')
    best_epoch = 0
    best_accuracy = 0

    compute_dist_loss = config.config['DISTANCE_START_EPOCH'] < config.config['TOTAL_EPOCHS']
    save_model_from_epoch = config.config['DISTANCE_START_EPOCH'] + 5
    model_save_dir = Path('ckpts') / 'resnet50'
    model_path = model_save_dir / f'{train_hospital}.pth'

    for epoch in range(config.config['TOTAL_EPOCHS']):
        epoch_start_time = time.time()

        if is_baseline_run(config.config):
            run_baseline_epoch(model, dataloader, epoch, optimizer, scheduler)
        else:
            run_epoch(model, config.config, dataloader, epoch, optimizer, scheduler)

        # SWA logic: Update after certain epoch
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        print(f'Epoch {epoch} took {timedelta(seconds=time.time() - epoch_start_time)}', flush=True)

        eval_start_time = time.time()
        val_accuracy, val_ce_loss, val_dist_loss = evaluate_model(model, val_dataloader, compute_dist_loss)
        total_val_loss = val_ce_loss + val_dist_loss
        print(f'Evaluating model after epoch {epoch} took {timedelta(seconds=time.time() - eval_start_time)}', flush=True)
        logging_dict = {
            'epoch': epoch,
            f'h{train_hospital}_val_ce_loss': val_ce_loss,
            f'h{train_hospital}_val_dist_loss': val_dist_loss,
            f'h{train_hospital}_val_total_loss': total_val_loss,
            f'h{train_hospital}_accuracy': val_accuracy,
        }
        print(logging_dict)

        if total_val_loss < best_loss:
            if not is_baseline_run(config.config) and epoch < save_model_from_epoch:
                print(f'Not saving the model yet: not a baseline run and current epoch={epoch} < {save_model_from_epoch}')
            else:
                best_loss = total_val_loss
                best_epoch = epoch
                best_accuracy = val_accuracy
                print(f'Saving model (best total val loss {total_val_loss})')
                model_path = save_checkpoint(f'{train_hospital}', model, model_save_dir)

    print(f'\n\nBest epoch: {best_epoch}, accuracy: {best_accuracy}\n')
    print('Model saved at:', model_path)

    # --- SWA Finalization ---
    if use_swa:
        print("Updating BatchNorm statistics for SWA model and evaluating...")
        update_bn(dataloader, swa_model, device=cuda_device)
        swa_acc, swa_loss, swa_dist = evaluate_model(swa_model, val_dataloader, compute_dist_loss)
        print(f"SWA Model Accuracy: {swa_acc:.4f}, Loss: {swa_loss:.4f}")
        swa_path = save_checkpoint(f'{train_hospital}_swa', swa_model, model_save_dir)
        print(f"SWA model saved at: {swa_path}")


def sup_contrast_with_mask(x_embeddings, mask_embeddings, margin=1.0):

    batch_size = x_embeddings.size(0)
    device = x_embeddings.device

    x_emb = F.normalize(x_embeddings, p=2, dim=1)
    mask_emb = F.normalize(mask_embeddings, p=2, dim=1)
    distances = torch.cdist(x_emb, mask_emb, p=2)

    pos_loss = torch.diag(distances).pow(2).mean()
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    neg_distances = distances[mask].view(batch_size, batch_size - 1)
    neg_loss = torch.clamp(margin - neg_distances, min=0.0).pow(2).mean()
    total_loss = 0.5 * (pos_loss + neg_loss)
    return total_loss


def sup_contrast_with_mask_cosine(x_embeddings, mask_embeddings, margin=0.3):

    batch_size = x_embeddings.size(0)
    device = x_embeddings.device

    x_norm = F.normalize(x_embeddings, p=2, dim=1)
    m_norm = F.normalize(mask_embeddings, p=2, dim=1)
    cos_sim = torch.mm(x_norm, m_norm.T)

    pos_loss = 1 - torch.diag(cos_sim)
    pos_loss = pos_loss.mean()

    mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    neg_sim = cos_sim[mask].view(batch_size, batch_size - 1)
    neg_loss = torch.clamp(neg_sim - margin, min=0).pow(2).mean()

    return 0.5 * (pos_loss + neg_loss)


def run_epoch(model, config, dataloader, epoch, optimizer, scheduler):
    dist_wt = config['DISTANCE_WEIGHT'] if epoch >= config['DISTANCE_START_EPOCH'] else 0.0
    model_loss = torch.nn.BCEWithLogitsLoss()
    for batch_idx, data in enumerate(dataloader, 0):
        (x, x_masks), labels, _ = data
        labels = labels.unsqueeze(1).to(cuda_device).float()

        x = x.to(cuda_device, memory_format=torch.channels_last)
        x_out = model(x)
        x_embeddings = features['layer4'].flatten(start_dim=1)
        x_loss = model_loss(x_out, labels)

        x_masks = x_masks.to(cuda_device, memory_format=torch.channels_last)
        mask_out = model(x_masks)
        mask_embeddings = features['layer4'].flatten(start_dim=1)
        masks_loss = model_loss(mask_out, labels)

        dist_loss_l4 = sup_contrast_with_mask(x_embeddings, mask_embeddings)
        total_loss = x_loss + masks_loss + (dist_wt * dist_loss_l4)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        if batch_idx % 100 == 0 or batch_idx == len(dataloader) - 1:
            print(f'RE:{batch_idx:04}\t{[round(l.item(), 6) for l in (x_loss, masks_loss, dist_loss_l4)]}', flush=True)
    # scheduler.step()  # SWA handles scheduler stepping


def run_baseline_epoch(model, dataloader, epoch, optimizer, scheduler):
    model_loss = torch.nn.BCEWithLogitsLoss()
    for batch_idx, data in enumerate(dataloader, 0):
        (x, _), labels, _ = data
        labels = labels.unsqueeze(1).to(cuda_device).float()

        x = x.to(cuda_device, memory_format=torch.channels_last)
        x_out = model(x)
        x_loss = model_loss(x_out, labels)

        x_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        if batch_idx % 100 == 0 or batch_idx == len(dataloader) - 1:
            print(f'BRE: {batch_idx:04}\t{round(x_loss.item(), 6)}', flush=True)
    # scheduler.step()  # SWA handles scheduler stepping


def save_checkpoint(file_name, model, root_dir):
    checkpoint = {'model': model.state_dict()}
    save_path = root_dir / f'{file_name}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    return save_path


def evaluate_model(model, dataloader, compute_dist_loss=False):
    model.eval()
    confusion_mat = np.zeros((2, 2), dtype=int)
    dist_loss_l4 = 0.
    all_scores = []
    all_labels = []
    distances = []
    bce_losses = []

    distance_fn = torch.nn.PairwiseDistance(p=2)
    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    with torch.no_grad():
        for data in dataloader:
            (x, x_masks), labels, _ = data
            x = x.to(cuda_device, memory_format=torch.channels_last)
            all_labels.append(labels.numpy())
            labels = labels.unsqueeze(1).float()

            outputs = model(x).cpu()
            all_scores.append(outputs.numpy())

            bce_losses.append(bce_loss_fn(outputs, labels))
            preds = torch.sigmoid(outputs) > 0.5
            confusion_mat += confusion_matrix(labels, preds, labels=[0, 1])

            if compute_dist_loss:
                x_embeddings = features['layer4'].flatten(start_dim=1)
                x_masks = x_masks.to(cuda_device, memory_format=torch.channels_last)
                model(x_masks)
                mask_embeddings = features['layer4'].flatten(start_dim=1)
                distances_l4 = distance_fn(x_embeddings, mask_embeddings)
                distances.append(distances_l4)

        acc = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
        print('Accuracy(mean): %.2f %%' % (100 * acc))
        print(pd.DataFrame(confusion_mat, index=['NoTumour', 'Tumour'], columns=['NoTumour', 'Tumour']))
    model.train()
    mean_bce_loss = torch.mean(torch.cat(bce_losses)).item()

    if compute_dist_loss and distances:
        distances = torch.cat(distances)
        dist_loss_l4 = mse_loss_fn(distances, torch.zeros_like(distances)).item()

    return acc, mean_bce_loss, dist_loss_l4


if __name__ == '__main__':
    import inspect  # Needed for printing config source code

    parser = argparse.ArgumentParser(description='Train a model on Camelyon17 with given config.')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--hospital', type=int, required=True, choices=range(5),
                        help='Hospital index (0-4) for training.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the CAMELYON17 data directory.')
    args = parser.parse_args()

    train(args.config_path, args.hospital, args.data_dir)
