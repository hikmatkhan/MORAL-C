import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

#from preprocess.OCELOT.generate_all_patch_coords import CENTER_SIZE
from generate_all_patch_coords import CENTER_SIZE


def write_patch_images_from_df(slide_root, output_root):
    # Load metadata
    metadata_path = os.path.join(slide_root, 'metadata.csv')
    df = pd.read_csv(metadata_path, index_col=0, dtype={'file_name': str, 'slide_name': str, 'organ': str, 'tumor': int})

    print("\nLoaded metadata:")
    print(df.head())  # Show first few rows for debugging

    center_size = CENTER_SIZE
    patch_size = center_size * 3

    for name, group in df.groupby('file_name'):
        print(f'\nProcessing file: {name}')

        # Construct image path
        image_path = os.path.join(slide_root, 'images', 'train', 'cell', f'{name}.jpg')
        print(f"Loading image: {image_path}")

        slide = cv2.imread(image_path)
        
        # Check if the image was loaded
        if slide is None:
            print(f"ERROR: Could not load image: {image_path}")
            continue

        # Show image details
        height, width, channels = slide.shape
        print(f"Image Loaded: {name} | Shape: {height}x{width} | Channels: {channels}")

        for idx in tqdm(group.index):
            orig_x = df.loc[idx, 'x_coord']
            orig_y = df.loc[idx, 'y_coord']

            # Compute patch coordinates
            x = orig_x - center_size
            y = orig_y - center_size

            print(f"\nPatch Coordinates: x={x}, y={y}, patch_size={patch_size}")

            # Ensure coordinates are within valid bounds
            if x < 0 or y < 0 or x + patch_size > width or y + patch_size > height:
                print(f"WARNING: Patch out of bounds for {name} at ({orig_x}, {orig_y}), skipping...")
                continue

            patch_folder = os.path.join(output_root, 'patches', name)
            patch_path = os.path.join(patch_folder, f'patch_{name}_x_{orig_x}_y_{orig_y}.jpg')

            os.makedirs(patch_folder, exist_ok=True)

            if os.path.isfile(patch_path):
                print(f"Skipping existing patch: {patch_path}")
                continue

            # Extract patch
            patch = slide[y: y + patch_size, x: x + patch_size]

            # Check if patch is valid
            if patch is None or patch.size == 0:
                print(f"WARNING: Empty patch at ({orig_x}, {orig_y}) for {name}, skipping...")
                continue

            # Save patch
            cv2.imwrite(patch_path, patch)
            print(f"Saved patch: {patch_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_root', default='./ocelot2023_v1.0.1/')
    parser.add_argument('--output_root', default='./process_data/')
    args = parser.parse_args()

    write_patch_images_from_df(slide_root=args.slide_root, output_root=args.output_root)