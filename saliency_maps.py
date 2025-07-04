import torch
from torchvision.models import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from captum.attr import IntegratedGradients


cuda_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
checkpoint_path = r'E:\Dataset_Hikmat\SFL\ckpts\fianl\With_augmentation\contrastive_loss_cosine_c0\ckpts\resnet50\0.pth'
checkpoint = torch.load(checkpoint_path, map_location=cuda_device)
model.load_state_dict(checkpoint['model'])
model.to(cuda_device)
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = transform(img_resized)
    return img_tensor.unsqueeze(0), img_resized


output_dir = Path(r'E:\Dataset_Hikmat\SFL\Neworiginal_images_saliancy\class1')
output_dir.mkdir(parents=True, exist_ok=True)


image_dir = Path(r'E:\Dataset_Hikmat\SFL\Neworiginal_images\class_1')
image_paths = list(image_dir.glob('*.png'))

# Initialize Integrated Gradients
ig = IntegratedGradients(model)

for image_path in image_paths:
    input_tensor, input_image = load_image(image_path)
    input_tensor = input_tensor.to(cuda_device)

    
    baseline = torch.zeros_like(input_tensor).to(cuda_device)
    attr = ig.attribute(input_tensor, baseline, target=None, n_steps=50)
    attr = attr.squeeze().detach().cpu().numpy()
    attr_sum = np.sum(attr, axis=0)
    attr_norm = attr_sum / np.max(np.abs(attr_sum))

    # Positive and negative maps
    positive_attr = np.clip(attr_norm, 0, 1)
    negative_attr = np.clip(-attr_norm, 0, 1)

    
    gray_background = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    gray_background = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2RGB)

    # Positive map (green)
    positive_map = gray_background.copy()
    green_layer = np.zeros_like(gray_background)
    green_layer[:, :, 1] = (positive_attr * 255).astype(np.uint8)
    positive_map = cv2.addWeighted(positive_map, 1.0, green_layer, 0.8, 0)

    # Negative map (red)
    negative_map = gray_background.copy()
    red_layer = np.zeros_like(gray_background)
    red_layer[:, :, 0] = (negative_attr * 255).astype(np.uint8)
    negative_map = cv2.addWeighted(negative_map, 1.0, red_layer, 0.8, 0)

    
    save_path = output_dir / image_path.name
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(input_image)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    axs[1].imshow(positive_map)
    axs[1].axis('off')
    axs[1].set_title('Positive (Tumor) Evidence')

    axs[2].imshow(negative_map)
    axs[2].axis('off')
    axs[2].set_title('Negative (Non-Tumor) Evidence')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig) 
    print(f"Saved: {save_path}")



