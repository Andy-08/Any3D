import os
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn

import torchvision.transforms as T
import matplotlib.pyplot as plt

# Paths to model checkpoint and config
model_checkpoint_path = "model/pytorch_model.pth"
model_config_path = "model/co_dino_5scale_lsj_vit_large_lvis_instance.py"  # Config is now a Python file

# Load the config
def load_config(config_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Load the model
def load_model(checkpoint_path):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Perform instance segmentation
def instance_segmentation(model, image_path):
    transform = T.Compose([T.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    return image, predictions

# Visualize results
def visualize_results(image, predictions, threshold=0.5):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for i, (box, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['scores'])):
        if score > threshold:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
            ax.text(x1, y1, f"{score:.2f}", fontsize=10, color='red')

    plt.axis('off')
    plt.show()

# Main script
if __name__ == "__main__":
    # Load config
    if not os.path.exists(model_config_path):
        print(f"Config file not found at {model_config_path}")
        exit(1)

    config = load_config(model_config_path)

    # Load model
    if not os.path.exists(model_checkpoint_path):
        print(f"Checkpoint not found at {model_checkpoint_path}")
        exit(1)

    model = load_model(model_checkpoint_path)

    # Path to input image
    image_path = "img/11.jpg"  # Replace with your image path

    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        exit(1)

    # Perform instance segmentation
    image, predictions = instance_segmentation(model, image_path)

    # Visualize results
    visualize_results(image, predictions)