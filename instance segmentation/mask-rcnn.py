import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os

# 加载预训练的 Mask R-CNN 模型 (使用 ResNet-50 backbone)
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()

# 定义图像预处理步骤
transform = T.Compose([T.ToTensor()])

# Define input and output directories
input_dir = "./img"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        continue

    img_tensor = transform(image)
    images = [img_tensor]

    # Perform inference
    with torch.no_grad():
        prediction = model(images)

    # Extract prediction results
    boxes = prediction[0]['boxes'].cpu().numpy().astype(np.int32)
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()

    # Filter results based on confidence threshold
    confidence_threshold = 0.4
    filtered_indices = np.where(scores > confidence_threshold)[0]
    filtered_boxes = boxes[filtered_indices]
    filtered_masks = masks[filtered_indices]
    filtered_labels = labels[filtered_indices]
    filtered_scores = scores[filtered_indices]

    # COCO dataset label names
    COCO_LABELS = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    # Update the visualization function to use predefined mask colors
    def visualize_and_save(image, masks, labels, scores, boxes, confidence_threshold=0.5, output_path="output.png"):
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        ax = plt.gca()

        # Predefined colors for masks
        predefined_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        num_colors = len(predefined_colors)

        for i, (mask, label, score, box) in enumerate(zip(masks, labels, scores, boxes)):
            if score < confidence_threshold:
                continue

            # Overlay the mask with predefined colors
            mask = mask[0] > 0.5  # Convert mask to binary
            color = predefined_colors[i % num_colors]  # Cycle through predefined colors
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            colored_mask[mask] = color
            plt.imshow(np.dstack((colored_mask, mask * 0.7)), alpha=0.6)  # Increased transparency for distinction

            # Draw the bounding box
            x_min, y_min, x_max, y_max = box
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2,
                                  edgecolor=np.array(color) / 255.0, facecolor='none')
            ax.add_patch(rect)

            # Add the label and score
            label_name = COCO_LABELS[label] if label < len(COCO_LABELS) else f"Label {label}"
            ax.text(x_min, y_min - 10, f"{label_name}: {score:.2f}", color='white', fontsize=12,
                    bbox=dict(facecolor=np.array(color) / 255.0, alpha=0.7))  # Adjusted alpha for better readability

        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Visualization saved to {output_path}")

    # Convert the image tensor back to a PIL image for visualization
    image_np = np.array(image)

    # Save the visualized result
    output_path = os.path.join(output_dir, f"visualized_{image_name}")
    visualize_and_save(image_np, filtered_masks, filtered_labels, filtered_scores, filtered_boxes, confidence_threshold, output_path)

print(f"All visualized images are saved in {output_dir}")