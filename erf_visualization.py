import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO


# =========================
# Load Image
# =========================
def load_image(path, img_size=320):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


# =========================
# Compute ERF Heatmap
# =========================
def compute_erf(model, layer, input_tensor):
    activation = None

    # Hook function
    def hook_fn(module, inp, out):
        nonlocal activation
        activation = out

    # Register hook
    handle = layer.register_forward_hook(hook_fn)

    # Forward
    input_tensor.requires_grad_(True)
    model(input_tensor)

    # Remove hook
    handle.remove()

    # Feature map shape: [1, C, H, W]
    _, C, H, W = activation.shape

    # Take center activation (sum over channels)
    center_activation = activation[0, :, H // 2, W // 2].sum()

    # Backward gradient
    model.zero_grad()
    center_activation.backward()

    # Gradient w.r.t input
    grad = input_tensor.grad.abs()  # [1,3,H,W]

    # Aggregate over RGB channels
    heatmap = grad.mean(dim=1).squeeze()

    # Normalize
    heatmap = heatmap / heatmap.max()
    return heatmap.detach().cpu().numpy()


# =========================
# Plot ERF for Backbone Layers
# =========================
def visualize_backbone_erf(weight_path, image_path, layers_to_extract):
    print("Loading model...")
    model = YOLO(weight_path).model
    model.eval()

    print("Loading image...")
    img = load_image(image_path).cuda()
    model.cuda()

    heatmaps = []

    print("Computing ERF for selected layers...")

    for idx in layers_to_extract:
        layer = model.model[idx]

        print(f"Extracting Layer index {idx}: {layer.__class__.__name__}")

        heatmap = compute_erf(model, layer, img.clone())
        heatmaps.append((idx, heatmap))

    # =========================
    # Plot grid
    # =========================
    n = len(layers_to_extract)
    plt.figure(figsize=(3*n, 4))

    for i, (idx, hm) in enumerate(heatmaps):
        plt.subplot(1, n, i + 1)
        plt.imshow(hm, cmap="YlGn")
        plt.axis("off")
        plt.title(f"Layer {idx}")

    plt.tight_layout()
    plt.show()



# =========================
# Run
# =========================
if __name__ == "__main__":
    weight_path = "E:/Data_KHOI/Project_YOLO/runs/segment/Experiment/POLIN/Compare_other/v8/weights/best.pt"     # or your trained model.pt
    image_path = "E:/Data_KHOI/Project_YOLO/datasets/hoa_test1/hoa3.jpg"       # input image
    # 🔥 Chỉ định các layer bạn muốn lấy receptive field
    layers_to_extract = [2, 4, 6, 8]

    visualize_backbone_erf(weight_path, image_path, layers_to_extract)
