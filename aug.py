import os
import cv2
import numpy as np
import random

# ==========================================
#             LOAD / SAVE YOLO
# ==========================================
def load_label(path):
    bboxes, labels = [], []
    if not os.path.exists(path):
        return bboxes, labels
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            c, x, y, w, h = map(float, parts)
            labels.append(int(c))
            bboxes.append([x, y, w, h])
    return bboxes, labels

def save_label(path, bboxes, labels):
    with open(path, "w") as f:
        for (x, y, w, h), c in zip(bboxes, labels):
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# ==========================================
#               CLEAN BBOX
# ==========================================
def clamp_bboxes(bboxes, labels):
    new_b, new_l = [], []
    for (x, y, w, h), c in zip(bboxes, labels):
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        w = np.clip(w, 0, 1)
        h = np.clip(h, 0, 1)
        if w <= 0 or h <= 0:
            continue
        new_b.append([x, y, w, h])
        new_l.append(c)
    return new_b, new_l

# ==========================================
#              SIMPLE AUG
# ==========================================
def simple_aug(img, bboxes, labels, h_flip=True, v_flip=True, brightness=True):
    ih, iw = img.shape[:2]
    b = [list(box) for box in bboxes]
    l = list(labels)
    aug_img = img.copy()

    # Horizontal flip
    if h_flip and random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        for i in range(len(b)):
            b[i][0] = 1 - b[i][0]  # flip x center

    # Vertical flip
    if v_flip and random.random() < 0.5:
        aug_img = cv2.flip(aug_img, 0)
        for i in range(len(b)):
            b[i][1] = 1 - b[i][1]  # flip y center

    # Brightness / Contrast
    if brightness and random.random() < 0.5:
        alpha = random.uniform(0.3, 1.8)  # contrast
        beta = random.randint(-30, 80)    # brightness
        aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    b, l = clamp_bboxes(b, l)
    return aug_img, b, l

# ==========================================
#                 MOSAIC 4
# ==========================================
def mosaic_4(img_paths, label_paths, size=1024):
    w = h = size
    xc = random.randint(int(0.25*w), int(0.75*w))
    yc = random.randint(int(0.25*h), int(0.75*h))

    final_img = np.full((h, w, 3), 114, dtype=np.uint8)
    out_bboxes, out_labels = [], []

    placements = [(0,0,xc,yc),(xc,0,w,yc),(0,yc,xc,h),(xc,yc,w,h)]

    for i in range(4):
        img = cv2.imread(img_paths[i])
        if img is None:
            continue
        ih, iw = img.shape[:2]
        bboxes, labels = load_label(label_paths[i])
        x1a, y1a, x2a, y2a = placements[i]
        pw, ph = x2a - x1a, y2a - y1a

        scale = max(pw/iw, ph/ih)
        nw, nh = int(iw*scale), int(ih*scale)
        img_rs = cv2.resize(img, (nw, nh))

        x1b = random.randint(0, max(0, nw-pw))
        y1b = random.randint(0, max(0, nh-ph))
        crop = img_rs[y1b:y1b+ph, x1b:x1b+pw]
        final_img[y1a:y2a, x1a:x2a] = crop

        for (cx, cy, bw, bh), lab in zip(bboxes, labels):
            cx *= nw; cy *= nh; bw *= nw; bh *= nh
            cx -= x1b; cy -= y1b
            if not (0<cx<pw and 0<cy<ph): 
                continue
            cx += x1a; cy += y1a
            out_bboxes.append([cx/w, cy/h, bw/w, bh/h])
            out_labels.append(lab)

    out_bboxes, out_labels = clamp_bboxes(out_bboxes, out_labels)
    return final_img, out_bboxes, out_labels

# ==========================================
#                   MAIN
# ==========================================
input_img_dir = r"E:/Data_KHOI/Project_YOLO/datasets/GlobalWheat2020/aug/images/arvalis_1"
input_lbl_dir = r"E:/Data_KHOI/Project_YOLO/datasets/GlobalWheat2020/aug/labels/arvalis_1"

out_root = input_img_dir + "_AUG"
out_img_dir = os.path.join(out_root, "images")
out_lbl_dir = os.path.join(out_root, "labels")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith(".png")]
AUG_TIMES = 2

for k in range(AUG_TIMES):
    print(f"\n===== AUGMENT {k} =====")
    for img_name in image_files:
        img_path = os.path.join(input_img_dir, img_name)
        lbl_path = os.path.join(input_lbl_dir, img_name.replace(".png", ".txt"))

        img = cv2.imread(img_path)
        b, l = load_label(lbl_path)

        # SIMPLE AUG
        img_aug, b_aug, l_aug = simple_aug(img, b, l)
        out_name = f"{img_name[:-4]}_aug{k}.jpg"
        cv2.imwrite(os.path.join(out_img_dir, out_name), img_aug)
        save_label(os.path.join(out_lbl_dir, out_name[:-4]+".txt"), b_aug, l_aug)

        # MOSAIC
        chosen = random.sample(image_files, 4)
        mosaic_imgs = [os.path.join(input_img_dir, x) for x in chosen]
        mosaic_lbls = [os.path.join(input_lbl_dir, x.replace(".png", ".txt")) for x in chosen]
        img_mos, b_mos, l_mos = mosaic_4(mosaic_imgs, mosaic_lbls)
        out_name = f"{img_name[:-4]}_mosaic{k}.jpg"
        cv2.imwrite(os.path.join(out_img_dir, out_name), img_mos)
        save_label(os.path.join(out_lbl_dir, out_name[:-4]+".txt"), b_mos, l_mos)

print("\n🔥 DONE! Dataset đã augment xong.")
