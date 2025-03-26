import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from tqdm import tqdm
import os
from PIL import Image
import random

# ---------------- MODEL ----------------
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)  # sigmoid added later in eval
        

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# ---------------- DATASET ----------------
class BDD100KDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Only include images that have corresponding mask files
        self.image_list = [
            img for img in os.listdir(image_dir)
            if os.path.exists(os.path.join(mask_dir, img.replace('.jpg', '.png')))
        ]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# ---------------- LOAD MODEL ----------------
device = torch.device("cpu")
model = UNet(in_channels=3, out_channels=1).to(device)

# Load the full model state dict
model.load_state_dict(torch.load(R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\model\best_unet_model.pth", map_location=device))
model.eval()

# ---------------- LOAD DATA ----------------
val_image_dir = r"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k\images\100k\val"
val_mask_dir = r"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k\labels\lane_masks\val"

full_dataset = BDD100KDataset(val_image_dir, val_mask_dir, transform)

# Evaluate on just 100 samples randomly
random_indices = random.sample(range(len(full_dataset)), 100)
val_subset = Subset(full_dataset, random_indices)
val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

# ---------------- EVALUATION ----------------
criterion = nn.BCEWithLogitsLoss()

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss, total_acc, total_f1, total_jaccard = 0, 0, 0, 0
    n = len(val_loader)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            true = (masks.cpu().numpy() > 0.5).astype(int)

            total_acc += accuracy_score(true.flatten(), preds.flatten())
            total_f1 += f1_score(true.flatten(), preds.flatten(), zero_division=1)
            total_jaccard += jaccard_score(true.flatten(), preds.flatten(), zero_division=1)

    print(f"\n✅ Validation Loss: {total_loss / n:.4f}")
    print(f"✅ Validation Accuracy: {total_acc / n:.4f}")
    print(f"✅ Validation F1 Score: {total_f1 / n:.4f}")
    print(f"✅ Validation Jaccard Score (IoU): {total_jaccard / n:.4f}")

# ---------------- RUN ----------------
evaluate(model, val_loader, criterion)
