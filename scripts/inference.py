from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('dark_background')

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # ✅ changed from self.conv

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

        return torch.sigmoid(self.final_conv(dec1))  # ✅ changed from self.conv

# Load the trained model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("model/best_unet_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Match training input size
    transforms.ToTensor()
])

# Preprocess image and mask
def preprocess(image_path, mask_path):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    original_size = image.size
    image = transform(image)
    mask = transform(mask)
    return image, mask, original_size

# Post-process prediction
def postprocess(output, original_size):
    output = output.squeeze().cpu().numpy()
    output = np.uint8(output * 255)
    output = Image.fromarray(output).resize(original_size, Image.BILINEAR)
    return np.array(output)

# Display image, mask, and prediction
def display_results(image_path, mask_path, predicted_mask):
    original_image = Image.open(image_path).convert('RGB')
    ground_truth_mask = Image.open(mask_path).convert('L')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(ground_truth_mask, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap="gray")
    plt.axis('off')

    plt.show()

# Run inference
def run_inference(image_path, mask_path):
    image, mask, original_size = preprocess(image_path, mask_path)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)

    predicted_mask = postprocess(output, original_size)

    save_folder = R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k\labels\lane_masks\New_folder"
    os.makedirs(save_folder, exist_ok=True)
    predicted_mask_image = Image.fromarray(predicted_mask)
    predicted_mask_image.save(os.path.join(save_folder, os.path.basename(mask_path)))

    display_results(image_path, mask_path, predicted_mask)

# Example usage
image_path = R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k\images\100k\val\b1c81faa-c80764c5.jpg"
mask_path = R"C:\Users\nishi\OneDrive\Desktop\Neu\Lane-Detection-UNet\data\bdd100k\labels\lane_masks\val\b1c81faa-c80764c5.png"
run_inference(image_path, mask_path)
