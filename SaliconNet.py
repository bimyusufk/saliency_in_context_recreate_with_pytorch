import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SaliconNet(nn.Module):
    def __init__(self):
        super(SaliconNet, self).__init__()
        
        # 1. LOAD BACKBONE (VGG-16)
        # Using the official PyTorch VGG16 weights (IMAGENET1K_V1)
        # This provides the robust feature extraction capabilities.
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 2. TRUNCATE VGG (Feature Extraction)
        # Cutting the VGG network at layer 30 ('conv5_3').
        # This keeps the spatial information before the final pooling layers flatten it too much.
        # The output of this section is 1/32 of the input size.
        self.features = nn.Sequential(*list(vgg16.features.children())[:30])
        
        # 3. INTEGRATION LAYER (The "Readout")
        # Fine Stream (512 channels) + Coarse Stream (512 channels) = 1024 channels
        # map 1024 channels -> 1 Channel (The Saliency Map)
        self.conv1x1 = nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x_fine, x_coarse):
        """
        Args:
            x_fine: Input image at standard resolution (e.g., 640x480)
            x_coarse: Input image downsampled by 0.5 (e.g., 320x240)
        """
        
        # --- STREAM 1: FINE SCALE ---
        # Extract features from the high-res image
        feat_fine = self.features(x_fine) 
        # Shape: [Batch, 512, H/32, W/32]
        
        # --- STREAM 2: COARSE SCALE ---
        # Extract features from the low-res image
        feat_coarse = self.features(x_coarse)
        # Shape: [Batch, 512, H/64, W/64]
        
        # --- FUSION PHASE ---
        # 1. Upsample Coarse features to match the spatial dimensions of Fine features
        feat_coarse_upsampled = F.interpolate(
            feat_coarse, 
            size=feat_fine.shape[2:], # Match (H_fine, W_fine)
            mode='bilinear', 
            align_corners=True
        )
        
        # 2. Concatenate along the Channel dimension (dim=1)
        # New Shape: [Batch, 1024, H/32, W/32]
        concat_feat = torch.cat((feat_fine, feat_coarse_upsampled), 1)
        
        # --- PREDICTION ---
        # Squash 1024 features down to 1 saliency activation map
        saliency_small = self.conv1x1(concat_feat)
        
        # --- FINAL RESTORATION ---
        # The map is currently 1/32 size (small). We must upsample it back to 
        # the original input size so the Loss Function compares pixels 1:1.
        output = F.interpolate(
            saliency_small, 
            size=x_fine.shape[2:], # Back to 640x480 (or whatever input was)
            mode='bilinear', 
            align_corners=True
        )
        
        return output

# --- QUICK TEST BLOCK ---
# Run this file directly specifically to check for errors: 'python SaliconNet.py'
if __name__ == "__main__":
    print("Testing SaliconNet Architecture...")
    model = SaliconNet()
    
    # Create dummy data: Batch 2, 3 channels, 640x480
    dummy_fine = torch.randn(2, 3, 480, 640)
    dummy_coarse = torch.randn(2, 3, 240, 320) # Half size
    
    try:
        out = model(dummy_fine, dummy_coarse)
        print(f"Success! Output Shape: {out.shape}")
        print("Expected: torch.Size([2, 1, 480, 640])")
    except Exception as e:
        print(f"Error: {e}")