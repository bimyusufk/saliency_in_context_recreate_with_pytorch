import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils as vutils
from PIL import Image
from tqdm import tqdm

# --- IMPORT YOUR MODEL ---
# This assumes SaliconNet.py is in the same directory
from SaliconNet import SaliconNet

# --- CONFIG ---
BATCH_SIZE = 16 # Reduce to 8 or 4 if you get "Out of Memory" errors
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_HEAD = 1e-3
EPOCHS = 50 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILE = "resume.pth"

# --- 1. DATASET CLASS (Defined Here directly) ---
class SaliconDataset(Dataset):
    def __init__(self, img_dir, map_dir=None, mode='train'):
        self.img_dir = img_dir
        self.map_dir = map_dir
        self.mode = mode
        
        # Filter only valid images
        self.image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # VGG Normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
        # SALICON standard resolution
        self.target_h, self.target_w = 480, 640

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load Image
        image = Image.open(img_path).convert('RGB')
        
        # Resize for Fine Stream
        img_fine = image.resize((self.target_w, self.target_h), Image.BILINEAR)
        # Resize for Coarse Stream (1/2 size)
        img_coarse = img_fine.resize((self.target_w // 2, self.target_h // 2), Image.BILINEAR)
        
        # To Tensor & Normalize
        x_fine = self.normalize(transforms.ToTensor()(img_fine))
        x_coarse = self.normalize(transforms.ToTensor()(img_coarse))

        # Load Map (Ground Truth)
        if self.mode != 'test' and self.map_dir:
            # Handle extension mismatch (e.g. jpg image vs png map)
            map_name = os.path.splitext(img_name)[0] + '.png'
            map_path = os.path.join(self.map_dir, map_name)
            
            if os.path.exists(map_path):
                gt_map = Image.open(map_path).convert('L') # Grayscale
                gt_map = gt_map.resize((self.target_w, self.target_h), Image.BILINEAR)
                y = transforms.ToTensor()(gt_map)
            else:
                # Safe fallback
                y = torch.zeros((1, self.target_h, self.target_w))
            
            return x_fine, x_coarse, y
            
        return x_fine, x_coarse, img_name

# --- 2. LOSS FUNCTION (Defined Here directly) ---
class SaliconLoss(nn.Module):
    def __init__(self):
        super(SaliconLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, preds, targets):
        # preds: [B, 1, H, W] 
        # targets: [B, 1, H, W]
        
        # 1. Regularize Ground Truth to sum to 1 (Probability Distribution)
        B = preds.shape[0]
        targets_flat = targets.view(B, -1)
        targets_prob = targets_flat / (targets_flat.sum(dim=1, keepdim=True) + 1e-7)
        
        # 2. Log Softmax Prediction
        preds_flat = preds.view(B, -1)
        preds_log_prob = F.log_softmax(preds_flat, dim=1)
        
        # 3. Calculate Loss
        return self.kl(preds_log_prob, targets_prob)

# --- 3. CHECKPOINT SAVER ---
def save_checkpoint(model, optimizer, epoch, loss, filename):
    print(f"\n=> Saving checkpoint to {filename}...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, filename)
    print("=> Save Complete!")

# --- 4. MAIN TRAINING LOOP ---
def main():
    # Tensorboard
    writer = SummaryWriter(log_dir="runs/salicon_experiment_1")
    
    print(f"Starting Training on {DEVICE}")
    
    # Data & Model
    # ENSURE THESE PATHS MATCH YOUR FOLDER STRUCTURE EXACTLY
    train_ds = SaliconDataset('images/train', 'maps/train', mode='train')
    val_ds = SaliconDataset('images/val', 'maps/val', mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = SaliconNet().to(DEVICE)
    criterion = SaliconLoss()
    
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': LEARNING_RATE_BACKBONE},
        {'params': model.conv1x1.parameters(), 'lr': LEARNING_RATE_HEAD}
    ], momentum=0.9, weight_decay=0.0005)
    
    # Resume Logic
    start_epoch = 0
    if os.path.isfile(CHECKPOINT_FILE):
        print(f"=> Loading checkpoint '{CHECKPOINT_FILE}'")
        checkpoint = torch.load(CHECKPOINT_FILE)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> Resumed from Epoch {start_epoch}")
    else:
        print("=> No checkpoint found. Starting from scratch.")

    # Training Loop with Safety Catch
    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            train_loss_accum = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for batch_idx, (x_fine, x_coarse, y_true) in enumerate(loop):
                x_fine, x_coarse, y_true = x_fine.to(DEVICE), x_coarse.to(DEVICE), y_true.to(DEVICE)
                
                optimizer.zero_grad()
                y_pred = model(x_fine, x_coarse)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                
                train_loss_accum += loss.item()
                loop.set_postfix(loss=loss.item())
                
                # Tensorboard Log
                if batch_idx % 10 == 0:
                    writer.add_scalar("Loss/Train", loss.item(), epoch * len(train_loader) + batch_idx)

            # End of Epoch
            avg_train_loss = train_loss_accum / len(train_loader)
            
            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                # Visualization Batch
                viz_batch = next(iter(val_loader))
                vx_fine, vx_coarse, vy_true = viz_batch[0].to(DEVICE), viz_batch[1].to(DEVICE), viz_batch[2].to(DEVICE)
                vy_pred = model(vx_fine, vx_coarse) # Raw output
                
                for x_fine, x_coarse, y_true in val_loader:
                    x_fine, x_coarse, y_true = x_fine.to(DEVICE), x_coarse.to(DEVICE), y_true.to(DEVICE)
                    pred = model(x_fine, x_coarse)
                    val_loss_accum += criterion(pred, y_true).item()
            
            avg_val_loss = val_loss_accum / len(val_loader)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            
            print(f"Ep {epoch+1} | Val Loss: {avg_val_loss:.4f}")

            # Save Visualization (Apply Sigmoid for display so it looks like a heatmap)
            viz_pred = torch.sigmoid(vy_pred[0]) 
            viz_gt = vy_true[0]
            grid = vutils.make_grid([viz_pred, viz_gt], normalize=True, scale_each=True)
            writer.add_image('Prediction vs GT', grid, epoch)
            
            # Save Checkpoint
            save_checkpoint(model, optimizer, epoch, avg_val_loss, CHECKPOINT_FILE)

    except KeyboardInterrupt:
        print("\n\n!!! PAUSED BY USER (Ctrl+C) !!!")
        save_checkpoint(model, optimizer, epoch, 0.0, CHECKPOINT_FILE)
        print("Progress saved. Run script again to resume.")
        writer.close()
        exit(0)

    writer.close()

if __name__ == "__main__":
    main()