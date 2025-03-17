import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gdown
import zipfile

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
TIMESTEPS = 1000
LEARNING_RATE = 0.0001
BETA_START = 1e-4
BETA_END = 0.02

# Dataset Class
class RestorationDataset(Dataset):
    """
    Custom Dataset class for image restoration tasks.
    
    Loads pairs of clean and degraded images from separate directories.
    Each pair must have the same filename in their respective directories.
    """
    def _init_(self, clean_dir, degraded_dir, transform=None):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.transform = transform
        
        # Get all file names
        self.clean_images = sorted([f for f in os.listdir(clean_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.degraded_images = sorted([f for f in os.listdir(degraded_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Ensure the same number of clean and degraded images
        assert len(self.clean_images) == len(self.degraded_images), "Number of clean and degraded images don't match"
        
    def _len_(self):
        return len(self.clean_images)
    
    def _getitem_(self, idx):
        """
        Return a dictionary containing a pair of clean and degraded images.
        
        Args:
            idx (int): Index of the image pair to fetch
            
        Returns:
            dict: Contains 'clean' and 'degraded' image tensors
        """
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])
        degraded_path = os.path.join(self.degraded_dir, self.degraded_images[idx])
        
        clean_img = Image.open(clean_path).convert('RGB')
        degraded_img = Image.open(degraded_path).convert('RGB')
        
        if self.transform:
            clean_img = self.transform(clean_img)
            degraded_img = self.transform(degraded_img)
        
        return {'clean': clean_img, 'degraded': degraded_img}

# Download and prepare data from Google Drive
def download_and_extract_data(url, output_path='data.zip', extract_to='./data'):
    """
    Download and extract dataset from Google Drive URL.
    
    Args:
        url (str): Google Drive URL for the dataset
        output_path (str): Local path to save the downloaded zip file
        extract_to (str): Directory to extract the dataset
        
    Returns:
        tuple: Paths to clean and degraded image directories
    """
    # Download the zip file from Google Drive
    gdown.download(url, output_path, quiet=False)
    
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Data extracted to {extract_to}")
    
    # Return paths to clean and degraded image directories
    return os.path.join(extract_to, 'clean'), os.path.join(extract_to, 'degraded')

# U-Net Model for Diffusion
class UNet(nn.Module):
    """
    U-Net architecture for noise prediction in diffusion models.
    
    Features:
    - Time embedding for conditioning on diffusion timestep
    - Skip connections between encoder and decoder
    - Multiple resolution levels with downsampling and upsampling
    """
    def _init_(self, in_channels=3, out_channels=3, time_dim=256):
        super()._init_()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Encoder
        self.enc1 = self.make_conv_block(in_channels, 64)
        self.enc2 = self.make_conv_block(64, 128)
        self.enc3 = self.make_conv_block(128, 256)
        self.enc4 = self.make_conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.make_conv_block(512, 1024)
        self.time_proj1 = nn.Linear(time_dim, 1024)
        
        # Decoder
        self.dec4 = self.make_conv_block(1024 + 512, 512)
        self.time_proj2 = nn.Linear(time_dim, 512)
        self.dec3 = self.make_conv_block(512 + 256, 256)
        self.time_proj3 = nn.Linear(time_dim, 256)
        self.dec2 = self.make_conv_block(256 + 128, 128)
        self.time_proj4 = nn.Linear(time_dim, 128)
        self.dec1 = self.make_conv_block(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Downsampling and upsampling
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def make_conv_block(self, in_channels, out_channels):
        """
        Create a convolutional block with batch normalization and ReLU activation.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            
        Returns:
            nn.Sequential: A block of convolutional layers
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, t):
        """
        Forward pass through the U-Net model.
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            t (torch.Tensor): Timestep tensor [B]
            
        Returns:
            torch.Tensor: Predicted noise
        """
        # Time embedding
        t = self.time_mlp(t)
        
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.down(x1))
        x3 = self.enc3(self.down(x2))
        x4 = self.enc4(self.down(x3))
        
        # Bottleneck
        x5 = self.bottleneck(self.down(x4))
        x5 = x5 + self.time_proj1(t)[..., None, None]
        
        # Decoder with skip connections
        x = self.up(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        x = x + self.time_proj2(t)[..., None, None]
        
        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        x = x + self.time_proj3(t)[..., None, None]
        
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = x + self.time_proj4(t)[..., None, None]
        
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        # Final layer
        return self.final(x)

# Sinusoidal position embedding for time
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    
    Based on the positional encoding from "Attention Is All You Need" paper.
    """
    def _init_(self, dim):
        super()._init_()
        self.dim = dim

    def forward(self, time):
        """
        Compute sinusoidal embeddings for timesteps.
        
        Args:
            time (torch.Tensor): Timestep tensor [B]
            
        Returns:
            torch.Tensor: Embeddings [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings

# Diffusion Model
class DiffusionModel:
    """
    Implementation of a denoising diffusion probabilistic model (DDPM).
    
    Handles the forward and reverse diffusion processes, including:
    - Adding noise gradually to images (forward process)
    - Removing noise step by step (reverse process)
    - Training the denoising network
    - Image restoration through partial denoising
    """
    def _init_(self, timesteps=TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END):
        """
        Initialize the diffusion model with a noise schedule.
        
        Args:
            timesteps (int): Number of diffusion steps
            beta_start (float): Starting noise level
            beta_end (float): Ending noise level
        """
        self.timesteps = timesteps
        
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        
        # Pre-calculate different terms for diffusion process
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to the image.
        
        Args:
            x_start (torch.Tensor): Clean image tensor [B, C, H, W]
            t (torch.Tensor): Timestep tensor [B]
            noise (torch.Tensor, optional): Noise to add
            
        Returns:
            torch.Tensor: Noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # Add noise according to the diffusion equation
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """
        Sample from p(x_{t-1} | x_t) - one step of denoising.
        
        Args:
            model (nn.Module): Noise prediction model
            x (torch.Tensor): Noisy image at timestep t
            t (torch.Tensor): Current timestep indices
            t_index (int): Current timestep index (for special handling of t=0)
            
        Returns:
            torch.Tensor: Denoised image at timestep t-1
        """
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = model(x, t)
        
        # Mean for p(x_{t-1} | x_t)
        mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # No noise if t == 0
        if t_index == 0:
            return mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """
        Generate samples from the model using the reverse diffusion process.
        
        Args:
            model (nn.Module): Noise prediction model
            shape (tuple): Shape of the output image batch [B, C, H, W]
            
        Returns:
            torch.Tensor: Generated images
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        # Iterate through all timesteps
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            
        return img

    @torch.no_grad()
    def restore_image(self, model, x_degraded, strength=0.75):
        """
        Restore a degraded image by starting from a partially noised version and running denoising.
        
        Args:
            model (nn.Module): Noise prediction model
            x_degraded (torch.Tensor): Degraded image tensor [B, C, H, W]
            strength (float): Amount of noise to add (0.0 = no change, 1.0 = full noise)
            
        Returns:
            torch.Tensor: Restored image
        """
        device = next(model.parameters()).device
        b = x_degraded.shape[0]
        
        # Determine the timestep to start from based on the strength parameter
        t_start = int(self.timesteps * strength)
        
        # Add noise to the degraded image to reach the start timestep
        t_tensor = torch.tensor([t_start] * b, device=device)
        noise = torch.randn_like(x_degraded)
        x_noisy = self.q_sample(x_degraded, t_tensor, noise=noise)
        
        # Iteratively denoise starting from t_start
        x = x_noisy
        for i in tqdm(reversed(range(0, t_start)), desc='Restoration timestep', total=t_start):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)
            
        return x

    def train_step(self, model, optimizer, clean_images, degraded_images=None):
        """
        Single training step for the diffusion model.
        
        Args:
            model (nn.Module): Noise prediction model
            optimizer (torch.optim.Optimizer): Optimizer
            clean_images (torch.Tensor): Clean image batch
            degraded_images (torch.Tensor, optional): Degraded image batch (unused in this implementation)
            
        Returns:
            float: Training loss value
        """
        optimizer.zero_grad()
        batch_size = clean_images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Add noise to the clean images
        noise = torch.randn_like(clean_images)
        x_noisy = self.q_sample(clean_images, t, noise=noise)
        
        # Predict noise
        predicted_noise = model(x_noisy, t)
        
        # Simple mean squared error loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        return loss.item()

# Training loop
def train_diffusion_model(model, diffusion, train_loader, val_loader, optimizer, epochs, device):
    """
    Train the diffusion model.
    
    Args:
        model (nn.Module): Noise prediction model
        diffusion (DiffusionModel): Diffusion process controller
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimizer
        epochs (int): Number of training epochs
        device (torch.device): Device to use for training
        
    Returns:
        tuple: Training and validation loss history
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            clean_images = batch['clean'].to(device)
            
            # Train on clean images (learning to denoise)
            batch_loss = diffusion.train_step(model, optimizer, clean_images)
            
            train_loss += batch_loss
            train_batches += 1
            progress_bar.set_postfix({"loss": batch_loss})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                clean_images = batch['clean'].to(device)
                
                # Sample random timesteps
                t = torch.randint(0, diffusion.timesteps, (clean_images.shape[0],), device=device, dtype=torch.long)
                
                # Add noise to the clean images
                noise = torch.randn_like(clean_images)
                x_noisy = diffusion.q_sample(clean_images, t, noise=noise)
                
                # Predict noise
                predicted_noise = model(x_noisy, t)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_diffusion_model.pth')
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
            
        # Save a generated sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_and_save(model, diffusion, val_loader, epoch, device)
    
    return train_losses, val_losses

# Generate and save samples
@torch.no_grad()
def sample_and_save(model, diffusion, val_loader, epoch, device):
    """
    Generate and save sample restoration results.
    
    Args:
        model (nn.Module): Noise prediction model
        diffusion (DiffusionModel): Diffusion process controller
        val_loader (DataLoader): Validation data loader
        epoch (int): Current epoch number
        device (torch.device): Device to use
    """
    model.eval()
    
    # Get a batch from validation set
    batch = next(iter(val_loader))
    clean_images = batch['clean'].to(device)
    degraded_images = batch['degraded'].to(device)
    
    # Restore degraded images
    restored_images = diffusion.restore_image(model, degraded_images)
    
    # Create a grid of images for comparison
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(4):
        # Original clean images
        clean_img = clean_images[i].cpu().permute(1, 2, 0).numpy()
        clean_img = np.clip(clean_img, 0, 1)
        axes[0, i].imshow(clean_img)
        axes[0, i].set_title('Clean')
        axes[0, i].axis('off')
        
        # Degraded images
        degraded_img = degraded_images[i].cpu().permute(1, 2, 0).numpy()
        degraded_img = np.clip(degraded_img, 0, 1)
        axes[1, i].imshow(degraded_img)
        axes[1, i].set_title('Degraded')
        axes[1, i].axis('off')
        
        # Restored images
        restored_img = restored_images[i].cpu().permute(1, 2, 0).numpy()
        restored_img = np.clip(restored_img, 0, 1)
        axes[2, i].imshow(restored_img)
        axes[2, i].set_title('Restored')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'samples_epoch_{epoch+1}.png')
    plt.close()

# Evaluation function (visual only)
def evaluate_model(model, diffusion, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model (nn.Module): Noise prediction model
        diffusion (DiffusionModel): Diffusion process controller
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Count successful restorations
    total_images = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            clean_images = batch['clean'].to(device)
            degraded_images = batch['degraded'].to(device)
            
            # Restore degraded images
            restored_images = diffusion.restore_image(model, degraded_images)
            
            # Count processed images
            total_images += clean_images.shape[0]
    
    # Print results
    print("\nEvaluation Complete:")
    print(f"Successfully processed {total_images} images")
    
    return {
        'total_images': total_images
    }

# Main function to run the entire diffusion model training and evaluation pipeline
def run_diffusion_pipeline(google_drive_url, epochs=EPOCHS):
    """
    Main function to run the entire diffusion model training and evaluation pipeline.
    
    Args:
        google_drive_url (str): Google Drive URL for the dataset
        epochs (int): Number of training epochs
    """
    # Download and extract data
    clean_dir, degraded_dir = download_and_extract_data(google_drive_url)
    
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and split into train, validation, and test
    dataset = RestorationDataset(clean_dir, degraded_dir, transform=transform)
    
    # Split dataset: 70% train, 15% validation, 15% test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Dataset split - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
    
    # Initialize model and diffusion process
    model = UNet().to(device)
    diffusion = DiffusionModel()
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_diffusion_model(
        model, diffusion, train_loader, val_loader, optimizer, epochs, device
    )
    
    # Load the best model
    checkpoint = torch.load('best_diffusion_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.6f}")
    
    # Evaluate the model visually
    print("Evaluating model on test set...")
    evaluate_model(model, diffusion, test_loader, device)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    
    print("Training and evaluation complete!")
    print(f"Check the generated samples in the current directory.")

# Entry point for the script
if _name_ == "_main_":
    # Replace this URL with your Google Drive URL containing the dataset
    GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    
    # Run the full pipeline with fewer epochs for testing
    run_diffusion_pipeline(GOOGLE_DRIVE_URL, epochs=50)  # Adjust epochs as needed