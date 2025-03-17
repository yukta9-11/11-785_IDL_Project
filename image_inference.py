import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def load_model_from_checkpoint(checkpoint_path, device=None):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device, optional): Device to load the model to.
    
    Returns:
        model: The loaded model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' not in checkpoint:
        raise ValueError("Checkpoint does not contain a valid model.")
    
    model = checkpoint['model']
    model.to(device)
    model.eval()
    
    print(f"Model successfully loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model

class ImageDataset(Dataset):
    def _init_(self, image_dir, transform=None):
        """
        Custom dataset for loading images.
        
        Args:
            image_dir (str): Directory containing images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path

def process_images(model, image_dir, device):
    """
    Process images using the trained model.
    
    Args:
        model (nn.Module): Trained PyTorch model.
        image_dir (str): Directory containing input images.
        device (torch.device): Device for inference.
    
    Returns:
        results (dict): Dictionary mapping image paths to predictions.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)  # Process one image at a time
    
    results = {}
    
    with torch.no_grad():
        for images, paths in tqdm(dataloader):
            images = images.to(device)
            predictions = model(images)  # Forward pass
            
            # Convert predictions to numpy arrays or other formats as needed
            predictions_np = predictions.squeeze().cpu().numpy()
            
            results[paths[0]] = predictions_np
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run inference with a trained model.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images.')
    
    args = parser.parse_args()
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, device=device)
    
    # Process images and collect results
    results = process_images(model=model, image_dir=args.image_dir, device=device)
    
    # Print out results summary
    print("Processing complete. Summary:")
    for path in results.keys():
        print(f"Processed: {path}")

if _name_ == "_main_":
    main()
