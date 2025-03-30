import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Import the dataset class and model
from dataset import SinogramDataset
from model import UNet  # or import LighterUNet if preferred

def load_model(checkpoint_path, device):
    # Create the model instance (n_channels=3 for 3-channel input, n_classes=3 for 3-channel output)
    model = UNet(n_channels=3, n_classes=3, bilinear=False, attention=True, pretrain=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Assuming the checkpoint dictionary contains 'model_state_dict'
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def convert_incomplete_to_predicted(data_dir, subset, checkpoint_path, output_dir, device):
    # data_dir should be the base folder containing subfolders (e.g., 'train' and 'test')
    subset_dir = os.path.join(data_dir, subset)
    
    # Instantiate the dataset (the 'test' flag can be set according to your use case)
    is_train = True if subset.lower() == 'train' else False
    dataset = SinogramDataset(subset_dir, is_train=is_train, transform=None, test=True)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = load_model(checkpoint_path, device)
    
    # Process each sample
    print("Converting incomplete sinograms to predicted complete sinograms...")
    for idx in tqdm(range(len(dataset))):
        # Get the 3-channel input (and ignore the complete one)
        incomplete_3ch, _ = dataset[idx]  # shape: (3, H, W)
        
        # Retrieve the file identification from dataset.pairs
        i, j = dataset.pairs[idx]
        # Keep the same filename as the incomplete image
        filename = f"incomplete_{i}_{j}.npy"
        output_path = os.path.join(output_dir, filename)
        
        # Prepare input tensor: add batch dimension and send to device
        input_tensor = incomplete_3ch.unsqueeze(0).to(device)  # shape: (1, 3, H, W)
        
        with torch.no_grad():
            # Get model prediction; output shape is assumed to be (1, 3, H, W)
            output = model(input_tensor)
        
        # As in training/evaluation, the middle channel (index 1) is the predicted complete image.
        predicted_complete = output[0, 1].cpu().numpy()
        
        # Save the predicted image with the same filename
        np.save(output_path, predicted_complete)
    
    print(f"Conversion complete. Predicted images are saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert all incomplete sinogram images into predicted complete images."
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the base data directory (should contain subfolders like 'train' and 'test').")
    parser.add_argument('--subset', type=str, default='test', choices=['train', 'test'],
                        help="Which subset to process ('train' or 'test').")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the trained model checkpoint (should include 'model_state_dict').")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the predicted complete images.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (e.g., 'cuda' or 'cpu').")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    convert_incomplete_to_predicted(args.data_dir, args.subset, args.checkpoint, args.output_dir, device)

if __name__ == '__main__':
    main()
