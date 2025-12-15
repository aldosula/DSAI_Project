import argparse
from src.model import PrototypicalNetwork
from src.features import FeatureExtractor
from src.inference import InferencePipeline
import torch
import os
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default="/data/msc-proj/Validation_Set_DSAI_2025_2026")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = PrototypicalNetwork(backbone='mobilenet_v3_small').to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    feature_extractor = FeatureExtractor().to(device)
    
    pipeline = InferencePipeline(model, feature_extractor, device=device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find wav files
    wav_files = glob.glob(os.path.join(args.val_dir, "**", "*.wav"), recursive=True)
    print(f"Found {len(wav_files)} validation files.")
    
    for wav_path in wav_files:
        csv_path = wav_path.replace(".wav", ".csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {wav_path} (no CSV found)")
            continue
            
        print(f"Processing {os.path.basename(wav_path)}...")
        output_csv = os.path.join(args.output_dir, os.path.basename(csv_path))
        try:
            pipeline.process_file(wav_path, csv_path, output_csv)
            print(f"Saved to {output_csv}")
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
        
if __name__ == "__main__":
    main()
