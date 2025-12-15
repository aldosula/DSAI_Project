import os
import glob
import pandas as pd
import torch
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset

class BioacousticDataset(Dataset):
    def __init__(self, root_dir, n_way=5, n_support=5, n_query=5, segment_len=0.2, sample_rate=22050, transform=None, episodes=1000):
        """
        Args:
            root_dir: Path to Training_Set
            n_way: Number of classes per episode
            n_support: Number of support samples per class
            n_query: Number of query samples per class
            segment_len: Duration of audio segments in seconds
            sample_rate: Target sample rate
            transform: Feature extractor transform
            episodes: Number of episodes per epoch (since dataset is infinite)
        """
        self.root_dir = root_dir
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.transform = transform
        self.episodes = episodes
        
        self.class_data = {} # {class_code: [(wav_path, start, end), ...]}
        self.all_classes = []
        
        self._index_data()
        
    def _index_data(self):
        print(f"Indexing data from {self.root_dir}...")
        # Find all CSV files recursively
        csv_files = glob.glob(os.path.join(self.root_dir, "**", "*.csv"), recursive=True)
        
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                wav_path = csv_path.replace(".csv", ".wav")
                
                if not os.path.exists(wav_path):
                    continue
                    
                # Identify class columns (skip metadata)
                meta_cols = {'Audiofilename', 'Starttime', 'Endtime'}
                class_cols = [c for c in df.columns if c not in meta_cols]
                
                for cls in class_cols:
                    # Get positive events
                    pos_events = df[df[cls] == 'POS']
                    if pos_events.empty:
                        continue
                        
                    if cls not in self.class_data:
                        self.class_data[cls] = []
                        
                    for _, row in pos_events.iterrows():
                        self.class_data[cls].append((wav_path, row['Starttime'], row['Endtime']))
                        
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                
        # Filter classes with enough samples
        min_samples = self.n_support + self.n_query
        self.all_classes = [c for c, samples in self.class_data.items() if len(samples) >= min_samples]
        print(f"Found {len(self.all_classes)} classes with >= {min_samples} samples.")

    def __len__(self):
        return self.episodes

    def load_audio_segment(self, wav_path, start, end):
        # Load specific segment
        # Calculate frame offsets
        
        # We want a fixed segment_len centered on the event, or contained within it?
        # If event is long, pick a random window within it.
        # If event is short, center it.
        
        event_dur = end - start
        
        if event_dur > self.segment_len:
            # Random crop within event
            max_offset = event_dur - self.segment_len
            offset = random.random() * max_offset
            seg_start = start + offset
        else:
            # Center the event
            center = (start + end) / 2
            seg_start = center - (self.segment_len / 2)
            
        # Ensure non-negative
        seg_start = max(0, seg_start)
        
        # Load
        frame_offset = int(seg_start * self.sample_rate)
        num_frames = int(self.segment_len * self.sample_rate)
        
        try:
            # Get file info to check length (torchaudio 2.0+ compatible)
            metadata = torchaudio.backend.soundfile_backend.info(wav_path)
            file_frames = metadata.num_frames
            file_sr = metadata.sample_rate
            
            # Adjust if SR differs (we should probably resample after loading, but loading partial with different SR is tricky)
            # For simplicity, load a bit more and resample, or assume SR is close/same.
            # The prompt says "Use 22.05 kHz or 24 kHz".
            # Let's assume we load, then resample.
            
            # If we need to resample, we can't rely on frame indices directly mapping.
            # Safer: Load the whole file? No, too slow.
            # Load with some buffer?
            
            # Let's try to load the segment. If SR is different, we need to adjust offsets.
            orig_frame_offset = int(seg_start * file_sr)
            orig_num_frames = int(self.segment_len * file_sr)
            
            waveform, sr = torchaudio.load(wav_path, frame_offset=orig_frame_offset, num_frames=orig_num_frames, backend="soundfile")
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # Pad if necessary (e.g. if near end of file)
            if waveform.shape[1] < num_frames:
                pad_amt = num_frames - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
            else:
                waveform = waveform[:, :num_frames]
                
            return waveform
            
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(1, int(self.segment_len * self.sample_rate))

    def __getitem__(self, idx):
        # Sample N classes
        selected_classes = random.sample(self.all_classes, self.n_way)
        
        support_set = []
        query_set = []
        support_labels = []
        query_labels = []
        
        for i, cls in enumerate(selected_classes):
            samples = self.class_data[cls]
            # Sample K + Q
            selected_samples = random.sample(samples, self.n_support + self.n_query)
            
            # Split
            supports = selected_samples[:self.n_support]
            queries = selected_samples[self.n_support:]
            
            for s in supports:
                wav = self.load_audio_segment(*s)
                if self.transform:
                    wav = self.transform(wav)
                support_set.append(wav)
                support_labels.append(i)
                
            for q in queries:
                wav = self.load_audio_segment(*q)
                if self.transform:
                    wav = self.transform(wav)
                query_set.append(wav)
                query_labels.append(i)
                
        # Stack
        support_set = torch.stack(support_set)
        query_set = torch.stack(query_set)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)
        
        return support_set, support_labels, query_set, query_labels
