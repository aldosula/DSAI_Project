import torch
import torchaudio
import pandas as pd
import numpy as np
import os
from scipy.signal import medfilt

class InferencePipeline:
    def __init__(self, model, feature_extractor, device='cuda', segment_len=0.2, hop_len=0.05, sample_rate=22050):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.segment_len = segment_len
        self.hop_len = hop_len
        self.sample_rate = sample_rate
        
    def load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path, backend="soundfile")
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform
        
    def extract_segment(self, waveform, start, end):
        # Extract segment, pad/crop to segment_len
        start_frame = int(start * self.sample_rate)
        end_frame = int(end * self.sample_rate)
        
        target_frames = int(self.segment_len * self.sample_rate)
        
        # If segment is provided (from shots), use it.
        # If it's longer than target, center crop?
        # If shorter, center pad.
        
        seg = waveform[:, start_frame:end_frame]
        
        if seg.shape[1] > target_frames:
            # Center crop
            center = seg.shape[1] // 2
            start_crop = center - (target_frames // 2)
            seg = seg[:, start_crop:start_crop+target_frames]
        elif seg.shape[1] < target_frames:
            # Center pad
            pad_amt = target_frames - seg.shape[1]
            left_pad = pad_amt // 2
            right_pad = pad_amt - left_pad
            seg = torch.nn.functional.pad(seg, (left_pad, right_pad))
            
        return seg

    def compute_prototypes(self, waveform, shots_df):
        # Positive prototype
        pos_segments = []
        for _, row in shots_df.iterrows():
            seg = self.extract_segment(waveform, row['Starttime'], row['Endtime'])
            pos_segments.append(seg)
            
        pos_batch = torch.stack(pos_segments).to(self.device)
        with torch.no_grad():
            pos_feats = self.feature_extractor(pos_batch)
            pos_embeds = self.model.encoder(pos_feats)
        
        pos_proto = pos_embeds.mean(dim=0)
        
        # Negative prototype(s)
        # Sample background segments
        # Avoid the shots
        duration = waveform.shape[1] / self.sample_rate
        neg_segments = []
        
        # Create a mask of occupied time
        mask = np.zeros(int(duration * 100)) # 10ms resolution
        for _, row in shots_df.iterrows():
            s = int(row['Starttime'] * 100)
            e = int(row['Endtime'] * 100)
            mask[s:e+1] = 1
            
        # Sample 10 negative segments
        attempts = 0
        while len(neg_segments) < 10 and attempts < 100:
            start_t = np.random.uniform(0, duration - self.segment_len)
            s_idx = int(start_t * 100)
            e_idx = int((start_t + self.segment_len) * 100)
            
            if np.sum(mask[s_idx:e_idx]) == 0:
                seg = self.extract_segment(waveform, start_t, start_t + self.segment_len)
                neg_segments.append(seg)
            attempts += 1
            
        if not neg_segments:
            # Fallback: just take random segments if we can't find clean ones (unlikely)
            pass
            
        neg_batch = torch.stack(neg_segments).to(self.device)
        with torch.no_grad():
            neg_feats = self.feature_extractor(neg_batch)
            neg_embeds = self.model.encoder(neg_feats)
            
        neg_proto = neg_embeds.mean(dim=0)
        
        return pos_proto, neg_proto

    def process_file(self, audio_path, shots_path, output_path=None):
        waveform = self.load_audio(audio_path)
        shots_df = pd.read_csv(shots_path)
        
        # Filter only POS shots if mixed
        if 'Q' in shots_df.columns:
            shots_df = shots_df[shots_df['Q'] == 'POS']
            
        pos_proto, neg_proto = self.compute_prototypes(waveform, shots_df)
        
        # Sliding window
        duration = waveform.shape[1] / self.sample_rate
        starts = np.arange(0, duration - self.segment_len, self.hop_len)
        
        segments = []
        for s in starts:
            seg = self.extract_segment(waveform, s, s + self.segment_len)
            segments.append(seg)
            
        # Batch processing
        batch_size = 64
        probs = []
        
        with torch.no_grad():
            for i in range(0, len(segments), batch_size):
                batch = torch.stack(segments[i:i+batch_size]).to(self.device)
                feats = self.feature_extractor(batch)
                embeds = self.model.encoder(feats)
                
                # Distance to pos and neg
                # embeds: (B, D)
                # pos_proto: (D,)
                # neg_proto: (D,)
                
                d_pos = torch.pow(embeds - pos_proto.unsqueeze(0), 2).sum(dim=1)
                d_neg = torch.pow(embeds - neg_proto.unsqueeze(0), 2).sum(dim=1)
                
                # Softmax
                # logits = [-d_pos, -d_neg]
                logits = torch.stack([-d_pos, -d_neg], dim=1)
                p = torch.softmax(logits, dim=1)
                
                probs.extend(p[:, 0].cpu().numpy()) # Prob of class 0 (Positive)
                
        probs = np.array(probs)
        
        # Post-processing
        # 1. Median filter
        probs_smooth = medfilt(probs, kernel_size=5)
        
        # 2. Threshold
        threshold = 0.5
        binary = probs_smooth > threshold
        
        # 3. Merge events
        events = []
        in_event = False
        start_event = 0
        
        for i, is_pos in enumerate(binary):
            if is_pos and not in_event:
                in_event = True
                start_event = starts[i]
            elif not is_pos and in_event:
                in_event = False
                end_event = starts[i] + self.segment_len # Approx end
                events.append((start_event, end_event))
                
        if in_event:
            events.append((start_event, starts[-1] + self.segment_len))
            
        # Filter events before 5th shot end time
        last_shot_end = shots_df['Endtime'].max()
        valid_events = [e for e in events if e[0] > last_shot_end]
        
        # Output CSV
        out_df = pd.DataFrame(valid_events, columns=['Starttime', 'Endtime'])
        out_df['Audiofilename'] = os.path.basename(audio_path)
        out_df['Q'] = 'POS'
        # Reorder
        out_df = out_df[['Audiofilename', 'Starttime', 'Endtime', 'Q']]
        
        if output_path:
            out_df.to_csv(output_path, index=False)
            
        return out_df
