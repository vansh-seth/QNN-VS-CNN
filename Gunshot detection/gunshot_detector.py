import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import librosa
import sounddevice as sd
import threading
import time
import os
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, duration=2.0, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.target_length = int(sample_rate * duration)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def load_audio(self, file_path):
        try:
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Normalize audio
            if torch.max(torch.abs(waveform)) > 0:
                waveform = waveform / torch.max(torch.abs(waveform))
            
            # Pad or truncate to target length
            if waveform.shape[1] > self.target_length:
                waveform = waveform[:, :self.target_length]
            elif waveform.shape[1] < self.target_length:
                pad_length = self.target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            return waveform
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_features(self, waveform):
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Improved normalization
        if mel_spec_db.std() > 1e-8:
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        else:
            mel_spec_db = mel_spec_db - mel_spec_db.mean()
        
        return mel_spec_db
    
    def calculate_spl(self, waveform):
        rms = torch.sqrt(torch.mean(waveform**2))
        spl = 20 * torch.log10(rms + 1e-8)
        return spl.item()

class GunshotDataset(Dataset):
    def __init__(self, data_dir, preprocessor, labels_file=None):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.samples = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        self._load_dataset()
        
        # Only fit label encoder if we have labels
        if self.labels:
            self.labels = self.label_encoder.fit_transform(self.labels)
        
    def _load_dataset(self):
        print("Loading dataset...")
        # Simplified label mapping - group similar weapons
        label_mapping = {
            'ak12': 'assault_rifle',
            'aug': 'assault_rifle', 
            'ak47': 'assault_rifle',
            'g36c': 'assault_rifle',
            'm4': 'assault_rifle',
            'm16': 'assault_rifle',
            'scar': 'assault_rifle',
            'awm': 'sniper_rifle',
            'kar': 'sniper_rifle',
            'sks': 'sniper_rifle',
            'slr': 'sniper_rifle',
            'mp5': 'submachine_gun',
            'p90': 'submachine_gun',
            'dbs': 'shotgun',
            's12k': 'shotgun',
            'remington_870_12_gauge': 'shotgun',
            'deagel': 'pistol',
            'p18': 'pistol',
            'p92': 'pistol',
            'glock_17_9mm_caliber': 'pistol',
            '38s&ws_dot38_caliber': 'pistol',
            'zastavam92': 'pistol',
            'noise': 'non_gunshot',
            'urban': 'non_gunshot',
            'background': 'non_gunshot'
        }
        
        for audio_file in self.data_dir.rglob('*.wav'):
            label = 'non_gunshot'  # default
            filename = audio_file.stem.lower()
            parent_folder = audio_file.parent.name.lower()
            
            # Check for matches in filename and parent folder
            for key, mapped_label in label_mapping.items():
                if key in filename or key in parent_folder:
                    label = mapped_label
                    break
            
            self.samples.append(audio_file)
            self.labels.append(label)
        
        print(f"Loaded {len(self.samples)} samples")
        if self.labels:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"Label distribution: {dict(zip(unique_labels, counts))}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_file = self.samples[idx]
        label = self.labels[idx]
        
        waveform = self.preprocessor.load_audio(audio_file)
        if waveform is None:
            waveform = torch.zeros(1, self.preprocessor.target_length)
        
        features = self.preprocessor.extract_features(waveform)
        return features, label

class GunshotClassifier(nn.Module):
    def __init__(self, num_classes=6, n_mels=128):
        super(GunshotClassifier, self).__init__()
        
        # Improved CNN architecture
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Distance estimation head
        self.distance_estimator = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1, scale later
        )
        
        # Direction estimation head (4 directions: front, back, left, right)
        self.direction_estimator = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 directions
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        features_flat = features.view(features.size(0), -1)
        
        classification = self.classifier(features_flat)
        distance = self.distance_estimator(features_flat) * 100  # Scale to 0-100 meters
        direction = self.direction_estimator(features_flat)  # Raw logits
        
        return classification, distance, direction

class GunshotTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.classification_loss = nn.CrossEntropyLoss()
        self.distance_loss = nn.MSELoss()
        self.direction_loss = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            class_pred, dist_pred, dir_pred = self.model(data)
            
            # Classification loss
            class_loss = self.classification_loss(class_pred, target)
            
            # For now, only use classification loss
            # You can add distance and direction losses when you have labeled data for them
            loss = class_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = class_pred.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                class_pred, _, _ = self.model(data)
                loss = self.classification_loss(class_pred, target)
                total_loss += loss.item()
                pred = class_pred.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, epochs=10):
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_gunshot_model.pth')
                print(f'New best model saved! Accuracy: {best_val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            self.scheduler.step()

class RealTimeDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.preprocessor = AudioPreprocessor()
        
        # Load label encoder
        try:
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
                self.labels = self.label_encoder.classes_
        except:
            # Default labels if encoder not found
            self.labels = ['assault_rifle', 'sniper_rifle', 'submachine_gun', 'shotgun', 'pistol', 'non_gunshot']
        
        # Initialize model
        self.model = GunshotClassifier(num_classes=len(self.labels))
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.directions = ['front', 'back', 'left', 'right']
        self.audio_buffer = np.zeros(self.preprocessor.target_length)
        self.buffer_lock = threading.Lock()
        self.detection_threshold = 0.7  # Lowered threshold
        self.min_detection_interval = 0.5  # Reduced interval
        self.last_detection_time = 0
        print(f"Real-time detector initialized with {len(self.labels)} classes")
        print(f"Classes: {self.labels}")
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        with self.buffer_lock:
            # Slide buffer and add new data
            self.audio_buffer[:-frames] = self.audio_buffer[frames:]
            self.audio_buffer[-frames:] = indata[:, 0]
    
    def detect_gunshot(self, audio_data):
        try:
            # Convert to tensor
            waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            
            # Extract features
            features = self.preprocessor.extract_features(waveform)
            features = features.unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                # Get predictions
                class_pred, dist_pred, dir_pred = self.model(features)
                
                # Process classification
                class_prob = torch.softmax(class_pred, dim=1)
                max_prob, predicted_class = torch.max(class_prob, 1)
                
                # Process direction
                dir_prob = torch.softmax(dir_pred, dim=1)
                max_dir_prob, predicted_dir = torch.max(dir_prob, 1)
                
                # Calculate SPL
                spl = self.preprocessor.calculate_spl(waveform)
                
                # Estimate distance from SPL
                estimated_distance = self.estimate_distance_from_spl(spl)
                
                return {
                    'class': self.labels[predicted_class.item()],
                    'confidence': max_prob.item(),
                    'distance': estimated_distance,
                    'direction': self.directions[predicted_dir.item()],
                    'direction_confidence': max_dir_prob.item(),
                    'spl': spl,
                    'all_class_probs': {self.labels[i]: class_prob[0][i].item() for i in range(len(self.labels))},
                    'all_dir_probs': {self.directions[i]: dir_prob[0][i].item() for i in range(len(self.directions))}
                }
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def estimate_distance_from_spl(self, spl):
        # Improved distance estimation
        # Reference: gunshot at 1m ≈ 140 dB
        reference_spl = 140
        reference_distance = 1.0
        
        # Distance doubles for every 6 dB reduction
        distance = reference_distance * 2 ** ((reference_spl - spl) / 6)
        
        # Clamp to reasonable range
        return max(1.0, min(distance, 500.0))
    
    def start_detection(self):
        print("Starting real-time gunshot detection...")
        print("Press Ctrl+C to stop")
        
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.preprocessor.sample_rate,
                blocksize=1024
            ):
                while True:
                    time.sleep(0.1)
                    
                    current_time = time.time()
                    if current_time - self.last_detection_time < self.min_detection_interval:
                        continue
                    
                    # Get audio data
                    with self.buffer_lock:
                        audio_copy = self.audio_buffer.copy()
                    
                    # Detect
                    result = self.detect_gunshot(audio_copy)
                    
                    if result is None:
                        continue
                    
                    # Check if audio is too quiet
                    if result['spl'] < -40:
                        continue
                    
                    # Display results
                    print(f"\n--- Detection Result ---")
                    print(f"Predicted Class: {result['class']}")
                    print(f"Confidence: {result['confidence']:.3f}")
                    print(f"Distance: {result['distance']:.1f}m")
                    print(f"Direction: {result['direction']} (conf: {result['direction_confidence']:.3f})")
                    print(f"SPL: {result['spl']:.1f} dB")
                    
                    # Show top predictions
                    print("\nTop class predictions:")
                    sorted_probs = sorted(result['all_class_probs'].items(), key=lambda x: x[1], reverse=True)
                    for cls, prob in sorted_probs[:3]:
                        print(f"  {cls}: {prob:.3f}")
                    
                    # Alert for gunshot detection
                    if result['class'] != 'non_gunshot' and result['confidence'] > self.detection_threshold:
                        print("\n🚨 GUNSHOT DETECTED! 🚨")
                        self.last_detection_time = current_time
                    
                    print("-" * 40)
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        except Exception as e:
            print(f"Detection error: {e}")

def train_model(data_dir):
    print("Starting model training...")
    
    # Initialize components
    preprocessor = AudioPreprocessor()
    dataset = GunshotDataset(data_dir, preprocessor)
    
    if len(dataset) == 0:
        print("No samples found in dataset!")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Initialize model
    num_classes = len(dataset.label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {dataset.label_encoder.classes_}")
    
    model = GunshotClassifier(num_classes=num_classes)
    
    # Train model
    trainer = GunshotTrainer(model)
    trainer.train(train_loader, val_loader, epochs=30)
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.label_encoder, f)
    
    print("Training completed!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Gunshot Detection System')
    parser.add_argument('--mode', choices=['train', 'detect'], required=True,
                        help='Mode: train or detect')
    parser.add_argument('--data_dir', type=str, help='Path to training data directory')
    parser.add_argument('--model_path', type=str, default='best_gunshot_model.pth',
                        help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data_dir:
            print("Error: --data_dir is required for training")
            return
        train_model(args.data_dir)
    
    elif args.mode == 'detect':
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found")
            return
        
        detector = RealTimeDetector(args.model_path)
        detector.start_detection()

if __name__ == "__main__":
    main()

"""
Usage:
1. Train the model:
   python gunshot_detector.py --mode train --data_dir ./data

2. Run real-time detection:
   python gunshot_detector.py --mode detect --model_path best_gunshot_model.pth
"""