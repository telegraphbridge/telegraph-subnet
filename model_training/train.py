# Fixed Telegraph LSTM Training Script
from google.colab import files
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import gc

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create output directory in Google Drive
import os
output_dir = '/content/drive/MyDrive/telegraph_model'
os.makedirs(output_dir, exist_ok=True)
print(f"Model output directory: {output_dir}")

# Path to your file in Google Drive
file_path = '/content/drive/MyDrive/nether-collected-data.json'

print("Loading JSON data (this may take a few minutes)...")
# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)
    
# If data is not a list (e.g., it's a dict with a container field), extract it
if not isinstance(data, list):
    if isinstance(data, dict) and any(isinstance(data.get(key), list) for key in data):
        # Find the first list in the dict
        for key in data:
            if isinstance(data[key], list):
                data = data[key]
                break
    else:
        raise ValueError("Expected data to be a list or contain a list")

print(f"Loaded {len(data)} transaction records")

# Display a sample record
print("Sample record:")
print(json.dumps(data[0], indent=2))

def extract_features(tx):
    """Extract features from a transaction record"""
    # Basic features
    basic_features = [
        float(tx.get('liquidityPoolSize', 0)),
        float(tx.get('volume24hUsd', 0)),
        float(tx.get('marketCapUsd', 0)),
        float(tx.get('buyAmount', 0)),
        float(tx.get('buyValueEth', 0)),
        float(tx.get('price', 0)),
        float(tx.get('priceInUsd', 0)),
        float(tx.get('walletEthBalance', 0))
    ]
    
    # Process wallet stats (we'll calculate these separately)
    wallet_stats = [
        float(tx.get('historicalROI', 0.0)),
        float(tx.get('tradeFrequency', 0.0)),
        float(tx.get('diversityScore', 0.0))
    ]
    
    # Extract historical liquidity sequence
    liquidity_seq = []
    if 'historicalLiquidity' in tx and tx['historicalLiquidity']:
        for item in tx['historicalLiquidity']:
            liquidity_seq.append(float(item.get('value', 0)))
        
        # Ensure exactly 12 values (pad or truncate)
        if len(liquidity_seq) < 12:
            liquidity_seq.extend([0.0] * (12 - len(liquidity_seq)))
        liquidity_seq = liquidity_seq[:12]
    else:
        liquidity_seq = [0.0] * 12
    
    # Extract historical volume sequence
    volume_seq = []
    if 'historicalVolume' in tx and tx['historicalVolume']:
        for item in tx['historicalVolume']:
            volume_seq.append(float(item.get('value', 0)))
        
        # Ensure exactly 12 values (pad or truncate)
        if len(volume_seq) < 12:
            volume_seq.extend([0.0] * (12 - len(volume_seq)))
        volume_seq = volume_seq[:12]
    else:
        volume_seq = [0.0] * 12
    
    # Combine all features
    features = basic_features + wallet_stats + liquidity_seq + volume_seq
    return features

# Calculate wallet statistics
def calculate_wallet_stats(data):
    wallet_stats = {}
    wallet_tokens = {}
    wallet_trades = {}
    
    # Group transactions by wallet
    for tx in data:
        wallet = tx.get('walletID')  # FIXED: using walletID instead of walletAddress
        if not wallet:
            continue
            
        # Track tokens per wallet
        if wallet not in wallet_tokens:
            wallet_tokens[wallet] = set()
        
        token = tx.get('tokenID')
        if token:
            wallet_tokens[wallet].add(token)
        
        wallet_trades[wallet] = wallet_trades.get(wallet, 0) + 1
    
    for wallet in wallet_trades:
        wallet_stats[wallet] = {
            'diversity_score': len(wallet_tokens[wallet]),
            'trade_frequency': wallet_trades[wallet],
            'historical_roi': 0.0
        }
    
    return wallet_stats

# Calculate labels (token performance)
def calculate_labels(data):
    # Group transactions by token and timestamp
    token_txs = {}
    for tx in data:
        token = tx.get('tokenID')  # FIXED: using tokenID instead of tokenAddress
        if not token:
            continue
            
        if token not in token_txs:
            token_txs[token] = []
        
        try:    
            timestamp = datetime.fromisoformat(tx['timestamp'].replace('Z', '+00:00'))
            token_txs[token].append((timestamp, tx))
        except (KeyError, ValueError) as e:
            # Skip transactions with invalid timestamps
            continue
    
    # Sort transactions by timestamp for each token
    for token in token_txs:
        token_txs[token].sort(key=lambda x: x[0])
    
    # Calculate price changes (success = 1 if price increases by >5% within next transaction)
    labels = {}
    for token, txs in token_txs.items():
        for i in range(len(txs) - 1):
            try:
                current_price = float(txs[i][1].get('priceInUsd', 0))
                next_price = float(txs[i+1][1].get('priceInUsd', 0))
                
                if current_price > 0:
                    price_change = (next_price - current_price) / current_price
                    # Success if price increases by more than 5%
                    labels[txs[i][1]['transactionHash']] = 1.0 if price_change > 0.05 else 0.0
            except (KeyError, ValueError, TypeError) as e:
                # Skip problematic transactions
                continue
    
    return labels

print("Calculating wallet statistics...")
# Process all data
wallet_stats = calculate_wallet_stats(data)
print(f"Calculated stats for {len(wallet_stats)} wallets")

print("Calculating token performance labels...")
labels = calculate_labels(data)
print(f"Generated {len(labels)} labels")

# Process data in chunks to avoid memory issues
chunk_size = 10000
num_chunks = (len(data) + chunk_size - 1) // chunk_size

X_all = []
y_all = []

print(f"Processing data in {num_chunks} chunks...")
for i in range(num_chunks):
    print(f"Processing chunk {i+1}/{num_chunks}")
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(data))
    chunk_data = data[start_idx:end_idx]
    
    # Process this chunk
    for tx in chunk_data:
        # Skip if no transaction hash (needed for label)
        if 'transactionHash' not in tx:
            continue
            
        # Skip if this transaction doesn't have a label
        if tx['transactionHash'] not in labels:
            continue
            
        # Extract features
        features = extract_features(tx)
        
        # Update wallet stats
        wallet = tx.get('walletID')  # FIXED: using walletID instead of walletAddress
        if wallet and wallet in wallet_stats:
            stats = wallet_stats[wallet]
            # These are now already included in extract_features
            # features[8] = stats['historical_roi']
            # features[9] = stats['trade_frequency']
            # features[10] = stats['diversity_score']
        
        # Add to our dataset
        X_all.append(features)
        y_all.append(labels[tx['transactionHash']])
    
    # Clear memory
    del chunk_data
    gc.collect()

# Convert to numpy arrays
X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.float32)

# Check for class imbalance
positive_examples = np.sum(y)
negative_examples = len(y) - positive_examples
ratio = negative_examples / positive_examples if positive_examples > 0 else float('inf')
print(f"Class distribution - Positive: {positive_examples}, Negative: {negative_examples}, Ratio: {ratio:.2f}:1")

# Balance classes if needed (if ratio > 10)
if ratio > 10:
    from sklearn.utils import resample
    print("Balancing classes...")
    # Get indices of positive and negative examples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # Downsample negative class to 3x positive class
    neg_indices_downsampled = resample(neg_indices, 
                                     replace=False, 
                                     n_samples=min(len(neg_indices), 3*len(pos_indices)),
                                     random_state=42)
    
    # Combine indices and use them to select from X and y
    combined_indices = np.concatenate([pos_indices, neg_indices_downsampled])
    X = X[combined_indices]
    y = y[combined_indices]
    
    print(f"After balancing - Total samples: {len(X)}")

print(f"Processed {len(X)} samples with {X.shape[1]} features")

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved feature scaler to {scaler_path}")
    
# Split into train/validation sets
print("Splitting into train/validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create PyTorch datasets
class TransactionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = TransactionDataset(X_train, y_train)
val_dataset = TransactionDataset(X_val, y_val)

# Create data loaders
batch_size = 32  # FIXED: Reduced from 64 for memory efficiency
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Define the LSTM model (matching the one in your subnet code)
class NetherilLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(NetherilLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for token scoring
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # If input is just features (no sequence dimension)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take output from last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out.squeeze()

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model
input_size = X_train.shape[1]
model = NetherilLSTM(input_size=input_size).to(device)

# Loss function and optimizer
# FIXED: Use weighted loss if ratio is high
if ratio > 5:
    pos_weight = torch.tensor([min(ratio, 10.0)])  # Cap at 10
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    print(f"Using weighted loss with weight: {pos_weight.item()}")
else:
    criterion = nn.BCELoss()
    print("Using standard BCE loss")
    
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training parameters
num_epochs = 30
train_losses = []
val_losses = []
best_val_loss = float('inf')

# Training loop
print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track loss
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track loss
            val_loss += loss.item() * inputs.size(0)
    
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)
    
    # Print statistics
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Adjust learning rate
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'token_scores': {},  # Empty dict for token scores
            'wallet_cache': {}   # Empty dict for wallet cache
        }, best_model_path)
        print(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch+1}")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
plt.show()

# Save the final model
final_model_path = os.path.join(output_dir, 'final_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'token_scores': {},
    'wallet_cache': {}
}, final_model_path)
print(f"Saved final model to {final_model_path}")

# Save the feature scaler again to ensure it's available
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("Training complete!")

# Evaluate the best model
best_model_path = os.path.join(output_dir, 'best_model.pth')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Calculate accuracy
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# Calculate precision, recall, F1 score
from sklearn.metrics import precision_recall_fscore_support

all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

   
from google.colab import files
files.download(os.path.join(output_dir, 'best_model.pth'))
files.download(os.path.join(output_dir, 'feature_scaler.pkl'))