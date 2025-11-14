# Deep Learning - Assignment 2: Legal Clause Similarity (PyTorch Version)
# Robust version with better error handling and checkpointing

import os
import re
import json
import glob
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             average_precision_score, confusion_matrix)
from collections import Counter

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
DATA_DIR = 'dataset'
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 256
MAX_POSITIVE_PAIRS = 40000
MAX_NEGATIVE_PAIRS = 40000
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# Speed-up config
LSTM_UNITS = 32
EPOCHS = 10
BATCH_SIZE = 128
TRAIN_SUBSET_SIZE = 15000
EMBEDDING_DIM = 128

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# File paths
TRAIN_DATA_FILE = 'train_data.npz'
VAL_DATA_FILE = 'val_data.npz'
VOCAB_FILE = 'vocab.json'
CONFIG_FILE = 'model_config.json'
MODEL_BILSTM_FILE = 'model_bilstm.pt'
MODEL_ATTENTION_FILE = 'model_attention.pt'
CHECKPOINT_BILSTM = 'checkpoint_bilstm.pt'
CHECKPOINT_ATTENTION = 'checkpoint_attention.pt'

# --- HELPER FUNCTIONS ---

def safe_operation(func, error_msg, default_return=None):
    """Wrapper for safe execution with error handling"""
    try:
        return func()
    except Exception as e:
        print(f"ERROR: {error_msg}")
        print(f"Details: {str(e)}")
        return default_return

def download_nltk_resources():
    """Download NLTK resources if missing"""
    try:
        nltk.data.find('corpora/stopwords')
        print("✓ NLTK 'stopwords' resource available.")
    except LookupError:
        print("Downloading NLTK 'stopwords'...")
        nltk.download('stopwords', quiet=True)
        print("✓ Download complete.")

def load_all_csvs_from_folder(folder_path):
    """Load and combine all CSV files from folder"""
    print(f"Scanning for CSV files in '{folder_path}'...")
    search_path = os.path.join(folder_path, '*.csv')
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"❌ No CSV files found in '{folder_path}'.")
        return None
        
    print(f"Found {len(csv_files)} CSV files. Reading and combining...")
    dataframes_list = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        def load_csv():
            df = pd.read_csv(filepath)
            if 'clause_text' in df.columns and 'clause_type' in df.columns:
                df = df.rename(columns={
                    'clause_text': 'Clause',
                    'clause_type': 'Category'
                })
            elif 'Clause' not in df.columns or 'Category' not in df.columns:
                print(f"⚠ Skipping {filename}: Missing required columns")
                return None
            return df
        
        df = safe_operation(load_csv, f"Failed to load {filename}")
        if df is not None:
            dataframes_list.append(df)
            
    if not dataframes_list:
        print("❌ No valid data loaded.")
        return None
        
    all_data = pd.concat(dataframes_list, ignore_index=True)
    print(f"✓ Successfully combined {len(dataframes_list)} CSV files ({len(all_data)} rows).")
    return all_data

def clean_text(text):
    """Clean a single text string"""
    if not isinstance(text, str):
        return ""
    if not hasattr(clean_text, 'stop_words'):
        clean_text.stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in clean_text.stop_words]
    return " ".join(tokens)

def create_similarity_pairs(df, max_pos, max_neg):
    """Generate positive and negative pairs"""
    print("Generating similarity pairs...")
    text_pairs_1, text_pairs_2, labels = [], [], []
    
    category_map = {cat: group['Clause'].tolist() for cat, group in df.groupby('Category')}
    categories_with_pairs = [cat for cat, clauses in category_map.items() if len(clauses) > 1]
    categories_all = list(category_map.keys())

    if not categories_with_pairs:
        print("❌ No categories with multiple clauses found.")
        return [], [], []
    
    # Positive pairs
    print(f"Sampling {max_pos} positive pairs...")
    positive_count = 0
    attempts = 0
    max_attempts = max_pos * 10
    
    while positive_count < max_pos and attempts < max_attempts:
        attempts += 1
        try:
            category = np.random.choice(categories_with_pairs)
            clause1, clause2 = np.random.choice(category_map[category], 2, replace=False)
            text_pairs_1.append(clause1)
            text_pairs_2.append(clause2)
            labels.append(1)
            positive_count += 1
        except:
            continue
    
    print(f"✓ Generated {positive_count} positive pairs")
    
    # Negative pairs
    print(f"Sampling {max_neg} negative pairs...")
    negative_count = 0
    attempts = 0
    
    while negative_count < max_neg and attempts < max_attempts:
        attempts += 1
        try:
            cat1_name, cat2_name = np.random.choice(categories_all, 2, replace=False)
            clause1 = np.random.choice(category_map[cat1_name])
            clause2 = np.random.choice(category_map[cat2_name])
            text_pairs_1.append(clause1)
            text_pairs_2.append(clause2)
            labels.append(0)
            negative_count += 1
        except:
            continue
    
    print(f"✓ Generated {negative_count} negative pairs")
    print(f"✓ Total pairs: {len(labels)}")
    return text_pairs_1, text_pairs_2, labels

def build_vocabulary(texts, max_vocab_size):
    """Build vocabulary from texts"""
    print("Building vocabulary...")
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.split())
    
    token_counts = Counter(all_tokens)
    most_common = token_counts.most_common(max_vocab_size - 2)  # -2 for special tokens
    
    # Create vocabulary
    vocab = {'<pad>': 0, '<unk>': 1}
    for idx, (token, _) in enumerate(most_common, start=2):
        vocab[token] = idx
    
    print(f"✓ Vocabulary built (size: {len(vocab)})")
    return vocab

def texts_to_sequences(texts, vocab):
    """Convert texts to sequences of indices"""
    sequences = []
    for text in texts:
        seq = [vocab.get(token, vocab['<unk>']) for token in text.split()]
        sequences.append(seq)
    return sequences

def pad_sequence(seq, max_len, pad_idx):
    """Pad or truncate sequence"""
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [pad_idx] * (max_len - len(seq))

def run_preprocessing():
    """Run full preprocessing pipeline"""
    print("\n" + "="*60)
    print("STARTING PREPROCESSING")
    print("="*60)
    start_time = time.time()
    
    # Step 1: Download NLTK resources
    download_nltk_resources()
    
    # Step 2: Load data
    df = load_all_csvs_from_folder(DATA_DIR)
    if df is None:
        return False
    
    # Step 3: Clean data
    print("\nCleaning data...")
    df = df.dropna(subset=['Clause', 'Category']).drop_duplicates(subset=['Clause'])
    print(f"✓ Data cleaned ({len(df)} unique clauses)")
    
    # Step 4: Generate pairs
    original_p1, original_p2, labels = create_similarity_pairs(df, MAX_POSITIVE_PAIRS, MAX_NEGATIVE_PAIRS)
    if not labels:
        return False
    
    # Step 5: Clean text
    print("\nCleaning text pairs...")
    cleaned_p1 = [clean_text(t) for t in original_p1]
    cleaned_p2 = [clean_text(t) for t in original_p2]
    all_cleaned_text = cleaned_p1 + cleaned_p2
    print("✓ Text cleaning complete")
    
    # Step 6: Build vocabulary
    vocab = build_vocabulary(all_cleaned_text, MAX_VOCAB_SIZE)
    PAD_IDX = vocab['<pad>']
    VOCAB_SIZE = len(vocab)
    
    # Save vocabulary
    with open(VOCAB_FILE, 'w') as f:
        json.dump(vocab, f)
    print(f"✓ Vocabulary saved to {VOCAB_FILE}")
    
    # Step 7: Convert to sequences
    print("\nConverting texts to sequences...")
    seq_p1 = texts_to_sequences(cleaned_p1, vocab)
    seq_p2 = texts_to_sequences(cleaned_p2, vocab)
    print("✓ Conversion complete")
    
    # Step 8: Pad sequences
    print(f"Padding sequences to length {MAX_SEQ_LEN}...")
    padded_p1 = np.array([pad_sequence(s, MAX_SEQ_LEN, PAD_IDX) for s in seq_p1])
    padded_p2 = np.array([pad_sequence(s, MAX_SEQ_LEN, PAD_IDX) for s in seq_p2])
    labels = np.array(labels)
    print("✓ Padding complete")
    
    # Step 9: Train/validation split
    print("\nSplitting data...")
    indices = np.arange(len(labels))
    (train_indices, val_indices, 
     train_labels, val_labels) = train_test_split(indices, labels, 
                                                   test_size=TEST_SPLIT_SIZE, 
                                                   random_state=RANDOM_STATE, 
                                                   stratify=labels)
    
    train_p1, val_p1 = padded_p1[train_indices], padded_p1[val_indices]
    train_p2, val_p2 = padded_p2[train_indices], padded_p2[val_indices]
    
    original_p1_np = np.array(original_p1)
    original_p2_np = np.array(original_p2)
    val_orig_p1 = original_p1_np[val_indices]
    val_orig_p2 = original_p2_np[val_indices]
    
    print(f"✓ Train size: {len(train_labels)}, Validation size: {len(val_labels)}")
    
    # Step 10: Save processed data
    print("\nSaving processed data...")
    np.savez_compressed(TRAIN_DATA_FILE, pair1=train_p1, pair2=train_p2, labels=train_labels)
    np.savez_compressed(VAL_DATA_FILE, pair1=val_p1, pair2=val_p2, labels=val_labels, 
                       orig_pair1=val_orig_p1, orig_pair2=val_orig_p2)
    
    config = {
        'max_seq_len': MAX_SEQ_LEN, 
        'vocab_size': VOCAB_SIZE, 
        'embedding_dim': EMBEDDING_DIM, 
        'pad_idx': PAD_IDX
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
    
    print(f"✓ Data saved to {TRAIN_DATA_FILE} and {VAL_DATA_FILE}")
    print(f"✓ Config saved to {CONFIG_FILE}")
    
    elapsed = time.time() - start_time
    print(f"\n✓ PREPROCESSING COMPLETE (Time: {elapsed:.2f}s)")
    print("="*60 + "\n")
    return True

# --- PYTORCH DATASET ---

class ClausePairDataset(Dataset):
    def __init__(self, p1, p2, labels):
        self.p1 = torch.tensor(p1, dtype=torch.long)
        self.p2 = torch.tensor(p2, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.p1[idx], self.p2[idx], self.labels[idx]

# --- MODELS ---

class SiameseBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, pad_idx):
        super(SiameseBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_units * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward_once(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.max(lstm_out, dim=1)[0]
        return pooled

    def forward(self, p1, p2):
        vec_a = self.forward_once(p1)
        vec_b = self.forward_once(p2)
        distance = torch.abs(vec_a - vec_b)
        out = self.fc(distance)
        return out.squeeze()

class SiameseAttentionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, pad_idx):
        super(SiameseAttentionBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
        
        lstm_output_dim = lstm_units * 2
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward_once(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        attn_weights = torch.tanh(self.attention(lstm_out))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        return context_vector

    def forward(self, p1, p2):
        vec_a = self.forward_once(p1)
        vec_b = self.forward_once(p2)
        distance = torch.abs(vec_a - vec_b)
        out = self.fc(distance)
        return out.squeeze()

# --- TRAINING FUNCTIONS ---

def train_model(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for p1, p2, labels in loader:
        p1, p2, labels = p1.to(device), p2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(p1, p2)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
    return total_loss / total_samples, total_correct / total_samples

def evaluate_model(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for p1, p2, labels in loader:
            p1, p2, labels = p1.to(device), p2.to(device), labels.to(device)
            
            outputs = model(p1, p2)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return (total_loss / total_samples, 
            total_correct / total_samples, 
            np.array(all_probs), 
            np.array(all_labels))

def save_checkpoint(model, optimizer, epoch, history, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    """Load training checkpoint"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['history']
    return 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

def plot_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{model_name}_training_graphs.png"
    plt.savefig(filename)
    print(f"✓ Saved training graphs to {filename}")
    plt.close()

def report_metrics(pred_probs, val_labels, model_name):
    """Calculate and report metrics"""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print('='*60)
    
    preds = (pred_probs > 0.5).astype(int)
    
    accuracy = accuracy_score(val_labels, preds)
    precision = precision_score(val_labels, preds, zero_division=0)
    recall = recall_score(val_labels, preds, zero_division=0)
    f1 = f1_score(val_labels, preds, zero_division=0)
    roc_auc = roc_auc_score(val_labels, pred_probs)
    pr_auc = average_precision_score(val_labels, pred_probs)
    
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(val_labels, preds, 
                                target_names=['Dissimilar (0)', 'Similar (1)'],
                                zero_division=0))
    
    return {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}

def train_model_full(model, train_loader, val_loader, model_name, model_file, checkpoint_file):
    """Full training loop with checkpointing"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print('='*60)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Try to load checkpoint
    start_epoch, history = load_checkpoint(model, optimizer, checkpoint_file)
    
    if start_epoch > 0:
        print(f"✓ Resuming from epoch {start_epoch}")
    else:
        print("Starting fresh training")
    
    best_val_acc = max(history['val_acc']) if history['val_acc'] else -1
    epochs_no_improve = 0
    patience = 3
    
    start_time = time.time()
    
    for epoch in range(start_epoch, EPOCHS):
        try:
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch + 1, history, checkpoint_file)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_file)
                print(f"✓ Best model saved (Val Acc: {val_acc:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered (no improvement for {patience} epochs)")
                    break
                    
        except Exception as e:
            print(f"⚠ Error during epoch {epoch+1}: {e}")
            print("Saving checkpoint and continuing...")
            save_checkpoint(model, optimizer, epoch + 1, history, checkpoint_file)
            continue
            
    elapsed = time.time() - start_time
    print(f"✓ Training complete (Time: {elapsed:.2f}s)")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Removed checkpoint file")
    
    plot_history(history, model_name)
    return history

# --- MAIN FUNCTION ---

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("LEGAL CLAUSE SIMILARITY - PYTORCH VERSION")
    print("="*60 + "\n")
    
    # Step 1: Preprocessing
    required_files = [TRAIN_DATA_FILE, VAL_DATA_FILE, VOCAB_FILE, CONFIG_FILE]
    if not all(os.path.exists(f) for f in required_files):
        print("⚠ Processed files missing. Running preprocessing...\n")
        if not run_preprocessing():
            print("❌ Preprocessing failed. Exiting.")
            return
    else:
        print("✓ Processed files found. Skipping preprocessing.\n")
    
    # Step 2: Load data
    print("Loading data...")
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        with open(VOCAB_FILE, 'r') as f:
            vocab = json.load(f)
        
        with np.load(TRAIN_DATA_FILE) as data:
            train_p1, train_p2, train_labels = data['pair1'], data['pair2'], data['labels']
        
        with np.load(VAL_DATA_FILE, allow_pickle=True) as data:
            val_p1, val_p2, val_labels = data['pair1'], data['pair2'], data['labels']
            val_orig_p1, val_orig_p2 = data['orig_pair1'], data['orig_pair2']
        
        vocab_size = config['vocab_size']
        pad_idx = config['pad_idx']
        
        print("✓ Data loaded successfully\n")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Try deleting all .npz, .json, and .pt files and re-running.")
        return
    
    # Subset training data for speed
    if len(train_labels) > TRAIN_SUBSET_SIZE:
        print(f"⚡ Using subset of {TRAIN_SUBSET_SIZE} training pairs for speed")
        train_p1 = train_p1[:TRAIN_SUBSET_SIZE]
        train_p2 = train_p2[:TRAIN_SUBSET_SIZE]
        train_labels = train_labels[:TRAIN_SUBSET_SIZE]
        print(f"✓ Training set size: {len(train_labels)}\n")
    
    # Create dataloaders
    train_dataset = ClausePairDataset(train_p1, train_p2, train_labels)
    val_dataset = ClausePairDataset(val_p1, val_p2, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Step 3: Train Model 1 (BiLSTM)
    if not os.path.exists(MODEL_BILSTM_FILE):
        model_bilstm = SiameseBiLSTM(vocab_size, EMBEDDING_DIM, LSTM_UNITS, pad_idx).to(DEVICE)
        print(f"Model params: {sum(p.numel() for p in model_bilstm.parameters()):,}")
        
        train_model_full(model_bilstm, train_loader, val_loader, 
                        "Siamese_BiLSTM", MODEL_BILSTM_FILE, CHECKPOINT_BILSTM)
    else:
        print(f"✓ {MODEL_BILSTM_FILE} exists. Skipping training.\n")
    
    # Step 4: Train Model 2 (Attention)
    if not os.path.exists(MODEL_ATTENTION_FILE):
        model_attention = SiameseAttentionBiLSTM(vocab_size, EMBEDDING_DIM, LSTM_UNITS, pad_idx).to(DEVICE)
        print(f"Model params: {sum(p.numel() for p in model_attention.parameters()):,}")
        
        train_model_full(model_attention, train_loader, val_loader,
                        "Siamese_Attention_BiLSTM", MODEL_ATTENTION_FILE, CHECKPOINT_ATTENTION)
    else:
        print(f"✓ {MODEL_ATTENTION_FILE} exists. Skipping training.\n")
    
    # Step 5: Evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate BiLSTM
    try:
        model_bilstm = SiameseBiLSTM(vocab_size, EMBEDDING_DIM, LSTM_UNITS, pad_idx).to(DEVICE)
        model_bilstm.load_state_dict(torch.load(MODEL_BILSTM_FILE, map_location=DEVICE))
        _, _, bilstm_probs, bilstm_labels = evaluate_model(model_bilstm, val_loader, criterion, DEVICE)
        metrics_bilstm = report_metrics(bilstm_probs, bilstm_labels, "Siamese BiLSTM")
    except Exception as e:
        print(f"⚠ Error evaluating BiLSTM: {e}")
        metrics_bilstm = {"accuracy": 0, "f1": 0, "roc_auc": 0}
    
    # Evaluate Attention
    try:
        model_attention = SiameseAttentionBiLSTM(vocab_size, EMBEDDING_DIM, LSTM_UNITS, pad_idx).to(DEVICE)
        model_attention.load_state_dict(torch.load(MODEL_ATTENTION_FILE, map_location=DEVICE))
        _, _, attn_probs, attn_labels = evaluate_model(model_attention, val_loader, criterion, DEVICE)
        metrics_attention = report_metrics(attn_probs, attn_labels, "Siamese BiLSTM with Attention")
    except Exception as e:
        print(f"⚠ Error evaluating Attention model: {e}")
        metrics_attention = {"accuracy": 0, "f1": 0, "roc_auc": 0}
    
    # Step 6: Comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Metric':<15} {'BiLSTM':<12} {'BiLSTM + Attention':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<15} {metrics_bilstm['accuracy']:<12.4f} {metrics_attention['accuracy']:<20.4f}")
    print(f"{'F1-Score':<15} {metrics_bilstm['f1']:<12.4f} {metrics_attention['f1']:<20.4f}")
    print(f"{'ROC-AUC':<15} {metrics_bilstm['roc_auc']:<12.4f} {metrics_attention['roc_auc']:<20.4f}")
    print("="*60)
    
    # Determine winner
    if metrics_attention['f1'] > metrics_bilstm['f1']:
        print(f"\n🏆 Winner: BiLSTM with Attention (F1: {metrics_attention['f1']:.4f})")
    elif metrics_bilstm['f1'] > metrics_attention['f1']:
        print(f"\n🏆 Winner: BiLSTM (F1: {metrics_bilstm['f1']:.4f})")
    else:
        print(f"\n🤝 Tie (F1: {metrics_bilstm['f1']:.4f})")
    
    print("\n" + "="*60)
    print("✓ SCRIPT COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()