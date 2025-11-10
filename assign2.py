# Deep Learning - Assignment 2: Legal Clause Similarity
#
# This script implements a complete pipeline to tackle the legal clause similarity task.
#
# INSTRUCTIONS:
# 1. Make sure you have a folder named 'dataset' in the same directory,
#    containing all your .csv files.
# 2. Run this command in your terminal first to install all required libraries:
#    pip install pandas tensorflow scikit-learn nltk
# 3. *** DELETE 'model_bilstm.keras' and 'model_attention.keras' if they exist! ***
# 4. Run this script:
#    python assign2.py

# --- 1. IMPORTS ---
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional, LSTM,
                                     GlobalMaxPooling1D, GlobalAveragePooling1D,
                                     Attention, Dense, Dropout, Lambda)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

print("TensorFlow Version:", tf.__version__)

# --- 2. CONFIGURATION ---

# Data parameters
DATA_DIR = 'dataset'  # Folder containing all the .csv files

# Preprocessing parameters
MAX_VOCAB_SIZE = 20000  # Max words to keep in the vocabulary
MAX_SEQ_LEN = 256       # Max length for a clause (truncate/pad)

# --- PAIR SAMPLING LIMITS ---
# 40,000 positive + 40,000 negative = 80,000 total pairs.
MAX_POSITIVE_PAIRS = 40000
MAX_NEGATIVE_PAIRS = 40000

TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# --- EMERGENCY SPEED-UP CONFIG ---
# These are set to get a result *fast*
LSTM_UNITS = 32         # Reduced from 64
EPOCHS = 10             # Reduced from 20
BATCH_SIZE = 128        # Increased from 64 (faster training)
TRAIN_SUBSET_SIZE = 15000 # Use only 15k pairs for training (key speedup)
# --- END SPEED-UP CONFIG ---

EMBEDDING_DIM = 128 # This can stay 128

# Processed file paths
TRAIN_DATA_FILE = 'train_data.npz'
VAL_DATA_FILE = 'val_data.npz'
TOKENIZER_FILE = 'tokenizer.json'
CONFIG_FILE = 'model_config.json'
MODEL_BILSTM_FILE = 'model_bilstm.keras'
MODEL_ATTENTION_FILE = 'model_attention.keras'

# --- 3. DATA LOADING & PREPROCESSING (HELPER FUNCTIONS) ---

def download_nltk_resources():
    """Checks for and downloads NLTK stopwords if missing."""
    try:
        nltk.data.find('corpora/stopwords')
        print("NLTK 'stopwords' resource already downloaded.")
    except LookupError:
        print("Downloading NLTK 'stopwords' resource...")
        nltk.download('stopwords')
        print("Download complete.")

def load_all_csvs_from_folder(folder_path):
    """
    Finds all .csv files in the specified folder, reads them into
    pandas DataFrames, and concatenates them into a single DataFrame.
    
    This function specifically looks for 'clause_text' and 'clause_type'
    as per the Kaggle dataset and renames them.
    """
    print(f"Scanning for CSV files in '{folder_path}'...")
    search_path = os.path.join(folder_path, '*.csv')
    csv_files = glob.glob(search_path)
    
    if not csv_files:
        print(f"No CSV files found in '{folder_path}'.")
        return None
        
    print(f"Found {len(csv_files)} CSV files. Reading and combining...")
    dataframes_list = []
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath)

            # --- Column Check & Rename (Kaggle Dataset Specific) ---
            if 'clause_text' in df.columns and 'clause_type' in df.columns:
                # Rename columns to match what the rest of the script expects
                df = df.rename(columns={
                    'clause_text': 'Clause',
                    'clause_type': 'Category'
                })
            elif 'Clause' not in df.columns or 'Category' not in df.columns:
                # If it doesn't have *either* set of names, then skip
                # print(f"Warning: Skipping {filename}. Missing required columns ('Clause'/'Category' or 'clause_text'/'clause_type').")
                continue
            
            dataframes_list.append(df)
            
        except pd.errors.EmptyDataError:
            # print(f"Warning: Skipping {filename} because it is empty.")
            continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    if not dataframes_list:
        print("No valid data was loaded. Aborting.")
        return None
        
    all_data = pd.concat(dataframes_list, ignore_index=True)
    print("Successfully combined all CSV files.")
    return all_data

def clean_text(text):
    """
    Cleans a single string of text.
    """
    if not isinstance(text, str):
        return ""
    # Get stopwords once and cache it
    if not hasattr(clean_text, 'stop_words'):
        clean_text.stop_words = set(stopwords.words('english'))
        
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    tokens = text.split()
    tokens = [word for word in tokens if word not in clean_text.stop_words]
    return " ".join(tokens)

def create_similarity_pairs(df, max_pos, max_neg):
    """
    Samples a fixed number of positive (similar) and negative (dissimilar) pairs
    to create a balanced, manageable dataset.
    Returns: (text_pairs_1, text_pairs_2, labels)
    """
    print("Generating similarity pairs by sampling...")
    text_pairs_1 = []
    text_pairs_2 = []
    labels = []
    
    # --- EFFICIENT SAMPLING FIX ---
    # Create a dictionary {category: [list of clauses]} ONCE.
    print("Creating clause lookup map for efficient sampling...")
    category_map = {cat: group['Clause'].tolist() for cat, group in df.groupby('Category')}
    
    # Filter for categories that can form pairs
    categories_with_pairs = [cat for cat, clauses in category_map.items() if len(clauses) > 1]
    categories_all = list(category_map.keys())
    # --- END FIX ---

    if not categories_with_pairs:
        print("Error: No categories found with more than one clause. Cannot generate positive pairs.")
        return [], [], []
    
    # --- 1. Generate Positive Pairs (Similar) by Sampling ---
    print(f"Sampling {max_pos} positive pairs...")
    positive_count = 0
    while positive_count < max_pos:
        try:
            # Pick a random category that has more than one clause
            category = np.random.choice(categories_with_pairs)
            
            # Sample two *different* clauses from the pre-computed list
            clause1, clause2 = np.random.choice(category_map[category], 2, replace=False)
            
            text_pairs_1.append(clause1)
            text_pairs_2.append(clause2)
            labels.append(1) # 1 = Similar
            positive_count += 1

            if positive_count % 10000 == 0:
                print(f"... {positive_count} positive pairs generated", end='\r')

        except ValueError:
            # This might happen if a category is small, just retry
            continue
        except Exception as e:
            print(f"Error sampling positive pair: {e}")
            continue
    
    print(f"\nTotal positive pairs generated: {positive_count}")
    
    # --- 2. Generate Negative Pairs (Dissimilar) by Sampling ---
    print(f"Sampling {max_neg} negative pairs...")
    negative_count = 0
    while negative_count < max_neg:
        try:
            # Pick two different random categories
            cat1_name, cat2_name = np.random.choice(categories_all, 2, replace=False)
            
            # Pick one random *original* clause from each pre-computed list
            clause1 = np.random.choice(category_map[cat1_name])
            clause2 = np.random.choice(category_map[cat2_name])
            
            text_pairs_1.append(clause1)
            text_pairs_2.append(clause2)
            labels.append(0) # 0 = Dissimilar
            negative_count += 1

            if negative_count % 10000 == 0:
                print(f"... {negative_count} negative pairs generated", end='\r')

        except Exception as e:
            print(f"Error sampling negative pair: {e}")
            continue
            
    print(f"\nTotal pairs generated (positive + negative): {len(labels)}")
    return text_pairs_1, text_pairs_2, labels

def run_preprocessing():
    """Runs the full preprocessing pipeline and saves data to disk."""
    
    print("--- STARTING PREPROCESSING ---")
    start_time = time.time()
    
    # 0. Download NLTK data
    download_nltk_resources()

    # 1. Load Data
    df = load_all_csvs_from_folder(DATA_DIR)
    if df is None: 
        print("Preprocessing failed at data loading step.")
        return False # Return False on failure
    
    df = df.dropna(subset=['Clause', 'Category'])
    df = df.drop_duplicates(subset=['Clause'])

    # 2. Create Pairs from ORIGINAL text
    original_p1, original_p2, labels = create_similarity_pairs(df, MAX_POSITIVE_PAIRS, MAX_NEGATIVE_PAIRS)
    labels = np.array(labels)
    
    if len(labels) == 0:
        print("No pairs were generated. Check your data and sampling logic.")
        return False

    # 3. Clean Text
    print("Cleaning text pairs...")
    cleaned_p1 = [clean_text(t) for t in original_p1]
    cleaned_p2 = [clean_text(t) for t in original_p2]

    # 4. Tokenization
    print("Tokenizing text...")
    all_cleaned_text = cleaned_p1 + cleaned_p2
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_cleaned_text)
    
    with open(TOKENIZER_FILE, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    print(f"Tokenizer saved to {TOKENIZER_FILE}")
    
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)

    # 5. Convert text to sequences & 6. Padding
    print(f"Padding sequences to max length {MAX_SEQ_LEN}...")
    seq_p1 = tokenizer.texts_to_sequences(cleaned_p1)
    seq_p2 = tokenizer.texts_to_sequences(cleaned_p2)
    
    padded_p1 = pad_sequences(seq_p1, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    padded_p2 = pad_sequences(seq_p2, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    
    # 7. Train/Test Split
    print("Splitting data into train and validation sets...")
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
     
    # 8. Save Processed Data
    print("Saving processed data to disk...")
    np.savez_compressed(TRAIN_DATA_FILE, 
                        pair1=train_p1, 
                        pair2=train_p2, 
                        labels=train_labels)
                        
    np.savez_compressed(VAL_DATA_FILE, 
                        pair1=val_p1, 
                        pair2=val_p2, 
                        labels=val_labels, 
                        orig_pair1=val_orig_p1, 
                        orig_pair2=val_orig_p2)
    
    config = {'max_seq_len': MAX_SEQ_LEN, 'vocab_size': vocab_size, 'embedding_dim': EMBEDDING_DIM}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
    
    print("--- PREPROCESSING COMPLETE ---")
    print(f"Total time: {(time.time() - start_time):.2f} seconds")
    print(f"Training pairs: {len(train_labels)}")
    print(f"Validation pairs: {len(val_labels)}")
    return True # Return True on success


# --- 4. LOAD PROCESSED DATA ---

def load_processed_data():
    """Loads all preprocessed data files from disk."""
    print("\n--- LOADING PROCESSED DATA ---")
    
    with open(CONFIG_FILE, 'r') as f:
        model_config = json.load(f)
    
    vocab_size = model_config['vocab_size']
    max_seq_len = model_config['max_seq_len']
    embedding_dim = model_config['embedding_dim']
    
    with np.load(TRAIN_DATA_FILE) as data:
        train_p1 = data['pair1']
        train_p2 = data['pair2']
        train_labels = data['labels']
    
    with np.load(VAL_DATA_FILE, allow_pickle=True) as data:
        val_p1 = data['pair1']
        val_p2 = data['pair2']
        val_labels = data['labels']
        val_orig_p1 = data['orig_pair1']
        val_orig_p2 = data['orig_pair2']
    
    print("Data loaded successfully.")
    print(f"Vocab Size: {vocab_size}")
    
    # Pack data into tuples for easier handling
    train_data = (train_p1, train_p2, train_labels)
    val_data = (val_p1, val_p2, val_labels, val_orig_p1, val_orig_p2)
    config = (vocab_size, max_seq_len, embedding_dim)
    
    return train_data, val_data, config


# --- 5. MODEL 1: SIAMESE BiLSTM ---

def build_bilstm_model(vocab_size, embedding_dim, max_seq_len, lstm_units):
    """Builds the shared encoder for the Siamese BiLSTM model."""
    
    input_layer = Input(shape=(max_seq_len,))
    embedding_layer = Embedding(input_dim=vocab_size, 
                                output_dim=embedding_dim, 
                                input_length=max_seq_len)(input_layer)
    
    # FIX: Added return_sequences=True
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_layer)
    
    pooling = GlobalMaxPooling1D()(bilstm) 
    
    # This is the output shape from the pooling layer
    encoder_output_shape = pooling.shape[-1] 
    
    encoder = Model(inputs=input_layer, outputs=pooling)
    
    # Define the Siamese inputs
    input_a = Input(shape=(max_seq_len,))
    input_b = Input(shape=(max_seq_len,))
    
    # Get the encoded vectors
    vec_a = encoder(input_a)
    vec_b = encoder(input_b)
    
    # --- FIX: Added output_shape to Lambda layer ---
    distance = Lambda(
        lambda x: K.abs(x[0] - x[1]),
        output_shape=(encoder_output_shape,) 
    )([vec_a, vec_b])
    
    # Final classifier
    classifier = Dense(64, activation='relu')(distance)
    classifier = Dropout(0.3)(classifier)
    output = Dense(1, activation='sigmoid')(classifier)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


# --- 6. MODEL 2: SIAMESE BiLSTM with ATTENTION ---

def build_attention_model(vocab_size, embedding_dim, max_seq_len, lstm_units):
    """Builds the shared encoder for the Siamese Attention model."""
    
    input_layer = Input(shape=(max_seq_len,))
    embedding_layer = Embedding(input_dim=vocab_size, 
                                output_dim=embedding_dim, 
                                input_length=max_seq_len)(input_layer)
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_layer)
    
    attention_output = Attention()([bilstm, bilstm])
    pooling = GlobalAveragePooling1D()(attention_output)

    # This is the output shape from the pooling layer
    encoder_output_shape = pooling.shape[-1]

    encoder = Model(inputs=input_layer, outputs=pooling)
    
    input_a = Input(shape=(max_seq_len,))
    input_b = Input(shape=(max_seq_len,))
    
    vec_a = encoder(input_a)
    vec_b = encoder(input_b)
    
    # --- FIX: Added output_shape to Lambda layer ---
    distance = Lambda(
        lambda x: K.abs(x[0] - x[1]),
        output_shape=(encoder_output_shape,)
    )([vec_a, vec_b])
    
    classifier = Dense(64, activation='relu')(distance)
    classifier = Dropout(0.3)(classifier)
    output = Dense(1, activation='sigmoid')(classifier)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model


# --- 7. EVALUATION & COMPARISON ---

def plot_history(history, model_name):
    """Plots training and validation loss and accuracy."""
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name}_training_graphs.png")
    print(f"Saved training graphs to {model_name}_training_graphs.png")
    # plt.show() # Uncomment this if running interactively

def evaluate_model(model, val_data, model_name):
    """Calculates and prints all required evaluation metrics."""
    val_p1, val_p2, val_labels, _, _ = val_data
    
    print(f"\n--- Evaluating {model_name} ---")
    
    # Get predictions
    pred_probs = model.predict([val_p1, val_p2])
    preds = (pred_probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(val_labels, preds)
    precision = precision_score(val_labels, preds)
    recall = recall_score(val_labels, preds)
    f1 = f1_score(val_labels, preds)
    roc_auc = roc_auc_score(val_labels, pred_probs)
    pr_auc = average_precision_score(val_labels, pred_probs)
    
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(val_labels, preds, target_names=['Dissimilar (0)', 'Similar (1)']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, preds))
    
    return {"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}


# --- 8. QUALITATIVE ANALYSIS ---

def show_qualitative_results(model, val_data, model_name, num_examples=5):
    """Shows examples of correct and incorrect predictions."""
    val_p1, val_p2, val_labels, val_orig_p1, val_orig_p2 = val_data
    
    pred_probs = model.predict([val_p1, val_p2])
    preds = (pred_probs > 0.5).astype(int).flatten()
    
    print(f"\n--- Qualitative Analysis for {model_name} ---")
    
    # 1. Correctly Identified as SIMILAR (True Positives)
    print("\n--- Correctly Identified as SIMILAR (True Positives) ---")
    tp_indices = np.where((preds == 1) & (val_labels == 1))[0]
    if len(tp_indices) >= num_examples:
        for i, idx in enumerate(np.random.choice(tp_indices, num_examples, replace=False)):
            print(f"\nExample {i+1} (Pred: 1, True: 1):")
            print(f"  Clause 1: {val_orig_p1[idx][:150]}...")
            print(f"  Clause 2: {val_orig_p2[idx][:150]}...")
    else:
        print(f"Not enough examples (found {len(tp_indices)}).")

    # 2. Correctly Identified as DISSIMILAR (True Negatives)
    print("\n--- Correctly Identified as DISSIMILAR (True Negatives) ---")
    tn_indices = np.where((preds == 0) & (val_labels == 0))[0]
    if len(tn_indices) >= num_examples:
        for i, idx in enumerate(np.random.choice(tn_indices, num_examples, replace=False)):
            print(f"\nExample {i+1} (Pred: 0, True: 0):")
            print(f"  Clause 1: {val_orig_p1[idx][:150]}...")
            print(f"  Clause 2: {val_orig_p2[idx][:150]}...")
    else:
        print(f"Not enough examples (found {len(tn_indices)}).")

    # 3. Incorrectly Identified as SIMILAR (False Positives)
    print("\n--- Incorrectly Identified as SIMILAR (False Positives) ---")
    fp_indices = np.where((preds == 1) & (val_labels == 0))[0]
    if len(fp_indices) >= num_examples:
        for i, idx in enumerate(np.random.choice(fp_indices, num_examples, replace=False)):
            print(f"\nExample {i+1} (Pred: 1, True: 0):")
            print(f"  Clause 1: {val_orig_p1[idx][:150]}...")
            print(f"  Clause 2: {val_orig_p2[idx][:150]}...")
    else:
        print(f"Not enough examples (found {len(fp_indices)}).")

    # 4. Incorrectly Identified as DISSIMILAR (False Negatives)
    print("\n--- Incorrectly Identified as DISSIMILAR (False Negatives) ---")
    fn_indices = np.where((preds == 0) & (val_labels == 1))[0]
    if len(fn_indices) >= num_examples:
        for i, idx in enumerate(np.random.choice(fn_indices, num_examples, replace=False)):
            print(f"\nExample {i+1} (Pred: 0, True: 1):")
            print(f"  Clause 1: {val_orig_p1[idx][:150]}...")
            print(f"  Clause 2: {val_orig_p2[idx][:150]}...")
    else:
        print(f"Not enough examples (found {len(fn_indices)}).")


# --- 9. MAIN EXECUTION ---

def main():
    """Main function to run the entire pipeline."""
    
    # Step 1: Preprocessing
    # We check if all the necessary files already exist to save time
    required_files = [TRAIN_DATA_FILE, VAL_DATA_FILE, TOKENIZER_FILE, CONFIG_FILE]
    all_files_exist = all(os.path.exists(f) for f in required_files)
    
    if not all_files_exist:
        print("One or more processed files are missing. Running preprocessing...")
        success = run_preprocessing()
        if not success:
            print("Preprocessing failed. Exiting script.")
            return # <-- This will stop the script if preprocessing fails
    else:
        print("Processed data files found. Skipping preprocessing.")
    
    # Step 2: Load Data
    try:
        train_data, val_data, config = load_processed_data()
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Please delete train_data.npz, val_data.npz, tokenizer.json, and model_config.json and re-run the script.")
        return
        
    train_p1, train_p2, train_labels = train_data
    val_p1, val_p2, val_labels, _, _ = val_data
    vocab_size, max_seq_len, embedding_dim = config
    
    
    # ---!!! EMERGENCY SPEED-UP: SUBSET THE TRAINING DATA !!!---
    print(f"\n--- SPEED-UP: Using a subset of {TRAIN_SUBSET_SIZE} training pairs ---")
    if len(train_labels) > TRAIN_SUBSET_SIZE:
        train_p1 = train_p1[:TRAIN_SUBSET_SIZE]
        train_p2 = train_p2[:TRAIN_SUBSET_SIZE]
        train_labels = train_labels[:TRAIN_SUBSET_SIZE]
    print(f"New training set size: {len(train_labels)}")
    # ---!!! END SPEED-UP ---


    # --- Step 3: Model 1 (BiLSTM) Training ---
    # --- MODIFICATION: Skip training if model file exists ---
    if not os.path.exists(MODEL_BILSTM_FILE):
        print("\n--- MODEL 1: Siamese BiLSTM ---")
        model_bilstm = build_bilstm_model(vocab_size, embedding_dim, max_seq_len, LSTM_UNITS)
        model_bilstm.summary()
        
        callbacks_bilstm = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=MODEL_BILSTM_FILE, monitor='val_accuracy', save_best_only=True)
        ]
        
        print("Training Model 1: Siamese BiLSTM...")
        start_time_bilstm = time.time()
        history_bilstm = model_bilstm.fit(
            [train_p1, train_p2], 
            train_labels, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=([val_p1, val_p2], val_labels),
            callbacks=callbacks_bilstm
        )
        print(f"BiLSTM Training Time: {(time.time() - start_time_bilstm):.2f} seconds")
        # --- Plot history right after training ---
        plot_history(history_bilstm, "Siamese_BiLSTM")
    else:
        print(f"\n--- SKIPPING BiLSTM Training: {MODEL_BILSTM_FILE} already exists. ---")
        history_bilstm = None # No history to plot if we skip training

    # --- Step 4: Model 2 (Attention) Training ---
    # --- MODIFICATION: Skip training if model file exists ---
    if not os.path.exists(MODEL_ATTENTION_FILE):
        print("\n--- MODEL 2: Siamese BiLSTM with Attention ---")
        model_attention = build_attention_model(vocab_size, embedding_dim, max_seq_len, LSTM_UNITS)
        model_attention.summary()
        
        callbacks_attention = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(filepath=MODEL_ATTENTION_FILE, monitor='val_accuracy', save_best_only=True)
        ]
        
        print("Training Model 2: Siamese BiLSTM with Attention...")
        start_time_attn = time.time()
        history_attention = model_attention.fit(
            [train_p1, train_p2], 
            train_labels, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=([val_p1, val_p2], val_labels),
            callbacks=callbacks_attention
        )
        print(f"Attention Model Training Time: {(time.time() - start_time_attn):.2f} seconds")
        # --- Plot history right after training ---
        plot_history(history_attention, "Siamese_Attention_BiLSTM")
    else:
        print(f"\n--- SKIPPING Attention Training: {MODEL_ATTENTION_FILE} already exists. ---")
        history_attention = None # No history to plot if we skip training


    # --- Step 5: Evaluation & Comparison ---
    print("\n--- FINAL EVALUATION ---")
    
    # --- MODIFICATION: Define variables as None first ---
    best_model_bilstm = None
    best_model_attention = None
    
    # Load the *best* saved weights for a fair comparison
    try:
        # --- MODIFICATION: Added safe_mode=False ---
        print(f"Loading {MODEL_BILSTM_FILE}...")
        best_model_bilstm = load_model(MODEL_BILSTM_FILE, safe_mode=False)
        metrics_bilstm = evaluate_model(best_model_bilstm, val_data, "Siamese BiLSTM")
    except Exception as e:
        print(f"Error loading or evaluating BiLSTM model: {e}")
        metrics_bilstm = {"accuracy": 0, "f1": 0, "roc_auc": 0}

    try:
        # --- MODIFICATION: Added safe_mode=False ---
        print(f"Loading {MODEL_ATTENTION_FILE}...")
        best_model_attention = load_model(MODEL_ATTENTION_FILE, safe_mode=False)
        metrics_attention = evaluate_model(best_model_attention, val_data, "Siamese BiLSTM with Attention")
    except Exception as e:
        print(f"Error loading or evaluating Attention model: {e}")
        metrics_attention = {"accuracy": 0, "f1": 0, "roc_auc": 0}
        
    print("\n--- Performance Comparison ---")
    print(f"| Metric   | BiLSTM   | BiLSTM w/ Attention |")
    print(f"|----------|----------|---------------------|")
    print(f"| Accuracy | {metrics_bilstm['accuracy']:.4f}   | {metrics_attention['accuracy']:.4f}            |")
    print(f"| F1-Score | {metrics_bilstm['f1']:.4f}   | {metrics_attention['f1']:.4f}            |")
    print(f"| ROC-AUC  | {metrics_bilstm['roc_auc']:.4f}   | {metrics_attention['roc_auc']:.4f}            |")
    
    # --- Step 6: Plotting ---
    # --- MODIFICATION: Check if history objects exist ---
    # Only plot if we just trained (history object exists)
    if history_bilstm:
        plot_history(history_bilstm, "Siamese_BiLSTM")
    else:
        print(f"Skipping plot for BiLSTM (no training history). Graph file '{'Siamese_BiLSTM'}_training_graphs.png' may be from a previous run.")
        
    if history_attention:
        plot_history(history_attention, "Siamese_Attention_BiLSTM")
    else:
        print(f"Skipping plot for Attention (no training history). Graph file '{'Siamese_Attention_BiLSTM'}_training_graphs.png' may be from a previous run.")

    
    # --- Step 7: Qualitative Analysis ---
    # --- MODIFICATION: Check if model objects exist ---
    try:
        if best_model_bilstm:
            show_qualitative_results(best_model_bilstm, val_data, "Siamese BiLSTM")
        else:
            raise ValueError("Model object 'best_model_bilstm' is None (loading failed).")
    except Exception as e:
        print(f"Error during qualitative analysis for BiLSTM: {e}")
        
    try:
        if best_model_attention:
            show_qualitative_results(best_model_attention, val_data, "Siamese BiLSTM with Attention")
        else:
            raise ValueError("Model object 'best_model_attention' is None (loading failed).")
    except Exception as e:
        print(f"Error during qualitative analysis for Attention model: {e}")
    
    print("\n--- SCRIPT COMPLETE ---")

if __name__ == "__main__":
    main()