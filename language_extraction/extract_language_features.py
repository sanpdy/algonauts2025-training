import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from preprocessing.config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(config):
    model_name = config.feature_extraction.language.model_name  # sentence-transformers/all-mpnet-base-v2
    model = SentenceTransformer(model_name, device=device)
    return model


def get_transcript_paths(config):
    input_dir = config.feature_extraction.language.input_dir
    return glob.glob(f"{input_dir}/**/*.tsv", recursive=True)


def filter_unprocessed_transcripts(paths, config):
    unprocessed = []
    input_dir = os.path.normpath(config.feature_extraction.language.input_dir)
    save_dir  = os.path.normpath(config.feature_extraction.language.save_dir)
    
    for p in paths:
        base_name = os.path.splitext(os.path.basename(p))[0]
        episode_dir = os.path.join(save_dir, base_name)
        
        if not os.path.exists(episode_dir):
            unprocessed.append(p)
        else:
            try:
                df = pd.read_csv(p, sep="\t")
                expected_chunks = len(df)
                existing_chunks = len([f for f in os.listdir(episode_dir) if f.endswith('.pt')])
                
                if existing_chunks < expected_chunks:
                    unprocessed.append(p)
            except Exception:
                unprocessed.append(p)
    
    return unprocessed


def save_feature(features, transcript_path, config):
    input_dir = os.path.normpath(config.feature_extraction.language.input_dir)
    save_dir  = os.path.normpath(config.feature_extraction.language.save_dir)
    
    # Create individual .pt files for each video chunk
    # Assuming transcript filename format like: season1_episode1.tsv
    base_name = os.path.splitext(os.path.basename(transcript_path))[0]
    episode_dir = os.path.join(save_dir, base_name)
    os.makedirs(episode_dir, exist_ok=True)
    
    embeddings = features["sentence_embeddings"] if "sentence_embeddings" in features else features["contextual_sentence_embeddings"]
    
    # Save each chunk as separate .pt file
    for i, embedding in enumerate(embeddings):
        chunk_filename = f"part{i+1:03d}.pt"  # part001.pt, part002.pt, etc.
        chunk_path = os.path.join(episode_dir, chunk_filename)
        torch.save(embedding, chunk_path)


def extract_language_features(transcript_path, model, config):
    df = pd.read_csv(transcript_path, sep="\t")
    print(f"Processing {os.path.basename(transcript_path)}: {len(df)} rows")
    
    # Handle missing text
    texts = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        if pd.notna(row["text_per_tr"]) and row["text_per_tr"].strip():
            texts.append(row["text_per_tr"].strip())
            valid_indices.append(idx)
    
    print(f"  Valid texts: {len(texts)}/{len(df)}")
    if texts:
        print(f"  Sample text: '{texts[0][:50]}...'")
        print(f"  Using device: {model.device}")
    
    # Initialize embeddings array
    embedding_dim = model.get_sentence_embedding_dimension()
    embeddings = np.full((len(df), embedding_dim), np.nan, dtype=np.float32)
    
    # Extract embeddings for valid texts in batches
    if texts:
        batch_size = getattr(config.feature_extraction.language, 'encoding_batch_size', 32)
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Optional: L2 normalize embeddings
            )
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            
            # Place embeddings back in their original positions
            for i, idx in enumerate(valid_indices):
                embeddings[idx] = all_embeddings[i]
    
    return {"sentence_embeddings": embeddings}


def extract_language_features_with_context(transcript_path, model, config):
    """
    Alternative version that considers context by combining multiple consecutive utterances
    """
    df = pd.read_csv(transcript_path, sep="\t")
    print(f"Processing {os.path.basename(transcript_path)}: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  First few rows of text_per_tr:")
    for i in range(min(3, len(df))):
        text_val = df.iloc[i]["text_per_tr"] if "text_per_tr" in df.columns else "COLUMN NOT FOUND"
        print(f"    Row {i}: {repr(text_val)}")
    
    # Parameters for context window
    context_window = getattr(config.feature_extraction.language, 'context_window', 3)
    
    texts = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        if pd.notna(row["text_per_tr"]) and row["text_per_tr"].strip():
            # Get context window around current utterance
            start_idx = max(0, idx - context_window // 2)
            end_idx = min(len(df), idx + context_window // 2 + 1)
            
            context_texts = []
            for j in range(start_idx, end_idx):
                if pd.notna(df.iloc[j]["text_per_tr"]) and df.iloc[j]["text_per_tr"].strip():
                    context_texts.append(df.iloc[j]["text_per_tr"].strip())
            
            if context_texts:
                # Join context with special separator
                combined_text = " [SEP] ".join(context_texts)
                texts.append(combined_text)
                valid_indices.append(idx)
    
    print(f"  Valid texts with context: {len(texts)}/{len(df)}")
    if texts:
        print(f"  Sample contextual text: '{texts[0][:100]}...'")
        print(f"  Using device: {model.device}")
    
    # Initialize embeddings array
    embedding_dim = model.get_sentence_embedding_dimension()
    embeddings = np.full((len(df), embedding_dim), np.nan, dtype=np.float32)
    
    # Extract embeddings
    if texts:
        batch_size = getattr(config.feature_extraction.language, 'encoding_batch_size', 32)
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with context"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
        
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            for i, idx in enumerate(valid_indices):
                embeddings[idx] = all_embeddings[i]
    
    return {"contextual_sentence_embeddings": embeddings}


def process_transcripts_in_batches(paths, model, config, batch_size=16):
    use_context = getattr(config.feature_extraction.language, 'use_context', False)
    
    for i in tqdm(range(0, len(paths), batch_size), desc="Processing transcripts"):
        batch = paths[i : i + batch_size]
        for p in batch:
            try:
                if use_context:
                    feats = extract_language_features_with_context(p, model, config)
                else:
                    feats = extract_language_features(p, model, config)
                save_feature(feats, p, config)
            except Exception as e:
                print(f"Error processing {p}: {e}")


if __name__ == "__main__":
    model = get_model(config)
    all_paths = get_transcript_paths(config)
    print(f"Found {len(all_paths)} transcript files")
    
    todo = filter_unprocessed_transcripts(all_paths, config)
    print(f"Unprocessed: {len(todo)}")
    
    if todo:
        for p in todo[:3]:
            print(" ", p)
        
        bs = getattr(config.feature_extraction.language, "batch_size", 16)
        process_transcripts_in_batches(todo, model, config, bs)
        print("Done!")
    else:
        print("All transcripts processed.")