import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
import torch
from moviepy.editor import VideoFileClip
import glob
from tqdm import tqdm
import numpy as np
from preprocessing.config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_feature_extractor(config):
    model_name = config.feature_extraction.audio.model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioXVector.from_pretrained(model_name).to(device)
    model.eval()
    return model, feature_extractor


def extract_audio_from_video(video_path, sampling_rate):
    with VideoFileClip(video_path) as clip:
        audio = clip.audio.to_soundarray(fps=sampling_rate)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        audio = audio.astype(np.float32)
        return audio  # shape: [length,]


def save_feature(features, video_path, input_dir, save_dir):
    input_dir = os.path.normpath(input_dir)
    save_dir = os.path.normpath(save_dir)
    video_path = os.path.normpath(video_path)
    save_path = video_path.replace(input_dir, save_dir)
    parent_dir = os.path.dirname(save_path)
    os.makedirs(parent_dir, exist_ok=True)
    save_path = save_path.replace(".mp4", ".pt")
    torch.save(features, save_path)


def get_video_paths(config):
    input_dir = config.feature_extraction.input_dir
    video_files = glob.glob(f"{input_dir}/**/*.mp4", recursive=True)
    return video_files


def filter_unprocessed_videos(video_paths, config):
    unprocessed = []
    input_dir = os.path.normpath(config.feature_extraction.input_dir)
    save_dir = os.path.normpath(config.feature_extraction.save_dir)
    
    for video_path in video_paths:
        video_path_norm = os.path.normpath(video_path)
        save_path = video_path_norm.replace(input_dir, save_dir).replace(".mp4", ".pt")
        if not os.path.exists(save_path):
            unprocessed.append(video_path)
    
    return unprocessed


def process_videos_in_batches(video_paths, model, feature_extractor, config, batch_size=16):
    sampling_rate = config.feature_extraction.audio.sampling_rate
    input_dir = config.feature_extraction.input_dir
    save_dir = config.feature_extraction.save_dir
    
    for i in tqdm(range(0, len(video_paths), batch_size), desc="Processing batches"):
        batch_paths = video_paths[i:i+batch_size]
        batch_audio = []
        valid_paths = []
    
        for video_path in batch_paths:
            try:
                audio = extract_audio_from_video(video_path, sampling_rate)
                batch_audio.append(audio)
                valid_paths.append(video_path)
            except Exception as e:
                print(f"Error extracting audio from {video_path}: {e}")
        
        if not batch_audio:
            continue
        
        try:
            inputs = feature_extractor(
                batch_audio,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            batch_features = outputs.embeddings.detach().cpu()
            
            for j, (features, video_path) in enumerate(zip(batch_features, valid_paths)):
                try:
                    feature_tensor = features.unsqueeze(0)
                    save_feature(feature_tensor, video_path, input_dir, save_dir)
                except Exception as e:
                    print(f"Error saving features for {video_path}: {e}")
                    
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            print("Falling back to individual processing for this batch...")
            for video_path in valid_paths:
                try:
                    process_single_video(video_path, model, feature_extractor, config)
                except Exception as single_e:
                    print(f"Error processing individual video {video_path}: {single_e}")


def process_single_video(video_path, model, feature_extractor, config):
    sampling_rate = config.feature_extraction.audio.sampling_rate
    input_dir = config.feature_extraction.input_dir
    save_dir = config.feature_extraction.save_dir
    
    audio_waveform = extract_audio_from_video(video_path, sampling_rate)
    
    inputs = feature_extractor(
        [audio_waveform],
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    audio_features = outputs.embeddings.detach().cpu()
    
    save_feature(audio_features, video_path, input_dir, save_dir)


if __name__ == "__main__":
    model, feature_extractor = get_model_and_feature_extractor(config)
    all_video_paths = get_video_paths(config)
    print(f"Found {len(all_video_paths)} total videos")
    video_paths = filter_unprocessed_videos(all_video_paths, config)
    print(f"Found {len(video_paths)} unprocessed videos")
    
    if video_paths:
        print(f"First 3 unprocessed videos:\n", "\n".join(video_paths[:3]))
        batch_size = getattr(config.feature_extraction.audio, 'batch_size', 128)
        process_videos_in_batches(video_paths, model, feature_extractor, config, batch_size)
        
        print("Processing complete!")
    else:
        print("All videos have already been processed!")