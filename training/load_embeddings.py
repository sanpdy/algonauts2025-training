from __future__ import print_function
import os
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_features(root_data_dir, modality):
    os.path.join(root_data_dir, 'algonauts_2025.competitors', 'stimuli', 'train_data', 'features')
    if modality == 'audio':
        data_dir = os.path.join(root_data_dir, 'whisper_w16.h5')
    elif modality == 'language':
        data_dir = os.path.join(root_data_dir, 'mistral_w16.h5')
    
    ### Load the stimulus features ###
    features_dict = {modality: {}}
    
    with h5py.File(data_dir, 'r') as data:
        total_samples = 0
        for episode in data.keys():
            episode_group = data[episode]
            if modality != 'language':
                if isinstance(episode_group, h5py.Dataset):
                    features = np.asarray(episode_group)
                else:
                    features = np.asarray(episode_group[modality])
            else:
                if isinstance(episode_group, h5py.Dataset):
                    features = np.asarray(episode_group)
            
            # Convert episode name to match fMRI naming convention
            # e.g., "movie10_wolf01" -> "wolf01"
            if episode.startswith('movie10_'):
                episode_key = episode[8:]  # Remove 'movie10_' prefix
            elif episode.startswith('friends_'):
                episode_key = episode[8:]  # Remove 'friends_' prefix
            else:
                episode_key = episode
            
            features_dict[modality][episode_key] = features.astype(np.float32)
            total_samples += features.shape[0]
            print(f"Features shape for episode {episode_key}: {features.shape}")
    
    print(f"{modality} features total samples: {total_samples}")
    print('Dictionary structure: {modality: {episode: features_array}}')

    ### Output ###
    return features_dict

def load_features_concatenated(root_data_dir, modality):
    os.path.join(root_data_dir, 'algonauts_2025.competitors', 'stimuli', 'train_data', 'features')
    if modality == 'audio':
        data_dir = os.path.join(root_data_dir, 'whisper_w16.h5')
    elif modality == 'language':
        data_dir = os.path.join(root_data_dir, 'mistral_w16.h5')
    
    ### Load the stimulus features ###
    with h5py.File(data_dir, 'r') as data:
        
        features_list = []
        for episode in data.keys():
            episode_group = data[episode]
            if modality != 'language':
                if isinstance(episode_group, h5py.Dataset):
                    features = np.asarray(episode_group)
                else:
                    features = np.asarray(episode_group[modality])
            else:
                if isinstance(episode_group, h5py.Dataset):
                    features = np.asarray(episode_group)
            
            features_list.append(features)
            print(f"Features shape for episode {episode}: {features.shape}")

    features = np.concatenate(features_list, axis=0)
    print(f"{modality} features original shape: {features.shape}")
    print('(Movie samples × Features)')

    ### Output ###
    return features

def preprocess_features_dict(features_dict, modality):
    preprocessed_dict = {modality: {}}
    
    # First, collect all features to fit the scaler
    all_features = []
    for episode_key, features in features_dict[modality].items():
        all_features.append(features)
    
    # Concatenate all features to fit scaler
    concatenated_features = np.concatenate(all_features, axis=0)
    concatenated_features = np.nan_to_num(concatenated_features)
    
    # Fit scaler on all data
    scaler = StandardScaler()
    scaler.fit(concatenated_features)
    
    # Apply preprocessing to each episode separately
    for episode_key, features in features_dict[modality].items():
        features_clean = np.nan_to_num(features)
        preprocessed_features = scaler.transform(features_clean)
        preprocessed_dict[modality][episode_key] = preprocessed_features.astype(np.float32)
    
    return preprocessed_dict

def preprocess_features(features):
    features = np.nan_to_num(features)
    scaler = StandardScaler()
    preprocessed_features = scaler.fit_transform(features)
    return preprocessed_features

def perform_pca(features, n_components=100):
    if n_components > features.shape[1]:
        n_components = features.shape[1]

    pca = PCA(n_components, random_state=20200220)
    features_pca = pca.fit_transform(features)
    print(f"PCA features shape: {features_pca.shape}")
    print(("Movie samples × PCA components)"))
    return features_pca