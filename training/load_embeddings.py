from __future__ import print_function
import os
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_features(root_data_dir, modality):
    features_dir = os.path.join(
        root_data_dir,
        "algonauts_2025.competitors",
        "stimuli",
        "train_data",
        "features"
    )
    if modality == "audio":
        h5_path = os.path.join(features_dir, "whisper_w16.h5")
    elif modality == "language":
        h5_path = os.path.join(features_dir, "mistral_w16.h5")
    else:
        raise ValueError(f"Unsupported modality '{modality}'. Choose 'audio' or 'language'.")

    all_features = []
    with h5py.File(h5_path, 'r') as data:
        total_samples = 0
        for episode in data.keys():
            grp = data[episode]
            if modality != "language":
                if isinstance(grp, h5py.Dataset):
                    feats = np.asarray(grp)
                else:
                    feats = np.asarray(grp[modality])
            else:
                if isinstance(grp, h5py.Dataset):
                    feats = np.asarray(grp)
                else:
                    raise RuntimeError("Unexpected H5 structure for language features.")

            all_features.append(feats.astype(np.float32))
            total_samples += feats.shape[0]
            print(f"Features shape for episode {episode}: {feats.shape}")

    concatenated = np.concatenate(all_features, axis=0)
    print(f"{modality} features original shape: {concatenated.shape}")
    print("(Movie samples × Features)")
    return concatenated  # dtype = float32 already

def preprocess_features(features):
    features = np.nan_to_num(features)
    scaler = StandardScaler()
    prepr = scaler.fit_transform(features)
    return prepr.astype(np.float32)

def perform_pca(features, n_components=100):
    if n_components > features.shape[1]:
        n_components = features.shape[1]

    pca = PCA(n_components=n_components, random_state=20200220)
    features_pca = pca.fit_transform(features)
    print(f"PCA features shape: {features_pca.shape}")
    print("(Movie samples × PCA components)")
    return features_pca.astype(np.float32)
