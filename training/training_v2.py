# training_v2.py

from __future__ import print_function
import os
from pathlib import Path
import numpy as np
import h5py
from load_embeddings import load_features, preprocess_features, perform_pca
from load_fmri import load_fmri, align_features_and_fmri_samples
from load_models import load_baseline_encoding_models, train_encoding
from load_models import compute_encoding_accuracy

if __name__ == "__main__":
    root_data_dir = Path("/home/sankalp/algonauts2025/data")
    n_components = 250
    subject = 1
    modality = "audio"  # @param ["visual", "audio", "language", "all"]
    excluded_samples_start = 5
    excluded_samples_end = 5
    hrf_delay = 3
    stimulus_window = 6

    movies_train = [
        "friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05",
        "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"
    ]
    movies_val = ["friends-s06"]
    fmri = load_fmri(root_data_dir, subject)
    if modality == "audio":
        episode_names = []
        episode_counts = []
        h5_path = root_data_dir / "algonauts_2025.competitors" / "stimuli" / "train_data" / "features" / "whisper_w16.h5"
        with h5py.File(str(h5_path), 'r') as data:
            total_samples = 0
            for episode in data.keys():
                if episode.startswith("movie10_"):
                    ep_key = episode[8:]
                elif episode.startswith("friends_"):
                    ep_key = episode[8:]
                else:
                    ep_key = episode

                grp = data[episode]
                if isinstance(grp, h5py.Dataset):
                    feats = np.asarray(grp)
                else:
                    feats = np.asarray(grp["audio"])

                episode_names.append(ep_key)
                episode_counts.append(feats.shape[0])
                total_samples += feats.shape[0]
                print(f"Features shape for episode {ep_key}: {feats.shape}")

            print(f"audio features total samples: {total_samples}")
            print("Dictionary structure: implicit, will reconstruct after PCA")

        features_all = load_features(str(root_data_dir), "audio")
        # features_all.shape == (total_samples, original_dim)

        prepr_all = preprocess_features(features_all)
        # prepr_all.shape == (total_samples, original_dim)

        features_pca_all = perform_pca(prepr_all, n_components)
        # features_pca_all.shape == (total_samples, n_components)

        pca_dict = {}
        start = 0
        for ep_key, count in zip(episode_names, episode_counts):
            end = start + count
            pca_dict[ep_key] = features_pca_all[start:end].astype(np.float32)
            print(f"Episode {ep_key}: PCA shape = {pca_dict[ep_key].shape}")
            start = end

        features_dict = {"audio": pca_dict}

    elif modality == "language":
        episode_names = []
        episode_counts = []

        h5_path = root_data_dir / "algonauts_2025.competitors" / "stimuli" / "train_data" / "features" / "mistral_w16.h5"
        with h5py.File(str(h5_path), 'r') as data:
            total_samples = 0
            for episode in data.keys():
                if episode.startswith("movie10_"):
                    ep_key = episode[8:]
                elif episode.startswith("friends_"):
                    ep_key = episode[8:]
                else:
                    ep_key = episode

                grp = data[episode]
                if isinstance(grp, h5py.Dataset):
                    feats = np.asarray(grp)
                else:
                    raise RuntimeError("Unexpected H5 structure for language.")

                episode_names.append(ep_key)
                episode_counts.append(feats.shape[0])
                total_samples += feats.shape[0]
                print(f"Features shape for episode {ep_key}: {feats.shape}")

            print(f"language features total samples: {total_samples}")
            print("Dictionary structure: implicit, will reconstruct after PCA")

        features_all = load_features(str(root_data_dir), "language")
        prepr_all = preprocess_features(features_all)
        features_pca_all = perform_pca(prepr_all, n_components)

        pca_dict = {}
        start = 0
        for ep_key, count in zip(episode_names, episode_counts):
            end = start + count
            pca_dict[ep_key] = features_pca_all[start:end].astype(np.float32)
            print(f"Episode {ep_key}: PCA shape = {pca_dict[ep_key].shape}")
            start = end

        features_dict = {"language": pca_dict}

    elif modality == "all":
        episode_names_a = []
        episode_counts_a = []

        h5_audio = root_data_dir / "algonauts_2025.competitors" / "stimuli" / "train_data" / "features" / "whisper_w16.h5"
        with h5py.File(str(h5_audio), 'r') as data:
            total_samples = 0
            for episode in data.keys():
                if episode.startswith("movie10_"):
                    ep_key = episode[8:]
                elif episode.startswith("friends_"):
                    ep_key = episode[8:]
                else:
                    ep_key = episode

                grp = data[episode]
                if isinstance(grp, h5py.Dataset):
                    feats = np.asarray(grp)
                else:
                    feats = np.asarray(grp["audio"])

                episode_names_a.append(ep_key)
                episode_counts_a.append(feats.shape[0])
                total_samples += feats.shape[0]
            print(f"audio features total samples: {total_samples}")

        features_all_a = load_features(str(root_data_dir), "audio")
        prepr_all_a = preprocess_features(features_all_a)
        features_pca_all_a = perform_pca(prepr_all_a, n_components)

        pca_audio = {}
        start = 0
        for ep_key, count in zip(episode_names_a, episode_counts_a):
            end = start + count
            pca_audio[ep_key] = features_pca_all_a[start:end].astype(np.float32)
            start = end

        episode_names_l = []
        episode_counts_l = []

        h5_lang = root_data_dir / "algonauts_2025.competitors" / "stimuli" / "train_data" / "features" / "mistral_w16.h5"
        with h5py.File(str(h5_lang), 'r') as data:
            total_samples = 0
            for episode in data.keys():
                if episode.startswith("movie10_"):
                    ep_key = episode[8:]
                elif episode.startswith("friends_"):
                    ep_key = episode[8:]
                else:
                    ep_key = episode

                grp = data[episode]
                if isinstance(grp, h5py.Dataset):
                    feats = np.asarray(grp)
                else:
                    raise RuntimeError("Unexpected H5 structure for language.")

                episode_names_l.append(ep_key)
                episode_counts_l.append(feats.shape[0])
                total_samples += feats.shape[0]
            print(f"language features total samples: {total_samples}")

        features_all_l = load_features(str(root_data_dir), "language")
        prepr_all_l = preprocess_features(features_all_l)
        features_pca_all_l = perform_pca(prepr_all_l, n_components)

        pca_language = {}
        start = 0
        for ep_key, count in zip(episode_names_l, episode_counts_l):
            end = start + count
            pca_language[ep_key] = features_pca_all_l[start:end].astype(np.float32)
            start = end

        # Combine both modality dicts:
        features_dict = {"audio": pca_audio, "language": pca_language}

    else:
        raise ValueError(f"Unsupported modality '{modality}'")

    # -------------------------------
    # 3) Align features with fMRI
    # -------------------------------
    features_train, fmri_train = align_features_and_fmri_samples(
        features_dict,
        fmri,
        excluded_samples_start,
        excluded_samples_end,
        hrf_delay,
        stimulus_window,
        movies_train
    )

    # Print shapes for debugging
    print("Training fMRI responses shape:")
    print(fmri_train.shape)
    print("(Train samples × Parcels)")
    print("\nTraining stimulus features shape:")
    print(features_train.shape)
    print("(Train samples × Features)\n")

    print(f"Subject {subject} fMRI movies splits name and shape:")
    for key, value in fmri.items():
        print(key + " " + str(value.shape))

    # -------------------------------
    # 4) Train encoding model
    # -------------------------------
    model = train_encoding(features_train, fmri_train)
    del features_train, fmri_train

    # -------------------------------
    # 5) Validation
    # -------------------------------
    features_val, fmri_val = align_features_and_fmri_samples(
        features_dict,
        fmri,
        excluded_samples_start,
        excluded_samples_end,
        hrf_delay,
        stimulus_window,
        movies_val
    )

    # We no longer need features_dict or fmri in memory
    del features_dict, fmri

    print("\nValidation fMRI responses shape:", fmri_val.shape)
    print("(Validation samples × Parcels)")
    print("\nValidation stimulus features shape:", features_val.shape)
    print("(Validation samples × Features)")

    fmri_val_pred = model.predict(features_val)

    print("\nValidation fMRI responses shape:", fmri_val.shape)
    print("(Validation samples × Parcels)")
    print("\nValidation predicted fMRI responses shape:", fmri_val_pred.shape)
    print("(Validation samples × Parcels)")

    compute_encoding_accuracy(root_data_dir, fmri_val, fmri_val_pred, subject, modality)
