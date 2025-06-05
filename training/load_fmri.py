from __future__ import print_function
import os
import numpy as np
import h5py

def load_fmri(root_data_dir, subject):
    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the fMRI responses
    fmri_friends = h5py.File(fmri_dir, 'r')
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
    fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'func', fmri_file)
    # Load the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, 'r')
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10

    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = 'figures' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = 'life' + format(s+1, '02')
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri

def align_features_and_fmri_samples(
    features, fmri, excluded_samples_start,
    excluded_samples_end, hrf_delay, stimulus_window, movies
):
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)

    ### Loop across movies ###
    for movie in movies:
        # Get the ID (e.g., "s01" or "wolf12") from "friends-s01" / "movie10-wolf12"
        if movie[:7] == 'friends':
            mid = movie[8:]
        elif movie[:7] == 'movie10':
            mid = movie[8:]
        else:
            mid = movie

        movie_splits = [key for key in fmri if mid in key[:len(mid)]]

        ### Loop over splits ###
        for split in movie_splits:
            fmri_split = fmri[split]
            # Exclude start/end TRs
            fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            ### Loop over each fMRI TR of this split ###
            for s in range(len(fmri_split)):
                f_all = np.empty(0)

                ### For each modality in features (e.g., "audio", "language", etc.) ###
                for mod in features.keys():
                    if mod == 'visual' or mod == 'audio':
                        # Determine window of stimulus‐feature chunks
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s + excluded_samples_start - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window

                        # If we run out of stimulus samples, use the last window
                        if idx_end > (len(features[mod][split])):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window

                        f = features[mod][split][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    elif mod == 'language':
                        # For language, only one embedding per TR is used (shifted by hrf_delay)
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay

                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1, :]
                        else:
                            f = features[mod][split][idx]

                        f_all = np.append(f_all, f.flatten())

                aligned_features.append(f_all)

    aligned_features = np.asarray(aligned_features, dtype=np.float32)
    return aligned_features, aligned_fmri
