from __future__ import print_function
import os
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tqdm

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

def load_stimulus_features_friends_s7(root_data_dir):
    features_friends_s7 = {}

    modalities = ['visual', 'audio', 'language']

    for modality in modalities:
        features_friends_s7[modality] = {}
        if modality == 'audio':
            data_dir = os.path.join(root_data_dir, 
                                    'algonauts_2025.competitors', 
                                    'stimuli', 'train_data', 'features', 
                                    'whisper_w16.h5')
        elif modality == 'language':
            data_dir = os.path.join(root_data_dir, 
                                    'algonauts_2025.competitors', 
                                    'stimuli', 'train_data', 'features', 
                                    'mistral_w16.h5')

        with h5py.File(data_dir, 'r') as data:
            for episode in data.keys():
                # Remove 'movie10_' or 'friends_' prefix to match fMRI naming convention
                if episode.startswith('movie10_'):
                    episode_key = episode[8:]
                elif episode.startswith('friends_'):
                    episode_key = episode[8:]
                else:
                    episode_key = episode
                
                # Only include Friends Season 7 episodes
                if episode_key.startswith('s07'):
                    episode_group = data[episode]
                    if isinstance(episode_group, h5py.Dataset):
                        features = np.asarray(episode_group)
                    else:
                        features = np.asarray(episode_group[modality])

                    features_friends_s7[modality][episode_key] = features.astype(np.float32)
                    print(f"Loaded {modality} features for episode {episode_key}: {features.shape}")

    return features_friends_s7

def align_features_and_fmri_samples_friends_s7(features_friends_s7,
    root_data_dir):

    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}

    ### HRF delay ###
    # fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
    # that reflects changes in blood oxygenation levels in response to activity
    # in the brain. Blood flow increases to a given brain region in response to
    # its activity. This vascular response, which follows the hemodynamic
    # response function (HRF), takes time. Typically, the HRF peaks around 5–6
    # seconds after a neural event: this delay reflects the time needed for
    # blood oxygenation changes to propagate and for the fMRI signal to capture
    # them. Therefore, this parameter introduces a delay between stimulus chunks
    # and fMRI samples for a better correspondence between input stimuli and the
    # brain response. For example, with a hrf_delay of 3, if the stimulus chunk
    # of interest is 17, the corresponding fMRI sample will be 20.
    hrf_delay = 3

    ### Stimulus window ###
    # stimulus_window indicates how many stimulus feature samples are used to
    # model each fMRI sample, starting from the stimulus sample corresponding to
    # the fMRI sample of interest, minus the hrf_delay, and going back in time.
    # For example, with a stimulus_window of 5, and a hrf_delay of 3, if the
    # fMRI sample of interest is 20, it will be modeled with stimulus samples
    # [13, 14, 15, 16, 17]. Note that this only applies to visual and audio
    # features, since the language features were already extracted using
    # transcript words spanning several movie samples (thus, each fMRI sample
    # will only be modeled using the corresponding language feature sample,
    # minus the hrf_delay). Also note that a larger stimulus window will
    # increase compute time, since it increases the amount of stimulus features
    # used to train and validate the fMRI encoding models. Here you will use a
    # value of 5, since this is how the challenge baseline encoding models were
    # trained.
    stimulus_window = 5

    ### Loop over subjects ###
    subjects = []
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_friends_s7[f'sub-0{sub}'] = {}

        ### Load the Friends season 7 fMRI samples ###
        samples_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
            'fmri', f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        ### Loop over Friends season 7 episodes ###
        for epi, samples in fmri_samples.items():
            features_epi = []

            ### Loop over fMRI samples ###
            for s in range(samples):
                # Empty variable containing the stimulus features of all
                # modalities for each sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features_friends_s7.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > len(features_friends_s7[mod][epi]):
                            idx_end = len(features_friends_s7[mod][epi])
                            idx_start = idx_end - stimulus_window
                        f = features_friends_s7[mod][epi][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = 0
                        else:
                            idx = s - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features_friends_s7[mod][epi]) - hrf_delay):
                            f = features_friends_s7[mod][epi][-1,:]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        f_all = np.append(f_all, f.flatten())

                ### Append the stimulus features of all modalities for this sample ###
                features_epi.append(f_all)

            ### Add the episode stimulus features to the features dictionary ###
            aligned_features_friends_s7[f'sub-0{sub}'][epi] = np.asarray(
                features_epi, dtype=np.float32)

    return aligned_features_friends_s7