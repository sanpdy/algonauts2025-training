from __future__ import print_function
import os
import glob
import numpy as np
import pandas as pd
import h5py
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


RESULTS_DIR = '/home/sankalp/algonauts2025/results/audio_no_pca_stim6'
os.makedirs(RESULTS_DIR, exist_ok=True)

def preprocess_features_chunked(features, chunk_size=10000):
    logger.info("Preprocessing Features with chunked processing")
    features = np.nan_to_num(features)
    scaler = StandardScaler()
    
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i+chunk_size]
        if i == 0:
            scaler.fit(chunk)
        else:
            scaler.partial_fit(chunk)
    
    prepr_features = np.empty_like(features)
    for i in range(0, len(features), chunk_size):
        chunk = features[i:i+chunk_size]
        prepr_features[i:i+chunk_size] = scaler.transform(chunk)
    
    logger.info(f"Preprocessed features shape: {prepr_features.shape}")
    return prepr_features, scaler

def perform_pca_chunked(prepr_features, n_components, modality, chunk_size=10000):
    logger.info(f"Performing PCA on {modality} Features with chunked processing")
    n_raw_dims = prepr_features.shape[1]
    if n_components > n_raw_dims:
        n_components = n_raw_dims

    pca = PCA(n_components=n_components, random_state=20200220)
    pca.fit(prepr_features)
    
    features_pca = np.empty((prepr_features.shape[0], n_components))
    for i in range(0, len(prepr_features), chunk_size):
        chunk = prepr_features[i:i+chunk_size]
        features_pca[i:i+chunk_size] = pca.transform(chunk)
    
    logger.info(f"{modality} features PCA shape: {features_pca.shape}")
    return features_pca, pca



def load_stimulus_features(root_data_dir, modality):
    features = {}
    if modality in ('visual', 'all'):
        features['visual'] = {}
        visual_root = os.path.join(root_data_dir, 'stimuli', 'train_data', 'frozen')
        for season in range(1, 7):
            split_pattern = os.path.join(visual_root, f'friends_s0{season}e*')
            split_dirs = sorted(glob.glob(split_pattern))
            for split_dir in split_dirs:
                split_name = os.path.basename(split_dir)
                embeds_path = os.path.join(split_dir, 'vid_embeds_test.npy')
                ids_path = os.path.join(split_dir, 'ids_test.csv')

                if not os.path.exists(embeds_path) or not os.path.exists(ids_path):
                    print(f"Warning: Missing visual data for {split_name}. Skipping.")
                    continue

                vid_embeds = np.load(embeds_path)
                ids_df = pd.read_csv(ids_path, header=None, skiprows=1)
                if len(vid_embeds) != len(ids_df):
                    raise ValueError(
                        f"Mismatch visual: {split_name} ({len(vid_embeds)} vs {len(ids_df)})"
                    )
                features['visual'][split_name] = vid_embeds.astype(np.float32)

    if modality in ('audio', 'all'):
        features['audio'] = {}
        whisper_h5 = os.path.join(
            root_data_dir,
            'stimuli',
            'train_data',
            'features',
            'whisper_w16.h5'
        )
        if not os.path.exists(whisper_h5):
            raise FileNotFoundError(f"Could not find audio HDF5 at {whisper_h5}")

        with h5py.File(whisper_h5, 'r') as audio_hf:
            for split_name in audio_hf.keys():
                features['audio'][split_name] = audio_hf[split_name][:].astype(np.float32)

    if modality in ('language', 'all'):
        features['language'] = {}
        mistral_h5 = os.path.join(
            root_data_dir,
            'stimuli',
            'train_data',
            'features',
            'mistral_w16.h5'
        )
        if not os.path.exists(mistral_h5):
            raise FileNotFoundError(f"Could not find language HDF5 at {mistral_h5}")

        with h5py.File(mistral_h5, 'r') as lang_hf:
            for split_name in lang_hf.keys():
                features['language'][split_name] = lang_hf[split_name][:].astype(np.float32)

    return features


def load_fmri(root_data_dir, subject):
    fmri = {}

    func_dir = os.path.join(root_data_dir, 'fmri', f'sub-0{subject}', 'func')
    pattern = os.path.join(func_dir, f'sub-0{subject}_task-*_*.h5')
    h5_files = sorted(glob.glob(pattern))

    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 files found for subject {subject} in {func_dir}")

    for h5_path in h5_files:
        task_name = os.path.basename(h5_path).split('_')[2]  # 'friends' or 'movie10'
        with h5py.File(h5_path, 'r') as hf:
            for raw_key, val in hf.items():
                fmri_key = raw_key[13:]
                fmri[fmri_key] = val[:].astype(np.float32)

    return fmri


def preprocess_features(features):
    print("\n=== Preprocessing Features ===")
    features = np.nan_to_num(features)
    scaler = StandardScaler()
    prepr_features = scaler.fit_transform(features)
    print(f"Preprocessed features shape: {prepr_features.shape} (samples × features)")
    return prepr_features


def perform_pca(prepr_features, n_components, modality):
    print(f"\n=== Performing PCA on {modality} Features ===")
    n_raw_dims = prepr_features.shape[1]
    if n_components > n_raw_dims:
        n_components = n_raw_dims

    pca = PCA(n_components=n_components, random_state=20200220)
    features_pca = pca.fit_transform(prepr_features)
    print(f"\n{modality} features PCA shape: {features_pca.shape}")
    print("(samples × principal components)\n")
    return features_pca


def align_features_and_fmri_samples(
    features,
    fmri,
    excluded_samples_start,
    excluded_samples_end,
    hrf_delay,
    stimulus_window,
    movies,
    modality
):
    print("\n=== Aligning Features and fMRI Samples ===")
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)

    for movie in movies:
        skip_movie = False

        if modality in ('visual', 'all'):
            if movie not in features.get('visual', {}):
                print(f"Skipping {movie} (no visual features found)")
                skip_movie = True
        if modality in ('audio', 'all'):
            if movie not in features.get('audio', {}):
                print(f"Skipping {movie} (no audio features found)")
                skip_movie = True
        if modality in ('language', 'all'):
            if movie not in features.get('language', {}):
                print(f"Skipping {movie} (no language features found)")
                skip_movie = True

        if skip_movie:
            continue

        if modality in ('visual', 'all'):
            vid_embeds = features['visual'][movie]
        if modality in ('audio', 'all'):
            audio_embeds = features['audio'][movie]
        if modality in ('language', 'all'):
            lang_embeds = features['language'][movie]

        fmri_key = movie.replace('friends_', '') if movie.startswith('friends_') else movie
        fmri_split = fmri.get(fmri_key)
        if fmri_split is None:
            print(f"Skipping {movie} (no fMRI found)")
            continue

        fmri_split = fmri_split[excluded_samples_start : -excluded_samples_end]
        aligned_fmri = np.append(aligned_fmri, fmri_split, axis=0)

        for s in range(len(fmri_split)):
            f_all = np.empty(0, dtype=np.float32)

            # VISUAL
            if modality in ('visual', 'all'):
                if s < (stimulus_window + hrf_delay):
                    idx_start_v = 0
                    idx_end_v = stimulus_window
                else:
                    idx_start_v = s - hrf_delay - stimulus_window + 1
                    idx_end_v = idx_start_v + stimulus_window
                if idx_end_v > len(vid_embeds):
                    idx_end_v = len(vid_embeds)
                    idx_start_v = idx_end_v - stimulus_window
                vis_window = vid_embeds[idx_start_v : idx_end_v]
                f_all = np.append(f_all, vis_window.flatten())

            # AUDIO
            if modality in ('audio', 'all'):
                if s < (stimulus_window + hrf_delay):
                    idx_start_a = 0
                    idx_end_a = stimulus_window
                else:
                    idx_start_a = s - hrf_delay - stimulus_window + 1
                    idx_end_a = idx_start_a + stimulus_window
                if idx_end_a > len(audio_embeds):
                    idx_end_a = len(audio_embeds)
                    idx_start_a = idx_end_a - stimulus_window
                aud_window = audio_embeds[idx_start_a : idx_end_a]
                f_all = np.append(f_all, aud_window.flatten())

            # LANGUAGE
            if modality in ('language', 'all'):
                if s < (stimulus_window + hrf_delay):
                    idx_start_t = 0
                    idx_end_t = stimulus_window
                else:
                    idx_start_t = s - hrf_delay - stimulus_window + 1
                    idx_end_t = idx_start_t + stimulus_window
                if idx_end_t > len(lang_embeds):
                    idx_end_t = len(lang_embeds)
                    idx_start_t = idx_end_t - stimulus_window
                txt_window = lang_embeds[idx_start_t : idx_end_t]
                f_all = np.append(f_all, txt_window.flatten())

            aligned_features.append(f_all)

    aligned_features = np.asarray(aligned_features, dtype=np.float32)
    return aligned_features, aligned_fmri


def split_movies_train_val(all_movies, val_ratio=0.2, random_seed=42):
    np.random.seed(random_seed)
    all_movies = sorted(all_movies)
    movies_s07 = [m for m in all_movies if m.startswith('friends_s07')]
    n_val = max(1, int(len(all_movies) * val_ratio))
    val_indices = np.random.choice(len(all_movies), size=n_val, replace=False)
    movies_train = [m for m in all_movies if m.startswith('friends_s01')
                                     or m.startswith('friends_s02')
                                     or m.startswith('friends_s03')
                                     or m.startswith('friends_s04')
                                     or m.startswith('friends_s05')]
    movies_val = [m for m in all_movies if m.startswith('friends_s06')]
    return movies_train, movies_val, movies_s07


def normalize_data(X_train, y_train):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1
    X_train_norm = (X_train - X_mean) / X_std
    y_mean = np.mean(y_train, axis=0)
    y_train_norm = y_train - y_mean
    return X_train_norm, y_train_norm, X_mean, X_std, y_mean


def train_encoding(features_train, fmri_train, alphas=np.logspace(2, 5, 10)):
    logger.info("Training Encoding Model")
    logger.info(f"Training features shape: {features_train.shape}")
    logger.info(f"Training fMRI shape: {fmri_train.shape}")

    features_train_norm, fmri_train_norm, X_mean, X_std, y_mean = normalize_data(
        features_train, fmri_train
    )

    model = RidgeCV(alphas=alphas, store_cv_values=False)
    model.fit(features_train_norm, fmri_train_norm)

    logger.info("Training complete.")
    logger.info(f"Optimal alpha chosen by cross-validation: {model.alpha_:.2f}")

    model.normalization_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean
    }
    return model



def predict_fmri(model, features_val):
    X_mean = model.normalization_params['X_mean']
    X_std = model.normalization_params['X_std']
    y_mean = model.normalization_params['y_mean']

    features_val_norm = (features_val - X_mean) / X_std
    fmri_pred_norm = model.predict(features_val_norm)
    fmri_pred = fmri_pred_norm + y_mean

    return fmri_pred

def align_features(
    features,
    hrf_delay,
    stimulus_window,
    movies,
    modality,
    target_length=None
    ):
    aligned_features = []
    
    for movie in movies:
        skip_movie = False
        
        if modality in ('visual', 'all'):
            if movie not in features.get('visual', {}):
                print(f"Skipping {movie} (no visual features found)")
                skip_movie = True
        if modality in ('audio', 'all'):
            if movie not in features.get('audio', {}):
                print(f"Skipping {movie} (no audio features found)")
                skip_movie = True
        if modality in ('language', 'all'):
            if movie not in features.get('language', {}):
                print(f"Skipping {movie} (no language features found)")
                skip_movie = True
                
        if skip_movie:
            continue
            
        if modality in ('visual', 'all'):
            vid_embeds = features['visual'][movie]
        if modality in ('audio', 'all'):
            audio_embeds = features['audio'][movie]
        if modality in ('language', 'all'):
            lang_embeds = features['language'][movie]
            
        if target_length is None:
            seq_lengths = []
            if modality in ('visual', 'all'):
                seq_lengths.append(len(vid_embeds))
            if modality in ('audio', 'all'):
                seq_lengths.append(len(audio_embeds))
            if modality in ('language', 'all'):
                seq_lengths.append(len(lang_embeds))
            seq_length = max(seq_lengths) if seq_lengths else 0
        else:
            seq_length = target_length
            
        if seq_length == 0:
            print(f"Skipping {movie} (zero sequence length)")
            continue
            
        print(f"Processing {movie} with sequence length {seq_length}")
        
        for s in range(seq_length):
            f_all = np.empty(0, dtype=np.float32)
            
            # VISUAL sliding window
            if modality in ('visual', 'all'):
                if s < (stimulus_window + hrf_delay):
                    idx_start_v = 0
                    idx_end_v = stimulus_window
                else:
                    idx_start_v = s - hrf_delay - stimulus_window + 1
                    idx_end_v = idx_start_v + stimulus_window
                if idx_end_v > len(vid_embeds):
                    idx_end_v = len(vid_embeds)
                    idx_start_v = idx_end_v - stimulus_window
                vis_window = vid_embeds[idx_start_v : idx_end_v]
                f_all = np.append(f_all, vis_window.flatten())
                
            # AUDIO sliding window
            if modality in ('audio', 'all'):
                if s < (stimulus_window + hrf_delay):
                    idx_start_a = 0
                    idx_end_a = stimulus_window
                else:
                    idx_start_a = s - hrf_delay - stimulus_window + 1
                    idx_end_a = idx_start_a + stimulus_window
                if idx_end_a > len(audio_embeds):
                    idx_end_a = len(audio_embeds)
                    idx_start_a = idx_end_a - stimulus_window
                aud_window = audio_embeds[idx_start_a : idx_end_a]
                f_all = np.append(f_all, aud_window.flatten())
                
            # LANGUAGE sliding window
            if modality in ('language', 'all'):
                if s < (stimulus_window + hrf_delay):
                    idx_start_t = 0
                    idx_end_t = stimulus_window
                else:
                    idx_start_t = s - hrf_delay - stimulus_window + 1
                    idx_end_t = idx_start_t + stimulus_window
                if idx_end_t > len(lang_embeds):
                    idx_end_t = len(lang_embeds)
                    idx_start_t = idx_end_t - stimulus_window
                txt_window = lang_embeds[idx_start_t : idx_end_t]
                f_all = np.append(f_all, txt_window.flatten())
                
            aligned_features.append(f_all)
    
    if len(aligned_features) == 0:
        return np.empty((0, 0), dtype=np.float32)
    
    aligned_features = np.asarray(aligned_features, dtype=np.float32)
    return aligned_features

def compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, root_data_dir):
    n_parcels = fmri_val.shape[1]
    encoding_accuracy = np.zeros((n_parcels,), dtype=np.float32)

    for p in range(n_parcels):
        corr, _ = pearsonr(fmri_val[:, p], fmri_val_pred[:, p])
        encoding_accuracy[p] = corr if not np.isnan(corr) else 0.0

    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)
    print(f"\n=== Encoding Accuracy Results ({modality}) - Subject {subject} ===")
    print(f"Mean correlation: {mean_encoding_accuracy:.3f}")
    print(f"Median correlation: {np.median(encoding_accuracy):.3f}")
    print(f"Std correlation: {np.std(encoding_accuracy):.3f}")
    print(f"Best parcel correlation: {np.max(encoding_accuracy):.3f}")
    print(f"Worst parcel correlation: {np.min(encoding_accuracy):.3f}")
    print(f"Parcels with r > 0.1: {np.sum(encoding_accuracy > 0.1)}/{n_parcels}")
    print(f"Parcels with r > 0.2: {np.sum(encoding_accuracy > 0.2)}/{n_parcels}")

    # Brain visualization
    try:
        atlas_file = (
            f"sub-0{subject}_space-MNI152NLin2009cAsym_"
            "atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
        )
        atlas_path = os.path.join(root_data_dir, 'fmri', f'sub-0{subject}', 'atlas', atlas_file)
        if os.path.exists(atlas_path):
            atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
            atlas_masker.fit()
            encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

            title = f"Encoding accuracy, sub-0{subject}, modality={modality}, mean={mean_encoding_accuracy}"
            display = plotting.plot_glass_brain(
                encoding_accuracy_nii,
                display_mode="lyrz",
                cmap='hot_r',
                colorbar=True,
                symmetric_cbar=False,
                plot_abs=False,
                title=title
            )
            colorbar = display._cbar
            colorbar.set_label("Pearson's r", rotation=90, labelpad=12, fontsize=12)
            out_fname = f"encoding_accuracy_sub-{subject:02d}_{modality}.png"
            out_path = os.path.join(RESULTS_DIR, out_fname)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')

            plotting.show()
        else:
            print(f"Warning: Atlas file not found at {atlas_path}. Skipping visualization.")
    except Exception as e:
        print(f"Warning: Could not create brain visualization: {e}")
        print("Continuing without brain plot.")

    return encoding_accuracy, mean_encoding_accuracy


if __name__ == '__main__':
    root_data_dir = '/home/sankalp/algonauts2025/data/algonauts_2025.competitors'
    subjects = [1]
    excluded_samples_start = 5
    excluded_samples_end = 5
    hrf_delay = 3
    stimulus_window = 6
    val_ratio = 0.2
    random_seed = 42

    # 'visual', 'audio', 'language', or 'all'
    modality = 'audio'

    # PCA params
    n_components = 250
    

    logger.info("Loading stimulus features...")
    features = load_stimulus_features(root_data_dir, modality=modality)

    print("\n=== Loaded Features ===")
    for mod in features:
        n_splits = len(features[mod])
        example_split = next(iter(features[mod].keys()))
        example_shape = features[mod][example_split].shape
        print(f"  • Modality '{mod}': {n_splits} splits, example '{example_split}' shape = {example_shape}")

    if modality == 'visual':
        all_movies = list(features['visual'].keys())
    elif modality == 'audio':
        all_movies = list(features['audio'].keys())
    elif modality == 'language':
        all_movies = list(features['language'].keys())
    elif modality == 'all':
        all_movies = list(features['visual'].keys())
    else:
        raise ValueError(f"Unknown modality: {modality}")

    movies_train_full = [
        m for m in all_movies
        if any(m.startswith(f'friends_s0{sea}') for sea in range(1, 6))
    ]
    movies_train, movies_internal_val = train_test_split(
        movies_train_full, 
        test_size=0.2, 
        random_state=random_seed
    )
    movies_val = [m for m in all_movies if m.startswith('friends_s06')]
    movies_s07 = [m for m in all_movies if m.startswith('friends_s07')]

    logger.info(f"Training splits: {len(movies_train)}")
    logger.info(f"Internal validation splits: {len(movies_internal_val)}")
    logger.info(f"External validation splits (Season 6): {len(movies_val)}")
    logger.info(f"Test splits (Season 7): {len(movies_s07)}")


    logger.info("Processing Training Data")
    all_aligned_features_train = []
    all_aligned_fmri_train = np.empty((0, 1000), dtype=np.float32)

    for subject in subjects:
        logger.info(f"\nProcessing subject {subject} for training")
        fmri = load_fmri(root_data_dir, subject)

        aligned_features, aligned_fmri = align_features_and_fmri_samples(
            features,
            fmri,
            excluded_samples_start,
            excluded_samples_end,
            hrf_delay,
            stimulus_window,
            movies_train,
            modality
        )
        logger.info(f"Subject {subject} → features: {aligned_features.shape}, fMRI: {aligned_fmri.shape}")

        all_aligned_features_train.append(aligned_features)
        all_aligned_fmri_train = np.append(all_aligned_fmri_train, aligned_fmri, axis=0)

    all_aligned_features_train = np.concatenate(all_aligned_features_train, axis=0)
    logger.info(f"Aggregated Training Data")
    logger.info(f"Raw training features shape: {all_aligned_features_train.shape}")
    logger.info(f"Training fMRI shape: {all_aligned_fmri_train.shape}")


    all_aligned_features_train_norm = preprocess_features(all_aligned_features_train)

    all_aligned_features_train_norm, trained_scaler = preprocess_features_chunked(all_aligned_features_train)
    train_features_pca, trained_pca = perform_pca_chunked(
        all_aligned_features_train_norm, n_components, modality
    )
    
    logger.info("Starting training encoding model")
    model = train_encoding(all_aligned_features_train_norm, all_aligned_fmri_train)
    print("\n=== Processing Validation Data … ===")
    for subject in subjects:
        print(f"\nValidating on subject {subject} …")
        fmri = load_fmri(root_data_dir, subject)

        features_val_raw, fmri_val = align_features_and_fmri_samples(
            features,
            fmri,
            excluded_samples_start,
            excluded_samples_end,
            hrf_delay,
            stimulus_window,
            movies_val,
            modality
        )
        print(f"  Raw validation features shape: {features_val_raw.shape}, fMRI shape: {fmri_val.shape}")

        features_val_norm = trained_scaler.transform(features_val_raw)
        features_val_pca = trained_pca.transform(features_val_norm)
        fmri_val_pred = predict_fmri(model, features_val_norm)
        compute_encoding_accuracy(fmri_val, fmri_val_pred, subject, modality, root_data_dir)

    model_fname = f"encoding_{modality}_pca{n_components}.pkl"
    model_save_path = os.path.join(RESULTS_DIR, model_fname)
    with open(model_save_path, 'wb') as f:
        pickle.dump({ 'model': model, 'scaler': trained_scaler }, f)
    print(f"\nTrained model and scaler saved to: {model_save_path}")

    print("\nTraining and validation complete!")

    print("\n=== Predicting on Season 7 splits ===")
    print(f"Season 7 splits to predict: {movies_s07}")

    for subject in subjects:
        print(f"\nPredicting for subject {subject} …")
        
        features_s07_raw = align_features(
            features,
            hrf_delay,
            stimulus_window,
            movies_s07,
            modality
        )
        
        if features_s07_raw.shape[0] == 0:
            print(f"  No Season 7 features found for subject {subject}. Skipping.")
            continue
    
        print(f"  Raw Season 7 features shape: {features_s07_raw.shape}")
        
        features_s07_norm = trained_scaler.transform(features_s07_raw)
        features_s07_pca = trained_pca.transform(features_s07_norm)
        
        # Predict fMRI on Season 7
        fmri_s07_pred = predict_fmri(model, features_s07_pca)
        print(f"  Predicted fMRI shape: {fmri_s07_pred.shape}")
        
        pred_fname = f"fmri_pred_subject{subject}_season7.npy"
        out_path = os.path.join(RESULTS_DIR, pred_fname)
        np.save(out_path, fmri_s07_pred)
        print(f"  → Saved predictions to {out_path}")

    print("\nSeason 7 prediction complete!")
