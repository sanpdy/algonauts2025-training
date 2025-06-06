from __future__ import print_function
import os
import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from pathlib import Path
from load_embeddings import load_features, preprocess_features_dict, perform_pca, load_stimulus_features_friends_s7, align_features_and_fmri_samples_friends_s7
from load_fmri import load_fmri, align_features_and_fmri_samples
from load_models import load_baseline_encoding_models, train_encoding, compute_encoding_accuracy
import numpy as np
if __name__ == "__main__":
    root_data_dir = Path("/home/sankalp/algonauts2025/data")
    data_path = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features/whisper_w16.h5"
    n_components = 250
    subject = 1
    modality = "audio"  #@param ["visual", "audio", "language", "all"]
    excluded_samples_start = 5
    excluded_samples_end = 5
    hrf_delay = 3
    stimulus_window = 5

    movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"]
    movies_val = ["friends-s06"]

    # Load fMRI data
    fmri = load_fmri(root_data_dir, subject)
    
    # Load and preprocess features
    if modality == "audio":
        audio_features_dict = load_features(
            "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features",
            "audio"
        )
        audio_features_dict = preprocess_features_dict(audio_features_dict, "audio")
        features_dict = audio_features_dict
        
    elif modality == "language":
        language_features_dict = load_features(
            "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features",
            "language"
        )
        language_features_dict = preprocess_features_dict(language_features_dict, "language")
        features_dict = language_features_dict
        
    elif modality == "all":
        audio_features_dict = load_features(
            "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features",
            "audio"
        )
        language_features_dict = load_features(
            "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features",
            "language"
        )

        audio_features_dict = preprocess_features_dict(audio_features_dict, "audio")
        language_features_dict = preprocess_features_dict(language_features_dict, "language")
        
        features_dict = {**audio_features_dict, **language_features_dict}
        
    # Align features with fMRI
    features_train, fmri_train = align_features_and_fmri_samples(features_dict, fmri,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_train)

    print("Training fMRI responses shape:")
    print(fmri_train.shape)
    print('(Train samples × Parcels)')
    print("\nTraining stimulus features shape:")
    print(features_train.shape)
    print('(Train samples × Features)')
    
    # Print all available movies
    print(f"Subject {subject} fMRI movies splits name and shape:")
    for key, value in fmri.items():
        print(key + " " + str(value.shape))
    
    model = train_encoding(features_train, fmri_train)
    del features_train, fmri_train

    model_weights = model_weights = {
    'coef_': model.coef_,
    'intercept_': model.intercept_,
    'n_features_in_': model.n_features_in_
    }

    save_path = "/home/sankalp/algonauts2025/models"
    np.save(save_path, model_weights, allow_pickle=True)

    features_val, fmri_val = align_features_and_fmri_samples(features_dict, fmri,
    excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
    movies_val)

    del features_dict, fmri

    # Print the shape of the test fMRI responses and stimulus features: note
    # that the two have the same sample size!
    print("Validation fMRI responses shape:", fmri_val.shape)
    print('(Validation samples × Parcels)')
    print("\nValidation stimulus features shape:", features_val.shape)
    print('(Validation samples × Features)')

    # Predict the fMRI responses for the validation movies
    fmri_val_pred = model.predict(features_val)

    # Print the shape of the recorded and predicted test fMRI responses: note that
    # the two have the same shape!
    print("Validation fMRI responses shape:", fmri_val.shape)
    print('(Validation samples × Parcels)')
    print("\nValidation predicted fMRI responses shape:", fmri_val_pred.shape)
    print('(Validation samples × Parcels)')
    compute_encoding_accuracy(root_data_dir, fmri_val, fmri_val_pred, subject, modality)

    features_friends_s7 = load_stimulus_features_friends_s7(root_data_dir)
    for key, value in features_friends_s7.items():
        print(f"Episode {key} features shape: {value.shape}")
        for key_movie, value_movie in value.items():
            print(key_movie + " " + str(value_movie.shape))

    aligned_features_friends_s7 = align_features_and_fmri_samples_friends_s7(
    features_friends_s7, root_data_dir)
    for key, value in aligned_features_friends_s7.items():
        episode_name = "s07e01a"
        example_episode_shape = value[episode_name].shape
        print(f"Subject: {key}")
        print(f"  Episode: {episode_name} - Features shape: {example_episode_shape}")
        print("-" * 40)
        
    baseline_models = 
    # Empty submission predictions dictionary
    submission_predictions = {}

    # Loop through each subject
    desc = "Predicting fMRI responses of each subject"
    for sub, features in tqdm(aligned_features_friends_s7.items(), desc=desc):

        # Initialize the nested dictionary for each subject's predictions
        submission_predictions[sub] = {}

        # Loop through each Friends season 7 episode
        for epi, feat_epi in features.items():

            # Predict fMRI responses for the aligned features of this episode, and
            # convert the predictions to float32
            fmri_pred = baseline_models[sub].predict(feat_epi).astype(np.float32)

            # Store formatted predictions in the nested dictionary
            submission_predictions[sub][epi] = fmri_pred