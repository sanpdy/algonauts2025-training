from __future__ import print_function
import os
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import h5py
import torch
import librosa
import ast
import string
import zipfile
from tqdm.notebook import tqdm
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import cv2
import nibabel as nib
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
import ipywidgets as widgets
from ipywidgets import VBox, Dropdown, Button
from IPython.display import Video, display, clear_output
from moviepy.editor import VideoFileClip
from transformers import BertTokenizer, BertModel
from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
from load_embeddings import load_features, preprocess_features_dict
from load_fmri import load_fmri, align_features_and_fmri_samples
from load_models import load_baseline_encoding_models, train_encoding
from load_models import compute_encoding_accuracy

if __name__ == "__main__":
    root_data_dir = Path("/home/sankalp/algonauts2025/data")
    data_path = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features/whisper_w16.h5"
    n_components = 250
    subject = 1
    modality = "language"  #@param ["visual", "audio", "language", "all"]
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
        # Load both modalities
        audio_features_dict = load_features(
            "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features",
            "audio"
        )
        language_features_dict = load_features(
            "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features",
            "language"
        )
        
        # Preprocess both
        audio_features_dict = preprocess_features_dict(audio_features_dict, "audio")
        language_features_dict = preprocess_features_dict(language_features_dict, "language")
        
        # Combine them
        features_dict = {**audio_features_dict, **language_features_dict}

    # Align features with fMRI
    features_train, fmri_train = align_features_and_fmri_samples(features_dict, fmri,
        excluded_samples_start, excluded_samples_end, hrf_delay, stimulus_window,
        movies_train)

    # Print the shape of the training fMRI responses and stimulus features
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