import numpy as np
import h5py
import pdb
import os
import xgboost as xgb

movies_train = [
    "friends-s01",
    "friends-s02",
    "friends-s03",
    "friends-s04",
    "friends-s05",
    "movie10-bourne",
    "movie10-figures",
    "movie10-life",
    "movie10-wolf",
]  # @param {allow-input: true}

movies_val = ["friends-s06"]  # @param {allow-input: true}


def align_features_and_fmri_samples(
    features,
    fmri,
    excluded_samples_start,
    excluded_samples_end,
    hrf_delay,
    stimulus_window,
    movies,
):
    """
    Align the stimulus feature with the fMRI response samples for the selected
    movies, later used to train and validate the encoding models.

    Parameters
    ----------
    features : dict
        Dictionary containing the stimulus features.
    fmri : dict
        Dictionary containing the fMRI responses.
    excluded_trs_start : int
        Integer indicating the first N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that due
        to the latency of the hemodynamic response the fMRI responses of first
        few fMRI TRs do not yet contain stimulus-related information.
    excluded_trs_end : int
        Integer indicating the last N fMRI TRs that will be excluded and not
        used for model training. The reason for excluding these TRs is that
        stimulus feature samples (i.e., the stimulus chunks) can be shorter than
        the fMRI samples (i.e., the fMRI TRs), since in some cases the fMRI run
        ran longer than the actual movie. However, keep in mind that the fMRI
        timeseries onset is ALWAYS SYNCHRONIZED with movie onset (i.e., the
        first fMRI TR is always synchronized with the first stimulus chunk).
    hrf_delay : int
        fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
        that reflects changes in blood oxygenation levels in response to
        activity in the brain. Blood flow increases to a given brain region in
        response to its activity. This vascular response, which follows the
        hemodynamic response function (HRF), takes time. Typically, the HRF
        peaks around 5–6 seconds after a neural event: this delay reflects the
        time needed for blood oxygenation changes to propagate and for the fMRI
        signal to capture them. Therefore, this parameter introduces a delay
        between stimulus chunks and fMRI samples for a better correspondence
        between input stimuli and the brain response. For example, with a
        hrf_delay of 3, if the stimulus chunk of interest is 17, the
        corresponding fMRI sample will be 20.
    stimulus_window : int
        Integer indicating how many stimulus features' chunks are used to model
        each fMRI TR, starting from the chunk corresponding to the TR of
        interest, and going back in time. For example, with a stimulus_window of
        5, if the fMRI TR of interest is 20, it will be modeled with stimulus
        chunks [16, 17, 18, 19, 20]. Note that this only applies to visual and
        audio features, since the language features were already extracted using
        transcript words spanning several movie chunks (thus, each fMRI TR will
        only be modeled using the corresponding language feature chunk). Also
        note that a larger stimulus window will increase compute time, since it
        increases the amount of stimulus features used to train and test the
        fMRI encoding models.
    movies: list
        List of strings indicating the movies for which the fMRI responses and
        stimulus features are aligned, out of the first six seasons of Friends
        ["friends-s01", "friends-s02", "friends-s03", "friends-s04",
        "friends-s05", "friends-s06"], and the four movies from Movie10
        ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"].

    Returns
    -------
    aligned_features : float
        Aligned stimulus features for the selected movies.
    aligned_fmri : float
        Aligned fMRI responses for the selected movies.

    """

    ### Empty data variables ###
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)

    ### Loop across movies ###
    for movie in movies:

        ### Get the IDs of all movies splits for the selected movie ###
        if movie[:7] == "friends":
            id = movie[8:]
        elif movie[:7] == "movie10":
            id = movie[8:]
        movie_splits = [key for key in fmri if id in key[: len(id)]]

        ### Loop over movie splits ###
        for split in movie_splits:

            ### Extract the fMRI ###
            fmri_split = fmri[split]
            # Exclude the first and last fMRI samples
            fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            ### Loop over fMRI samples ###
            for s in range(len(fmri_split)):
                # Empty variable containing the stimulus features of all
                # modalities for each fMRI sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == "visual" or mod == "audio":
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = (
                                s
                                + excluded_samples_start
                                - hrf_delay
                                - stimulus_window
                                + 1
                            )
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > (len(features[mod][split])):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window
                        f = features[mod][split][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == "language":
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1, :]
                        else:
                            f = features[mod][split][idx]
                        f_all = np.append(f_all, f.flatten())

                ### Append the stimulus features of all modalities for this sample ###
                aligned_features.append(f_all)

    ### Convert the aligned features to a numpy array ###
    aligned_features = np.asarray(aligned_features, dtype=np.float32)

    ### Output ###
    return aligned_features, aligned_fmri


def load_fmri(root_data_dir, subject):
    """
    Load the fMRI responses for the selected subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    subject : int
        Subject used to train and validate the encoding model.

    Returns
    -------
    fmri : dict
        Dictionary containing the  fMRI responses.

    """

    fmri = {}

    ### Load the fMRI responses for Friends ###
    # Data directory
    fmri_file = f"sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
    fmri_dir = os.path.join(
        root_data_dir,
        "algonauts_2025.competitors",
        "fmri",
        f"sub-0{subject}",
        "func",
        fmri_file,
    )
    # Load the the fMRI responses
    fmri_friends = h5py.File(fmri_dir, "r")
    for key, val in fmri_friends.items():
        fmri[str(key[13:])] = val[:].astype(np.float32)
    del fmri_friends

    ### Load the fMRI responses for Movie10 ###
    # Data directory
    fmri_file = f"sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
    fmri_dir = os.path.join(
        root_data_dir,
        "algonauts_2025.competitors",
        "fmri",
        f"sub-0{subject}",
        "func",
        fmri_file,
    )
    # Load the the fMRI responses
    fmri_movie10 = h5py.File(fmri_dir, "r")
    for key, val in fmri_movie10.items():
        fmri[key[13:]] = val[:].astype(np.float32)
    del fmri_movie10
    # Average the fMRI responses across the two repeats for 'figures'
    keys_all = fmri.keys()
    figures_splits = 12
    for s in range(figures_splits):
        movie = "figures" + format(s + 1, "02")
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(
            np.float32
        )
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]
    # Average the fMRI responses across the two repeats for 'life'
    keys_all = fmri.keys()
    life_splits = 5
    for s in range(life_splits):
        movie = "life" + format(s + 1, "02")
        keys_movie = [rep for rep in keys_all if movie in rep]
        fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(
            np.float32
        )
        del fmri[keys_movie[0]]
        del fmri[keys_movie[1]]

    ### Output ###
    return fmri


def load_features(feature_path):
    features = {}

    data = h5py.File(feature_path, "r")
    for k, v in data.items():
        features[k.split("_")[-1]] = v.astype(np.float32)

    return features


if __name__ == "__main__":
    # features_train, fmri_train = align_features_and_fmri_samples(
    #     features,
    #     fmri,
    #     excluded_samples_start,
    #     excluded_samples_end,
    #     hrf_delay,
    #     stimulus_window,
    #     movies_train,
    # )
    # features = h5py.File("features/mistral_w8.h5", "r")
    # excluded_samples_start = 5
    # excluded_samples_end = 5
    # stimulus_window = 5
    # hrf_delay = 3
    # features = {}
    # fmri = load_fmri(root_data_dir="data", subject=1)
    # features["language"] = load_features(feature_path="features/mistral_w8.h5")
    # # features["language"] = load_features(feature_path="features/mistral_w8.h5")

    # features_train, fmri_train = align_features_and_fmri_samples(
    #     features,
    #     fmri,
    #     excluded_samples_start,
    #     excluded_samples_end,
    #     hrf_delay,
    #     stimulus_window,
    #     movies_train,
    # )

    # features_val, fmri_val = align_features_and_fmri_samples(
    #     features,
    #     fmri,
    #     excluded_samples_start,
    #     excluded_samples_end,
    #     hrf_delay,
    #     stimulus_window,
    #     movies_val,
    # )

    features_train = np.load("features_train.npy")
    features_val = np.load("features_train.npy")
    fmri_train = np.load("fmri_train.npy")
    fmri_val = np.load("fmri_val.npy")

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "verbosity": 2,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBRegressor(**xgb_params)

    # Split
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.2, random_state=42
    # )

    # Train
    model.fit(features_train[:1000], fmri_train[:1000])

    # Evaluate
    # results = model.evaluate(features_val, fmri_val)
    # print("Mean R²:", results["mean_r2"])
    # print("Mean Corr:", results["mean_corr"])

    # # Save models
    # model.save()
    pdb.set_trace()
