from __future__ import print_function
import h5py
import numpy as np

def load_features(root_data_dir, modality):
    """
    Load the extracted features from the HDF5 file.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        The modality of the features ('visual', 'audio', or 'language').

    Returns
    -------
    features : np.ndarray
        Stimulus features.

    """

    ### Get the stimulus features file directory ###
    data_dir = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features/mistral_w16.h5"

    # Even though modality is passed in, it’s hard-coded here.
    # If you want to use the parameter, replace the line below with: modality = modality
    modality = "language"

    ### Load the stimulus features ###
    with h5py.File(data_dir, 'r') as data:
        features_list = []

        for episode in data.keys():
            print(f"Exploring structure for episode: {episode}")
            obj = data[episode]

            if isinstance(obj, h5py.Group):
                if modality in obj:
                    features = np.asarray(obj[modality])
                    features_list.append(features)
                else:
                    print(f"Modality '{modality}' not found in group '{episode}'. Skipping.")
            elif isinstance(obj, h5py.Dataset):
                # Directly use dataset if it's not organized by modality
                features = np.asarray(obj)
                features_list.append(features)
            else:
                print(f"Unexpected HDF5 object type: {type(obj)}. Skipping.")

        if not features_list:
            raise ValueError("No features found for the specified modality.")

        features = np.concatenate(features_list, axis=0)

    print(f"{modality} features final shape: {features.shape}")
    print('(Movie samples × Features)')
    print("Feature vector for 5 chunks: \n")
    print(features[:5])

    ### Output ###
    return features

# Run the function
if __name__ == "__main__":
    load_features(
        "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/features",
        "audio"
    )
