import os
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, RidgeCV, MultiTaskLassoCV, ElasticNetCV
import nibabel as nib
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import pearsonr

def compute_encoding_accuracy(root_data_dir, fmri_val, fmri_val_pred, subject, modality):
    ### Correlate recorded and predicted fMRI responses ###
    encoding_accuracy = np.zeros((fmri_val.shape[1]), dtype=np.float32)
    for p in range(len(encoding_accuracy)):
        encoding_accuracy[p] = pearsonr(fmri_val[:, p],
            fmri_val_pred[:, p])[0]
    mean_encoding_accuracy = np.round(np.mean(encoding_accuracy), 3)

    ### Map the prediction accuracy onto a 3D brain atlas for plotting ###
    atlas_file = f'sub-0{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz'
    atlas_path = os.path.join(root_data_dir, 'algonauts_2025.competitors',
        'fmri', f'sub-0{subject}', 'atlas', atlas_file)
    atlas_masker = NiftiLabelsMasker(labels_img=atlas_path)
    atlas_masker.fit()
    encoding_accuracy_nii = atlas_masker.inverse_transform(encoding_accuracy)

    ### Plot the encoding accuracy ###
    title = f"Encoding accuracy, sub-0{subject}, modality-{modality}, mean accuracy: " + str(mean_encoding_accuracy)
    display = plotting.plot_glass_brain(
        encoding_accuracy_nii,
        display_mode="lyrz",
        cmap='hot_r',
        colorbar=True,
        plot_abs=False,
        symmetric_cbar=False,
        title=title,
    )
    colorbar = display._cbar
    colorbar.set_label("Pearson's $r$", rotation=90, labelpad=12, fontsize=12)
    output_file="/home/sankalp/algonauts2025/results/training_v2_lasso/encoding_accuracy_sub-0" + str(subject) + "_modality-" + modality + ".png"
    display.savefig(output_file, dpi=300)
    display.close()
    print(f"Encoding accuracy plot saved to: {output_file}")

def load_baseline_encoding_models(root_data_dir):
    baseline_models = {}

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    for s in subjects:
        ### Load the trained encoding model weights ###
        weights_dir = os.path.join(root_data_dir, 'trained_encoding_models',
            'trained_encoding_model_sub-0'+str(s)+'_modality-all.npy')
        model_weights = np.load(weights_dir, allow_pickle=True).item()

        ### Initialize the Ridge regression and load the trained weights ###
        model = Ridge()
        model.coef_ = model_weights['coef_']
        model.intercept_ = model_weights['intercept_']
        model.n_features_in_ = model_weights['n_features_in_']

        ### Store the pretrained encoding model into a dictionary ###
        baseline_models['sub-0'+str(s)] = model
        del model

    ### Output ###
    return baseline_models

def train_encoding(features_train, fmri_train):
    #model = LinearRegression().fit(features_train, fmri_train)
    model = RidgeCV(cv=5).fit(features_train, fmri_train)
    return model


