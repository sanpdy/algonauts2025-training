# preprocessing/config.py

class FeatureExtractionConfig:
    class Audio:
        model_name     = "microsoft/wavlm-base-plus"
        sampling_rate  = 16000
        batch_size     = 128
        input_dir  = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/transcripts/friends"
        save_dir   = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/wavlm"

    class Language:
        model_name                     = "sentence-transformers/all-mpnet-base-v2"
        input_dir                      = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/transcripts/friends"
        save_dir                       = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/mpnet"
        batch_size                     = 16
        encoding_batch_size            = 32
        use_context                    = True
        context_window                 = 5

    audio      = Audio()
    language   = Language()


class Config:
    feature_extraction = FeatureExtractionConfig()

config = Config()