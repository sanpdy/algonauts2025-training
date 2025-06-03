export PYTHONPATH=$PYTHONPATH:/home/sankalp/algonauts2025/VideoLLaMA2
python /home/sankalp/algonauts2025/VideoLLaMA2/videollama2/eval/inference_audio_video.py \
  --model-path "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV" \
  --video-folder "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/movies_chunks/friends/s1/friends_s01e01a/" \
  --output-file "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/train_data/video_llama/friends_s01e01a_summary.json" \
  --dataset AVQA \
  --batch-size 1 \
  --num-workers 4 \
  --save_tensors
