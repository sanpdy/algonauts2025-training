from moviepy.editor import VideoFileClip
import glob
import os
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser("")

tr = 1.49
season = "s1"
save_dir = "/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/movies_chunks/"
video_dir = f"/home/sankalp/algonauts2025/data/algonauts_2025.competitors/stimuli/movies/friends/{season}/"
print("Chunking from: ", video_dir)

processed_videos_dir = os.path.join(save_dir, "friends", season)
if os.path.exists(processed_videos_dir):
    existing_video_names = os.listdir(processed_videos_dir)
else:
    existing_video_names = []

all_videos = glob.glob(f"{video_dir}/*.mkv")
for video in tqdm(all_videos, total=len(all_videos)):
    video_name = video.split("/")[-1].split(".")[0]

    if video_name in existing_video_names:
        tqdm.write(f"Skipping {video_name} (already processed)")
        continue

    prefix = "/".join(video.split("/")[-3:-1])
    video = VideoFileClip(video)
    # video.subclipped()

    starts = np.arange(0, video.duration, tr)
    tqdm_run = tqdm(starts, total=len(starts))
    for chunk_idx, start in enumerate(tqdm_run):
        file_name = str(chunk_idx).zfill(6)
        save_segment_dir = f"{save_dir}/{prefix}/{video_name}/"
        os.makedirs(save_segment_dir, exist_ok=True)

        end = start + tr
        if end > video.duration:
            continue
        segment = video.subclip(start, start + tr)
        # movie_segment = VideoFileClip(movie_path).subclip(start_time, end_time)
        # print(f"\nWriting movie file from {start_time}s until {end_time}s")

        tqdm_run.set_description(f"duration: {segment.duration}")
        # Write video file
        segment.write_videofile(
            f"{save_segment_dir}/{file_name}.mp4",
            codec="libx264",
            audio_codec="aac",
            # verbose=Falsed,
            logger=None,
        )