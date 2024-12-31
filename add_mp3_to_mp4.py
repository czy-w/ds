from moviepy.editor import VideoFileClip, AudioFileClip

# 读取视频文件，替换为你实际的视频路径
video_path = "box_office_video_summary.mp4"
video_clip = VideoFileClip(video_path)

# 读取音频文件，替换为你实际的音频（MP3）路径
audio_path = "2.mp3"
audio_clip = AudioFileClip(audio_path)

# 将音频的时长调整为和视频时长一致，如果音频更长会截断，如果更短会循环
if audio_clip.duration < video_clip.duration:
    num_loops = int(video_clip.duration // audio_clip.duration) + 1
    audio_clip = audio_clip.loop(duration=video_clip.duration, n=num_loops)
else:
    audio_clip = audio_clip.set_duration(video_clip.duration)

# 将音频添加到视频中
video_with_audio = video_clip.set_audio(audio_clip)

# 输出新的视频文件，可指定格式及路径等，这里以输出MP4为例
output_path = "output_with_bgm.mp4"
video_with_audio.write_videofile(output_path)