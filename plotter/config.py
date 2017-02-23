scale = 0.5
width = int(4000 * scale)
height = int(3000 * scale)
padding = int(600 * scale)

n_threads = 4
cache = False
gpu = False

verbosity_level = 24
ffmpeg_video = ' '.join([
    f'ffmpeg -v {verbosity_level}',
    '-vcodec hevc_cuvid' if gpu else '',
    '-i {video_path}',  # input
    '-vf "select=gte(n\,{left_frame_idx}),setpts=PTS-STARTPTS" -vframes {number_of_frames}',  # subset selection
    '-crf 28',  # quality
    '-c:v h264',  # encoder
    '-pix_fmt yuv420p',  # for outdated players
    '{output_path}'  # output path
])

ffmpeg_extract_all_frames = ' '.join([
    f'ffmpeg -v {verbosity_level}',
    '-vcodec hevc_cuvid' if gpu else '',
    '-r 3',
    '-i {video_path}',
    '-start_number 0',
    f'-vf scale=iw*{scale}:ih*{scale}',
    '-qscale:v 2',
    '{output_path}/%04d.jpg'
])

ffmpeg_frames_to_video = ' '.join([
    f'ffmpeg -v {verbosity_level}',
    '-r 3',  # input framerate
    '-i {input_path}',
    '-r 3',  # video framerate (kinda irrelevant)
    '-pix_fmt yuv420p',
    '-vcodec libx264',
    '{output_path}'
])

ffmpeg_extract_single_frame = ' '.join([
    f'ffmpeg -v {verbosity_level}',
    '-vcodec hevc_cuvid' if gpu else '',
    '-i {video_path}',
    '-vf "select=gte(n\,{frame_index}),'  # first filter: select specific frame
    f'scale=iw*{scale}:ih*{scale}"',  # second filter: scale output down or up
    '-vframes 1',
    '{output_path}'
])
