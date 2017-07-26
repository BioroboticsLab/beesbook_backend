width = 4000
height = 3000

n_threads = 4
gpu = True
enable_caching = True

binary_location = '/opt/bin/ffmpeg'

verbosity_level = 24
ffmpeg_video = ' '.join([
    f'{binary_location} -y -v {verbosity_level}',
    '-vcodec hevc_cuvid' if gpu else '',
    '-i {video_path}',  # input
    '-vf "select=gte(n\,{left_frame_idx}),setpts=PTS-STARTPTS" -vframes {number_of_frames}',  # subset selection
    '-crf 28',  # quality
    '-c:v h264',  # encoder
    '-pix_fmt yuv420p',  # for outdated players
    '{output_path}'  # output path
])

ffmpeg_extract_all_frames = ' '.join([
    f'{binary_location} -y -v {verbosity_level}',
    '-vcodec hevc_cuvid' if gpu else '',
    '-r 3',
    '-i {video_path}',
    '-start_number 0',
    '-vf scale=iw*{scale}:ih*{scale}',
    '-qscale:v 2',
    '{output_path}/%04d.jpg'
])

ffmpeg_frames_to_video = ' '.join([
    f'{binary_location} -y -v {verbosity_level}',
    '-r {framerate}',  # input framerate
    '-i {input_path}',
    '-r {framerate}',  # video framerate (kinda irrelevant)
    '-pix_fmt yuv420p',
    '-vcodec h264_nvenc',
    '{output_path}'
])

ffmpeg_extract_single_frame = ' '.join([
    f'{binary_location} -y -v {verbosity_level}',
    '-vcodec hevc_cuvid' if gpu else '',
    '-i {video_path}',
    '-vf "select=gte(n\,{frame_index}),'  # first filter: select specific frame
    'scale=iw*{scale}:ih*{scale}"',  # second filter: scale output down or up
    '-vframes 1',
    '{output_path}'
])
