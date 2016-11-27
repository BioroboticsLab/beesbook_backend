# #### api config #### #
video_filenames = '/mnt/storage/beesbook/2015/video_filenames_2015.txt'
out_dir = '/tmp/{}'

ffmpeg_frame = 'ffmpeg -v 24 -i {input_path} -vf "select=gte(n\,{frame})" -vframes 1 {output_path}'
ffmpeg_video = ' '.join([
    'ffmpeg -v 24',  # only show warnings or higher
    '-i {input_path}',  # input
    '-vf "select=gte(n\,{left_frame}),setpts=PTS-STARTPTS" -vframes {number_of_frames}',  # subset selection
    '-crf 28',  # quality
    '-c:v h264',  # encoder
    '-pix_fmt yuv420p',  # squash warning
    '{output_path}'  # output path
])

# #### plotting config #### #
BASE = 'http://localhost:5000'
IMAGE_ENDPOINT = BASE + '/get_frame?filename={video_name}&frame={frame}'
VIDEO_ENDPOINT = BASE + '/get_video?filename={video_name}&left_frame={left_frame}&right_frame={right_frame}'
API_IS_UP = BASE + '/up'
FRAMES_DIR = 'frame_media'

width = 800
height = 600

VIDEO_HTML = """
<video style='margin: 0 auto;' width="{width}" height="{height}" controls>
    <source src="{src}" type="video/mp4">
</video>
"""

# #### shared config #### #
img_name = '{}-frame_{}.png'
video_name = '{name}-frame_{left_frame}-{right_frame}.mp4'
