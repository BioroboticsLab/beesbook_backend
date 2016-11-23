video_filenames = '/mnt/storage/beesbook/2015/video_filenames_2015.txt'
ffmpeg_cmd = 'ffmpeg -i {input_path} -vf "select=gte(n\,{frame_id})" -vframes 1 {output_path}'
out_dir = '/tmp/{}'