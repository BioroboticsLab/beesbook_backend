from flask import Flask
from flask import abort
from flask import request
from flask import send_file
import os
import config
from subprocess import check_output

app = Flask(__name__)
video_dict = {}  # maps video filenames to their path


def prepare():
    with open(config.video_filenames, 'r') as f:
        for video in f.read().split('\n'):
            if '/' not in video:  # safeguard
                continue
            name = video.split('/')[-1]
            video_dict[name] = video


@app.route('/get_frame', methods=['GET'])
def get_frame():
    frame = request.args.get('frame', '')
    filename = request.args.get('filename', '')
    if not frame and not filename:
        abort(400)

    name, ext = os.path.splitext(filename)
    input_path = video_dict[filename]

    out_name = config.img_name.format(name, frame)
    output_path = config.out_dir.format(out_name)

    if not os.path.exists(output_path):
        cmd = config.ffmpeg_frame.format(
            input_path=input_path,
            frame=frame,
            output_path=output_path
        )
        output = check_output(cmd, shell=True)
        print('output:', output)

    return send_file(output_path, mimetype='image/png')


@app.route('/get_video', methods=['GET'])
def get_video():
    left_frame = request.args.get('left_frame', '')
    right_frame = request.args.get('right_frame', '')
    filename = request.args.get('filename', '')
    if not left_frame and not right_frame and not filename:
        abort(400)

    name, ext = os.path.splitext(filename)
    input_path = video_dict[filename]

    out_name = config.video_name.format(
        name=name,
        left_frame=left_frame,
        right_frame=right_frame
    )
    output_path = config.out_dir.format(out_name)

    if not os.path.exists(output_path):
        cmd = config.ffmpeg_video.format(
            input_path=input_path,
            left_frame=left_frame,
            right_frame=right_frame,
            output_path=output_path
        )
        check_output(cmd, shell=True)

    return send_file(output_path, mimetype='video/mp4')


@app.route('/up', methods=['GET'])
def success():
    return ''

if __name__ == '__main__':
    prepare()
    app.run()
