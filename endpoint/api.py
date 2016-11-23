from flask import Flask
from flask import abort
from flask import request
from flask import send_file
import os
import config

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
    frame_id = request.args.get('frame_id', '')
    filename = request.args.get('filename', '')
    if not frame_id and not filename:
        abort(400)

    name, ext = os.path.splitext(filename)
    input_path = video_dict[filename]

    out_name = '{}-frame_{}.{}'.format(name, frame_id, 'png')
    output_path = config.out_dir.format(out_name)

    if not os.path.exists(output_path):
        cmd = config.ffmpeg_cmd.format(
            input_path=input_path,
            frame_id=frame_id,
            output_path=output_path
        )
        success_code = os.system(cmd)
        print(success_code)

    return send_file(output_path, mimetype='image/png')


@app.route('/up', methods=['GET'])
def success():
    return ''

if __name__ == '__main__':
    prepare()
    app.run()