from django.core.management.base import BaseCommand, CommandError
from plotter.models import FrameContainer, Frame, Video
from bb_binary import Repository, load_frame_container
import glob
import os
from plotter.models import Video

class Command(BaseCommand):
    help = 'Reads a video_path and saves all paths and their names'

    def add_arguments(self, parser):
        parser.add_argument('video_path', type=str)

    def handle(self, *args, **options):
        video_path = options['video_path']

        if video_path[0] != '/':
            print('Path has to be absolute.')
            return

        path = os.path.join(video_path, '**/*.mkv')
        paths = glob.glob(path, recursive=True)

        for path in paths:
            name = os.path.split(path)[1]
            v = Video(video_name=name, video_path=path)
            v.save()
