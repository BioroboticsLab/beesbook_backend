from django.core.management.base import BaseCommand, CommandError

import glob
import os
from plotter.models import Video
from plotter.utils import try_tqdm


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

        for path in try_tqdm(paths):
            name = os.path.split(path)[1]
            v = Video(video_name=name, video_path=path)
            v.save()
