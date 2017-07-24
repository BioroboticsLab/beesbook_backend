from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from plotter.models import FrameContainer, Frame, Video
from bb_binary import Repository, load_frame_container

from plotter.utils import try_tqdm


class Command(BaseCommand):
    help = 'Reads a repo and saves all relevant information'

    def add_arguments(self, parser):
        parser.add_argument('repo_path', type=str)

    def handle(self, *args, **options):
        repo = Repository(options['repo_path'])
        fnames = list(repo.iter_fnames())
        for fn in try_tqdm(fnames):
            fc = load_frame_container(fn)
            fco = FrameContainer(fc_id=fc.id, fc_path=fn, video_name=fc.dataSources[0].filename)
            fco.save()

            with transaction.atomic():
                for frame in fc.frames:
                    f = Frame(fc=fco, frame_id=frame.id, index=frame.frameIdx, timestamp=frame.timestamp)
                    f.save()



# start with python manage.py make_db_repo [repo_path]