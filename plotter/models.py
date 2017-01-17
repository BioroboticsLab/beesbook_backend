from django.db import models
from bb_binary import load_frame_container
import os
from . import media


class Video(models.Model):
    video_name = models.TextField(primary_key=True)
    video_path = models.TextField()


class FrameContainer(models.Model):
    id = models.AutoField(primary_key=True)
    fc_id = models.DecimalField(max_digits=32, decimal_places=0, db_index=True)
    fc_path = models.TextField()
    video_name = models.TextField()

    def get_binary(self):
        return load_frame_container(self.fc_path)

    @property
    def video_path(self):
        return Video.objects.get(video_name=self.video_name).video_path


class Frame(models.Model):
    frame_id = models.DecimalField(max_digits=32, decimal_places=0, primary_key=True)
    fc = models.ForeignKey(FrameContainer)
    index = models.IntegerField()

    def get_image_path(self, extract=None):
        """
        Gets the path to the image. Also enabled the option to extract the picture if not done yet.
        Options for extract are 'single' or 'all'.
        'single' extracts just that frame and returns the path.
        'all' extracts all frames of the containing framecontainer and returns the single frame path.
        """
        path = f'/tmp/{self.fc.video_name}/{self.index:04}.png'
        if os.path.exists(path):
            return path

        if extract is None:
            return None

        if extract == 'single':
            return media.extract_single_frame(self)

        if extract == 'all':
            image_folder = media.extract_frames(framecontainer=self.fc)
            image_path = os.path.join(image_folder, f'{self.id}.png')
            return image_path
