from django.db import models
from bb_binary import load_frame_container


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

    def get_video_path(self):
        return Video.objects.get(video_name=self.video_name).video_path


class Frame(models.Model):
    frame_id = models.DecimalField(max_digits=32, decimal_places=0, primary_key=True)
    fc = models.ForeignKey(FrameContainer)
    index = models.IntegerField()

