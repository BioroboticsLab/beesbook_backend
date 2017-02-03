from django.contrib import admin

from plotter.models import Frame, Video, FrameContainer
from plotter.models import FrameContainer

# Register your models here.

admin.site.register(Frame)
admin.site.register(FrameContainer)
admin.site.register(Video)

