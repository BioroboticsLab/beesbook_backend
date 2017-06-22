import json
from wsgiref.util import FileWrapper

from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from plotter.models import Frame, FrameContainer


@csrf_exempt
def get_frame(request):
    if request.method != 'POST':
        raise HttpResponseBadRequest('Only POST requests allowed.')

    frame_id = request.POST.get('frame_id')
    if frame_id is None:
        raise HttpResponseBadRequest('Parameter frame_id required')
    frame = Frame.objects.get(frame_id=frame_id)
    buffer = frame.get_image(extract='single')

    return HttpResponse(FileWrapper(buffer), content_type='image/jpg')


@csrf_exempt
def get_video(request):
    if request.method != 'POST':
        raise HttpResponseBadRequest('Only POST requests allowed.')

    frame_ids_param = request.POST.get('frame_ids', None)
    frame_container_id = request.POST.get('frame_container_id', None)

    if not (frame_ids_param or frame_container_id):
        raise HttpResponseBadRequest('Either parameter frame_ids_param or frame_container_id required')

    if frame_ids_param:
        if not isinstance(frame_ids_param, str):
            raise HttpResponseBadRequest('Parameter frame_ids_param has to be json-array string containing frame_ids.')
        frame_ids = json.loads(frame_ids_param)
    else:
        if not isinstance(frame_container_id, int):
            raise HttpResponseBadRequest('Parameter frame_container_id has to be an int')
        fc = FrameContainer.objects.get(fc_id=frame_container_id)
        frame_ids = fc.frame_set.all().values_list('frame_id', flat=True)

    buffer = Frame.get_video(frame_ids)
    return HttpResponse(FileWrapper(buffer), content_type='video/mp4')


@csrf_exempt
def plot_frame(request):
    if request.method != 'POST':
        raise HttpResponseBadRequest('Only POST requests allowed.')

    data_json = request.POST.get('data', None)
    if not data_json:
        raise HttpResponseBadRequest('`data` parameter required')

    data = json.loads(data_json)
    buffer = Frame.plot_frame(**data)
    return HttpResponse(FileWrapper(buffer), content_type='image/jpg')


@csrf_exempt
def plot_video(request):
    if request.method != 'POST':
        raise HttpResponseBadRequest('Only POST requests allowed.')

    data_json = request.POST.get('data', None)
    fill_gap = request.POST.get('fillgap', 'False') == 'True'
    crop = request.POST.get('crop', 'False') == 'True'

    if not data_json:
        raise HttpResponseBadRequest('`data` parameter required')

    data = json.loads(data_json)
    buffer = Frame.plot_video(data, fill_gap, crop)
    return HttpResponse(FileWrapper(buffer), content_type='video/mp4')
