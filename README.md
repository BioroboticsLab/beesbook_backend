# beesbook_backend

The purpose of this backend is to interface the data on the servers and the user and developers. It currently holds the ability to plot videos and images from [bb_binary](https://github.com/BioroboticsLab/bb_binary) and [bb_tracking](https://github.com/BioroboticsLab/bb_tracking) data..
The plots can be cropped to show only relevant areas and the filling of gaps (for missing frames in tracks).

## How To (for users)
1. install api part of bb_backend (no dependencies required)
    - pip install git+https://github.com/BioroboticsLab/beesbook_backend.git
2. tunnel

`ssh -N -L 8000:localhost:8000 thekla.imp.fu-berlin.de`

3. use bb_backend.api to fetch images and videos:

```
import bb_backend.api
from bb_backend.api import FramePlotter, VideoPlotter
bb_backend.api.server_adress = '127.0.0.1:8000'

frame_plotter = FramePlotter(frame_id=12058089920919369671, xs=[350, 450], ys=[350, 450],
                                sizes=[40,200],title="Image title (e.g. timestamp)",
                                labels=["Abee","Beebee"], scale=1.0, colors=["r", "y"])
plt.imshow(frame_plotter.get_image())

# Make video consisting of 3 times the same frame.
video_plotter = VideoPlotter(frames=[frame_plotter, frame_plotter, frame_plotter])

video_plotter.get_video(display_in_notebook=True)
```


## Setup (not for users)

1. **Install software and library requirements:**
- docker
- docker-compose (python package)
- python virtualenv with packages in `requirements.txt`
- ffmpeg with x264 encoding and h264 decoding capability

2. **Initialize database data volume:**

`docker create -v /var/lib/postgresql --name postgres_data postgres:9.6`

`docker-compose up -d`

`docker-compose exec postgres psql -U postgres -c "CREATE DATABASE beesbook;"`

`python manage.py migrate`

3. **Find and store all video locations**

`python manage.py make_db_video [video_path_location]`

e.g. `python manage.py make_db_video /home/beesbook/hlrn/work1-hlrn/videos_HD_2016/`

4. **Read all repository data**

`python manage.py make_db_repo [repo_folder]`

e.g. `python manage.py make_db_repo /mnt/storage/beesbook/repo_season_2016_fixed`

5. **Run the server**

Please configure nginx or something similar

For testing `python manage.py runserver` is fine.
