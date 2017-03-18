# beesbook_backend

The purpose of this backend is to interface the data on the servers and the user and developers. It currently holds the ability to plot videos and images from [bb_binary](https://github.com/BioroboticsLab/bb_binary) and [bb_tracking](https://github.com/BioroboticsLab/bb_tracking) data..
The plots can be cropped to show only relevant areas and the filling of gaps (for missing frames in tracks).

## How To (for users)
1. tunnel

`ssh -N -L 8000:localhost:8000 thekla.imp.fu-berlin.de`

2. post-request similar to [test-notebook](https://github.com/BioroboticsLab/beesbook_backend/blob/master/plotter/tests/test.ipynb)


## Setup (not for users)

1. **Install software and library requirements:**
- docker
- docker-compose (python package)
- python virtualenv with packages in `requirements.txt`

2. **Initialize database data volume:**

`docker create -v /var/lib/postgresql --name postgres_data postgres:9.6`

3. **Find and store all video locations**

`python manage.py make_db_video [video_path_location]`

e.g. `python manage.py make_db_video /home/beesbook/hlrn/work1-hlrn/videos_HD_2016/`

4. **Read all repository data**

`python manage.py make_db_repo [repo_folder]`

e.g. `python manage.py make_db_repo /mnt/storage/beesbook/repo_season_2016_fixed`

5. **Run the server**

Please configure nginx or something similar

For testing `python manage.py runserver` is fine.
