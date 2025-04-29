import os

def create_path(path):
    dirs = path.split(os.sep)

    cur_path = ""
    for dir in dirs:
        cur_path = os.path.join(cur_path, dir)
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)
