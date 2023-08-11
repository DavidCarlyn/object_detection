import os
import io
import signal
import sys
import subprocess
import tempfile

import json
import yaml

import cv2
import torch

def get_available_gpus():
    return [str(i) for i in range(torch.cuda.device_count())]

def is_on_windows():
    return sys.platform == "win32"

def get_tmp_dir():
    return tempfile.gettempdir()

def load_cache(fname):
    path = os.path.join(get_tmp_dir(), fname)
    if os.path.exists(path):
        return load_json(path)
    return None

def save_cache(obj, fname):
    path = os.path.join(get_tmp_dir(), fname)
    save_json(obj, path)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def save_text(text, path, is_list=False):
    with open(path, 'w') as f:
        if is_list:
            f.writelines(text)
        else:
            f.write(text)

def execute_command(cmd_str, process_queue=None, connection=None):
    print("Executing command:")
    print(cmd_str)
    p = subprocess.Popen(cmd_str, shell=True)

    if process_queue is not None:
        while True:
            data = process_queue.get()
            print(data)
            if data == "STOP" or pipe_conn.closed:
                # This works on Windows, may need to test on other platforms
                # See: https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true/4791612#4791612
                
                p.send_signal(signal.CTRL_C_EVENT)
                p.wait()
                print("Process ended")
                process_queue.close()
                return

    if connection is not None:
        p = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

        while True:
            if p.poll(): break
            line = p.stdout.readline()
            connection.send(line)

class VideoScanner:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def get_frame_size(self):
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (int(width), int(height))

    def get_num_frames(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

class PipeFile(io.RawIOBase):
    def __init__(self, conn):
        super().__init__()
        self.conn = conn

    def write(self, b):
        self.conn.send(b)

    