import os
import sys
import subprocess
import tempfile

import json

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
    
def save_text(text, path, is_list=False):
    with open(path, 'w') as f:
        if is_list:
            f.writelines(text)
        else:
            f.write(text)

def execute_command(cmd_str, process_queue=None):
    p = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

    if process_queue is not None:
        while True:
            data = process_queue.get()
            if data == "STOP" or pipe_conn.closed:
                p.kill()
                process_queue.close()
                return
    