import subprocess

import json

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

def execute_command(cmd_str, pipe_conn=None):
    p = subprocess.Popen(cmd_str, shell=True)

    if pipe_conn is not None:
        while p.poll() is None:
            for line in p.stdout.readlines():
                pipe_conn.send(line)
        pipe_conn.close()
    