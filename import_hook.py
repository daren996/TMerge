import os
import sys


def _append_pythonpath(path_value):
    for _path in path_value.split(':'):
        if _path.strip():
            sys.path.append(_path)


env_pythonpath = os.environ.get('PYTHONPATH', '')
if env_pythonpath:
    _append_pythonpath(env_pythonpath)

if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f.readlines():
            if line.startswith('PYTHONPATH='):
                _append_pythonpath(line[11:])
