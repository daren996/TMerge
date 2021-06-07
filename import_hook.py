import sys

with open('.env', 'r') as f:
    for line in f.readlines():
        if line.startswith('PYTHONPATH='):
            for _path in line[11:].split(':'):
                if len(_path.strip()) > 0:
                    sys.path.append(_path)
