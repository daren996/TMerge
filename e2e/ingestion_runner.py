import argparse
import sys
sys.path.append('../mmtracking')
sys.path.append('../mmdetection')
sys.path.append('../UMA-MOT')
sys.path.append('.')
import importlib
from videosys.ingestion.base import SimplePipelineBuilder


def parse_general_args(input_params):
    parser = argparse.ArgumentParser(description='run end to end pipelines')
    parser.add_argument('pipeline_config', help='the configuration file for the pipeline')
    return parser.parse_args(input_params)


def parse_config_args(desc, defaults, input_params, parsed_args):
    parser = argparse.ArgumentParser(description=desc, prog=' '.join(parsed_args))
    for key, value in defaults.items():
        _help = None
        _default = value
        if isinstance(value, tuple):
            _help = value[0]
            _default = value[1]
        _required = _default is None
        if isinstance(_default, bool):
            parser.add_argument('--{}'.format(key), help=_help,
                                action='store_true' if _default is False else 'store_false', default=_default)
        else:
            parser.add_argument('--{}'.format(key), help=_help,
                                default=_default, required=_required)
    return parser.parse_args(input_params)


if __name__ == '__main__':
    # argv: [this_script, config_script, ...]
    args = parse_general_args(sys.argv[1:2])
    # load python.
    config_file = args.pipeline_config
    config_import = importlib.import_module(config_file[:-3].replace('/', '.'))
    default_args = getattr(config_import, 'default_args')
    description = getattr(config_import, 'description') if hasattr(config_import, 'description') \
        else 'WARNNING: NO DESCRIPTION PROVIDED'
    # parse args
    speficied_args = parse_config_args(description, default_args, sys.argv[2:], sys.argv[:2])

    operators = getattr(config_import, 'operators')
    builder = SimplePipelineBuilder()
    for op in operators(speficied_args):
        builder.add_operator(op)
    builder.build().start()
