import argparse
import importlib
from videosys.ingestion.base import SimplePipelineBuilder

def parse_args():
    parser = argparse.ArgumentParser(description='run end to end pipelines')
    parser.add_argument('pipeline_config', help='the configuration file for the pipeline')
    return parser.parse_args()

if __name__ =='__main__':
    args = parse_args()
    # load python.
    config_file = args.pipeline_config
    config_import = importlib.import_module(config_file[:-3].replace('/','.'))
    operators = getattr(config_import, 'operators')
    builder = SimplePipelineBuilder()
    for op in operators:
        builder.add_operator(op)
    builder.build().start()
