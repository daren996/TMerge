from videosys.ingestion.io.base import ImageSource, VideoSink

description = 'generate video from images'

default_args = dict(
    folder=None,
    output_folder = None,
    output_name = None,
    fps='',
)

def operators(args):
    return [
        ImageSource(args.folder),
        VideoSink(args.output_folder, args.output_name, 
        fps= None if args.fps == '' else int(args.fps))
    ]
