
from e2e.configs.utils import create_source_for_path
from videosys.ingestion.observer.reporter import ProgressReporter
from videosys.ingestion.io.tracking import MOTResultSink
from videosys.ingestion.tracking.centertrack import CenterTrackTracking
from videosys.ingestion.visualize.track import ObjectTrackingVisualizer


default_args = dict(
    path='../storage/dataset/MOT17/train/MOT17-11-DPM/img1',
    model_path = 'models/centertrack/mot17_half.pth',
    # model_path = 'models/centertrack/mot17_fulltrain.pth',
    # model_path = 'models/centertrack/coco_tracking.pth',
    display=False,
    output='../storage/results/mot17/MOT17-11-DPM/centertrack.txt',
    no_report_save = False,
    classes='',
    num_class='1'
)

def operators(args):
    ops = [
        create_source_for_path(args.path),
        CenterTrackTracking(args.model_path, 
            args.classes.split(',') if args.classes != '' else None,
            args.num_class if args.num_class != '' else None
            ),
    ]
    if not args.output == '':
        ops.append(MOTResultSink(args.output))
    if args.display:
        ops.append(ObjectTrackingVisualizer(display=True))
    ops.append(ProgressReporter(save_file = args.output+'.report' 
            if not args.no_report_save else None))
    return ops
