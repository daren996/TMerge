from videosys.ingestion.visualize import ImageVisualizer
from videosys.ingestion.base import SimplePipelineBuilder
from videosys.ingestion.io import VideoSource

def play_video(video_path, auto_play=False):
    print('play video:', video_path)
    builder = SimplePipelineBuilder()
    source_config = { 'file': video_path }
    builder.add_operator(VideoSource(source_config))
    builder.add_operator(ImageVisualizer({'auto_play': auto_play}))
    builder.build().start()

if __name__ == '__main__':
    play_video('/media/ytchen/hdd/dataset/videos/MOT16-03.mp4', False)
