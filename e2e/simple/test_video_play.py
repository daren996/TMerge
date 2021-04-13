from videosys.ingestion.visualize.image import ImageVisualizer
from videosys.ingestion.base import SimplePipelineBuilder
from videosys.ingestion.io.base import ImageSource, VideoSource

def play_video(video_path, auto_play=False):
    print('play video:', video_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(VideoSource(video_path))
    builder.add_operator(ImageVisualizer(auto_play=auto_play))
    builder.build().start()

def play_image_folder(folder_path, auto_play = False):
    print('play images in folder:', folder_path)
    builder = SimplePipelineBuilder()
    builder.add_operator(ImageSource(folder_path))
    builder.add_operator(ImageVisualizer(auto_play=auto_play))
    builder.build().start()

if __name__ == '__main__':
    play_video('/media/ytchen/hdd/dataset/videos/MOT16-03.mp4', True)
    # play_image_folder('/media/ytchen/hdd/dataset/2DMOT2015/test/ADL-Rundle-1/img1', False)
