from tools.mot.mot_duration_statistics import StatisticsProducer

def show_statistics_for_dataset(dataset):
    data_folder = '../storage/results/mot17'
    det_method = 'faster_rcnn'
    template = "{}/{}/{}-{}-person.txt"
    p = StatisticsProducer([
        ('centertrack', template.format(data_folder, dataset, 'center_net', 'center_track')),
        ('tracktor', template.format(data_folder, dataset, det_method, 'tracktor')),
        ('sort', template.format(data_folder, dataset, det_method, 'sort')),
        ('deepsort', template.format(data_folder, dataset, det_method, 'deepsort')),
        ('uma', template.format(data_folder, dataset, det_method, 'uma'))
    ])
    p.load_and_compute_all()
    # p.show()
    p.save('../storage/results/mot17/{}/statistics.png'.format(dataset), dpi=1000)

if __name__ == '__main__':
    show_statistics_for_dataset('MOT17-09-DPM')
    show_statistics_for_dataset('MOT17-11-DPM')
    show_statistics_for_dataset('MOT17-13-DPM')
