import os
from tools.mot.mot_mapping import generate_summary, load_and_compute_mapping

def produce_result_for_dataset(dataset):
    print('producing result for ', dataset)
    
    data_folder = '../storage/results/mot17'
    det_method = 'faster_rcnn'
    template = "{}/{}/{}-{}-person.txt"
    gt_template = '../storage/dataset/MOT17/train/{}/gt/gt.txt'
    output_template = '../storage/results/mot17/{}/mapping/{}-{}-person.xlsx'

    methods = [('center_track', 'center_net'), 'tracktor', 'sort', 'deepsort', 'uma']

    for method in methods:
        if isinstance(method, tuple):
            method, _det_method = method
        else:
            _det_method = det_method
        oid_results, event_df = load_and_compute_mapping(
            template.format(data_folder, dataset, _det_method, method) , 
            gt_template.format(dataset)
        )
        pdf = generate_summary(oid_results, event_df)

        output_path = output_template.format(dataset, _det_method, method)
        ppath= os.path.dirname(output_path)
        if not os.path.exists(ppath):
            os.makedirs(ppath)
        pdf.to_excel(output_path)


if __name__ == '__main__':
    produce_result_for_dataset('MOT17-09-DPM')
    produce_result_for_dataset('MOT17-11-DPM')
    produce_result_for_dataset('MOT17-13-DPM')
