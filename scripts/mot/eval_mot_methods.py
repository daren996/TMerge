from tools.mot.mot_method_eval import load_and_eval

def eval_all_methods(dataset):
    print('dataset:', dataset)
    result_template= '../storage/results/mot17/{dataset}/{det_method}-{method}-person.txt'
    methods_list = [
        ('SORT', 'sort', 'faster_rcnn'),
        ('DeepSORT', 'deepsort', 'faster_rcnn'),
        ('Tracktor', 'tracktor', 'faster_rcnn'),
        ('UMA-MOT', 'uma', 'faster_rcnn'),
        # ('CenterTrack', 'center_track', 'center_net')
    ]
    gt_file = '../storage/dataset/MOT17/train/{dataset}/gt/gt.txt'

    load_and_eval(
        [(m[0], result_template.format(dataset=dataset, method=m[1], det_method=m[2])) 
            for m in methods_list], 
        gt_file.format(dataset=dataset)
    )

if __name__ == '__main__':
    eval_all_methods('MOT17-04-FRCNN')
    eval_all_methods('MOT17-09-FRCNN')
    eval_all_methods('MOT17-11-FRCNN')
    # eval_all_methods('MOT17-02-FRCNN')
    # eval_all_methods('MOT17-13-FRCNN')
