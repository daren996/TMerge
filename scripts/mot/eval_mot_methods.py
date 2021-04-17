from tools.mot.mot_method_eval import load_and_eval

if __name__ == '__main__':
    dataset = 'MOT17-09-DPM'
    result_template= '../storage/results/mot17/{dataset}/faster_rcnn-{method}-person.txt'
    methods_list = [
        ('SORT', 'sort'),
        ('DeepSORT', 'deepsort'),
        ('Tracktor', 'tracktor'),
        ('UMA-MOT', 'uma'),
    ]
    gt_file = '/media/ytchen/hdd/dataset/MOT17/train/{dataset}/gt/gt.txt'


    load_and_eval(
        [(m[0], result_template.format(dataset=dataset, method=m[1])) for m in methods_list], 
        gt_file.format(dataset=dataset)
    )
