# pylint: disable-all
import os
import motmetrics as mm
from motmetrics.io import Format
import pandas as pd
import matplotlib.pyplot as plt

# configs
default_configs = dict(
    sort=dict(
        obj_score_threshold='0.5',
        match_iou_threshold='0.5',
    ),
    deepsort=dict(
        obj_score_threshold='0.5',
        match_iou_threshold='0.5',
        num_samples='10',
        match_score_threshold='2.0',
    ),
    tracktor=dict(
        obj_score_threshold='0.5',
        match_iou_threshold='0.5',
        num_samples='10',
        match_score_threshold='2.0'
    )
)

vary_configs = dict(
    obj_score_threshold=['0.3', '0.4', '0.6', '0.7', '0.8'],
    match_iou_threshold=['0.3', '0.4', '0.6', '0.7', '0.8'],
    num_samples=['5', '15', '20'],
    match_score_threshold=['1.0', '1.5', '2.5', '3.0']
)

def run_method_for_dataset(dataset):
    print('producing result for', dataset)

    dataset_template = '../storage/dataset/MOT17/train/{dataset_name}/img1'
    model_config = './e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py'
    checkpoint = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
    output_template = '../storage/results/mot17/{dataset_name}/params/{method}-{params}.txt'

    method_lists = [
        ('sort', 'mmt_sort_private.py'),
        ('deepsort', 'mmt_deepsort_private.py'),
        ('tracktor', 'mmt_tracktor_private.py'),
    ]

    def run_with_params(name, pyf, params):
        tokens = [
            'python', 'e2e/ingestion_runner.py', 'e2e/configs/tracking/'+pyf,
            '--config', model_config,
            '--checkpoint', checkpoint,
            '--path', dataset_template.format(dataset_name=dataset),
            '--output', output_template.format(dataset_name=dataset, method=name, 
                params='-'.join([v for v in params.values()])),
            *['--{} {}'.format(k, v) for k,v in params.items()]]
        command = ' '.join(tokens)
        print('working on method: ', name)
        print('executing command: ', command)
        os.system(command)

    for method in method_lists:
        name, pyf = method
        default_settings = dict(default_configs[name])
        # 1. eval default
        run_with_params(name, pyf, default_settings)

        for config_key, config_values in vary_configs.items():
            param_settings = dict(default_settings)
            if config_key in param_settings:
                print('varying config: {} for {}'.format(config_key, name))
                for value in config_values:
                    # replace
                    param_settings[config_key] = value
                    run_with_params(name, pyf, param_settings)
            else:
                print('skip config: {} for {}'.format(config_key, name))
        

def eval_param(dataset):
    print('eval dataset: ', dataset)
    gt_file = '../storage/dataset/MOT17/train/{dataset}/gt/gt.txt'
    result_template = '../storage/results/mot17/{dataset_name}/params/{method}-{params}.txt'
    output_template = '../storage/results/mot17/{dataset_name}/params/{method}-motmetrics.txt'

    # load gt
    gt = mm.io.loadtxt(gt_file.format(dataset=dataset), fmt=Format.MOT16, min_confidence=1)

    def load_method_and_compare(gt, method, params, names, accs):
        method_file = result_template.format(dataset_name=dataset, method=method, 
                params='-'.join([v for v in params.values()]))
        method_pd = mm.io.loadtxt(method_file, fmt=Format.MOT16)
        acc= mm.utils.compare_to_groundtruth(gt, method_pd, 'iou', distth=0.5)
        names.append('-'.join([v for v in params.values()]))
        accs.append(acc)

    # for each method.
    for method, default_settings in default_configs.items():
        accs = []
        names = []
        # get default setting & eval
        load_method_and_compare(gt, method, default_settings, names, accs)

        for config_key, config_values in vary_configs.items():
            if config_key in default_settings:
                param_settings = dict(default_settings)
                for value in config_values:
                    param_settings[config_key] = value
                    load_method_and_compare(gt, method, param_settings, names, accs)
        # store.
        mh = mm.metrics.create()
        summary = mh.compute_many(accs, names=names, metrics=list(mm.metrics.motchallenge_metrics))
        strsummary = mm.io.render_summary(summary, formatters=mh.formatters, 
            namemap=mm.io.motchallenge_metric_names)
        print(strsummary)
        with open(output_template.format(dataset_name=dataset, method=method), 'w') as f:
            f.write(strsummary)
        

def parse_motmetrics(dataset):
    metrics_template = '../storage/results/mot17/{dataset_name}/params/{method}-motmetrics.txt'
    txt_data_template = '../storage/results/mot17/{dataset_name}/params/plot-data-{method}-{param}.txt'
    pic_data_template = '../storage/results/mot17/{dataset_name}/params/plot-{method}-{param}.png'

    def gen_key_from_params(params):
        return '-'.join([v for v in params.values()])

    for method, default_settings in default_configs.items():
        metrics_pd = pd.read_csv(metrics_template.format(dataset_name=dataset, method=method),
            sep=r'\s+')
        for config_key, config_values in vary_configs.items():
            if config_key in default_settings:
                # 1. get default
                default_value = float(default_settings[config_key])
                columns = [config_key, 'MOTA', 'IDs']

                default_data = metrics_pd.loc[gen_key_from_params(default_settings)]
                data = []
                data.append([default_value, default_data['MOTA'], default_data['IDs']])

                params = dict(default_settings)
                for value in config_values:
                    params[config_key] = value
                    current_data = metrics_pd.loc[gen_key_from_params(params)]
                    data.append([float(value), current_data['MOTA'], current_data['IDs']])

                method_param_pd = pd.DataFrame(data=data, columns=columns)
                method_param_pd['MOTA'] = method_param_pd['MOTA'].apply(lambda x: float(x[:-1]))
                method_param_pd['IDs'] = method_param_pd['IDs'].apply(int)
                method_param_pd.sort_values(by=config_key, inplace=True)
                method_param_pd.reset_index()
                
                # save row data
                method_param_pd.to_csv(txt_data_template.format(dataset_name=dataset, method=method, param=config_key))

                # draw pic.
                ax = method_param_pd.plot(x=config_key, y="MOTA", legend=False, marker='+')
                ax2 = ax.twinx()
                method_param_pd.plot(x=config_key, y="IDs", ax=ax2, legend=False, color="r", marker='x')
                ax.figure.legend()
                ax.set_title('{}-{}'.format(dataset, config_key))
                plt.savefig(pic_data_template.format(dataset_name=dataset, method=method, param=config_key), dpi=100)
                plt.close()
                


if __name__ == '__main__':
    # run_method_for_dataset('MOT17-09-DPM')
    # run_method_for_dataset('MOT17-11-DPM')
    # run_method_for_dataset('MOT17-13-DPM')
    # eval_param('MOT17-09-DPM')
    # eval_param('MOT17-11-DPM')
    # eval_param('MOT17-13-DPM')
    parse_motmetrics('MOT17-09-DPM')
    parse_motmetrics('MOT17-11-DPM')
    parse_motmetrics('MOT17-13-DPM')
