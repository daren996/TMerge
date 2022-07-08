# pylint: disable-all

import os


def run_all_methods(dataset_name, train_test='selected'):
    dataset_template = '../storage/dataset/PathTrack/{train_test}/{dataset_name}/img1'
    model_config = './e2e/configs/mmtracking/detector/faster_rcnn_r50_fpn_one_class.py'
    checkpoint = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
    output_template = '../storage/results/pathtrack/{dataset_name}/{det_method}-{method}-person.txt'

    # name, file, detection_method, [params]
    params_for_faster_rcnn = [
            '--config', model_config, 
            '--checkpoint', checkpoint,
    ]
    method_lists = [
        # ('sort', 'mmt_sort_private.py', 'faster_rcnn', params_for_faster_rcnn),
        # ('deepsort','mmt_deepsort_private.py', 'faster_rcnn', params_for_faster_rcnn),
        ('tracktor', 'mmt_tracktor_private.py', 'faster_rcnn', params_for_faster_rcnn),
        # ('uma', 'uma_private.py', 'faster_rcnn', params_for_faster_rcnn),
        # ('center_track', 'centertrack_private.py', 'center_net', [])
    ]

    for py in method_lists:
        name, pyf, det_method, extra_params = py
        tokens = [
            'python', 'e2e/ingestion_runner.py', 'e2e/configs/tracking/'+pyf,
            '--path', dataset_template.format(train_test=train_test, dataset_name=dataset_name),
            '--output', output_template.format(dataset_name=dataset_name, method=name, det_method=det_method),
            *extra_params
        ]
        command = ' '.join(tokens)
        print('working on method: ', name)
        print('executing command: ', command)
        os.system(command)
        # print(' '.join(tokens))


if __name__ == '__main__':
    dataset_names = ['0Xtp77A4zF4_0_2000', '0Xtp77A4zF4_1000_3000', '0Xtp77A4zF4_2000_4000', '0Xtp77A4zF4_3000_5000', '0Xtp77A4zF4_4000_6000', '0Xtp77A4zF4_5000_7000', '0Xtp77A4zF4_6000_8000', '0Xtp77A4zF4_7000_9000', '0Xtp77A4zF4_8000_10000', '0Xtp77A4zF4_9000_11000', '0Xtp77A4zF4_10000_12000', '0Xtp77A4zF4_11000_13000', '0Xtp77A4zF4_12000_14000', '0Xtp77A4zF4_13000_15000', '0Xtp77A4zF4_14000_16000', '0Xtp77A4zF4_15000_17000', '0Xtp77A4zF4_16000_18000', '0Xtp77A4zF4_17000_19000', '0Xtp77A4zF4_18000_20000', '0Xtp77A4zF4_19000_21000', '0Xtp77A4zF4_20000_21635',]
    # dataset_names = ['4LFwzgwRyrY_0_2000', '4LFwzgwRyrY_1000_3000', '4LFwzgwRyrY_2000_4000', '4LFwzgwRyrY_3000_5000', '4LFwzgwRyrY_4000_6000', '4LFwzgwRyrY_5000_7000', '4LFwzgwRyrY_6000_8000', '4LFwzgwRyrY_7000_9000', '4LFwzgwRyrY_8000_10000', '4LFwzgwRyrY_9000_11000', '4LFwzgwRyrY_10000_12000', '4LFwzgwRyrY_11000_12000', ]
    # dataset_names = ['5fhJSO5al8o_0_2000', '5fhJSO5al8o_1000_3000', '5fhJSO5al8o_2000_4000', '5fhJSO5al8o_3000_5000', '5fhJSO5al8o_4000_6000', '5fhJSO5al8o_5000_7000', '5fhJSO5al8o_6000_8000', '5fhJSO5al8o_7000_9000', '5fhJSO5al8o_8000_10000', '5fhJSO5al8o_9000_11000', '5fhJSO5al8o_10000_12000', '5fhJSO5al8o_11000_13000', '5fhJSO5al8o_12000_14000', '5fhJSO5al8o_13000_15000', '5fhJSO5al8o_14000_16000', '5fhJSO5al8o_15000_16000', ]
    # dataset_names = ['81ahuidFUWw_0_2000', '81ahuidFUWw_1000_3000', '81ahuidFUWw_2000_4000', '81ahuidFUWw_3000_5000', '81ahuidFUWw_4000_6000', '81ahuidFUWw_5000_7000', '81ahuidFUWw_6000_8000', '81ahuidFUWw_7000_9000', '81ahuidFUWw_8000_10000', '81ahuidFUWw_9000_11000', '81ahuidFUWw_10000_12000', '81ahuidFUWw_11000_13000', '81ahuidFUWw_12000_14000', '81ahuidFUWw_13000_15000', '81ahuidFUWw_14000_16000', '81ahuidFUWw_15000_17000', '81ahuidFUWw_16000_17500', ]
    # dataset_names = ['KQqFixbb-o8_0_2000', 'KQqFixbb-o8_1000_3000', 'KQqFixbb-o8_2000_4000', 'KQqFixbb-o8_3000_5000', 'KQqFixbb-o8_4000_6000', 'KQqFixbb-o8_5000_7000', 'KQqFixbb-o8_6000_8000', 'KQqFixbb-o8_7000_9000', 'KQqFixbb-o8_8000_9099']
    
    # dataset_names = ['6dHWDBvkR2g_0_2000', '6dHWDBvkR2g_1000_3000', '6dHWDBvkR2g_2000_4000', '6dHWDBvkR2g_3000_4471']
    # dataset_names = ['6zk5L6WxAXc_0_2000', '6zk5L6WxAXc_1000_3000', '6zk5L6WxAXc_2000_4000', '6zk5L6WxAXc_3000_5000', '6zk5L6WxAXc_4000_6000', '6zk5L6WxAXc_5000_7000', '6zk5L6WxAXc_6000_8000', '6zk5L6WxAXc_7000_9000', '6zk5L6WxAXc_8000_10000', '6zk5L6WxAXc_9000_11000', '6zk5L6WxAXc_10000_12000', '6zk5L6WxAXc_11000_13000', '6zk5L6WxAXc_12000_14000', '6zk5L6WxAXc_13000_15000', '6zk5L6WxAXc_14000_16000', '6zk5L6WxAXc_15000_17000', '6zk5L6WxAXc_16000_18000', '6zk5L6WxAXc_17000_19000', '6zk5L6WxAXc_18000_20000', '6zk5L6WxAXc_19000_20374']
    # dataset_names = ['8BQ-nVvJJMQ_0_2000', '8BQ-nVvJJMQ_1000_3000']
    # dataset_names = ['K6H9sY6hIeA_0_2000', 'K6H9sY6hIeA_1000_3000', 'K6H9sY6hIeA_2000_4000', 'K6H9sY6hIeA_3000_4765']
    
    for dn in dataset_names:
        run_all_methods(dn)
