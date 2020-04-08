import pathlib

gpu_ids = [1, 2, 3, 4, 5, 6, 7, ]
process_per_gpus = 2

pretrained_model_paths = sorted(list(pathlib.Path('./pretrained_weights').glob('*.pth')), reverse=True)
comparisons = ('input', 'label')
target_indices = [0, 1, 2, 3, 4, ]

experiments = []
for pretrained_model_path in pretrained_model_paths:
    for compare_to in comparisons:
        for target_index in target_indices:
            experiments.append({
                'pretrained_model_path': pretrained_model_path,
                'compare_to': compare_to,
                'target_index': target_index,
                'model': 'resnet18',
            })

experiments_per_queue = [
    {'commands': [], 'gpu_id': gpu_id, 'process_id': process_id, 'compiled_command': ''}
    for gpu_id in gpu_ids
    for process_id in range(process_per_gpus)
]
for index, experiment in enumerate(experiments):
    target_queue_index = index % len(experiments_per_queue)
    gpu_id = experiments_per_queue[target_queue_index]['gpu_id']
    experiments_per_queue[target_queue_index]['commands'].append(
        f'python3 run.py --gpu_id {gpu_id} ' + ' '.join(f'--{k} {v}' for k, v in experiment.items())
    )

for q in experiments_per_queue:
    q['compiled_command'] = '(%s)&' % '; '.join(q['commands'])

print('\n'.join([q['compiled_command'] for q in experiments_per_queue]))
