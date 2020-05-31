gpu_ids = (0, 1, 2, 3, 4, 5, 6, 7, )
process_per_gpus = 7

batch_sizes = list(range(1, 101))
regularizer_weights = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]

experiments = []
for batch_size in batch_sizes:
    for regularizer_weight in regularizer_weights:
        experiments.append({
            'batch_size': batch_size,
            'regularizer_weight': regularizer_weight,
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
        f'python3 run_single.py --gpu_id {gpu_id} ' + ' '.join(f'--{k} {v}' for k, v in experiment.items())
    )

for q in experiments_per_queue:
    q['compiled_command'] = '(%s)&' % '; '.join(q['commands'])

print('\n'.join([q['compiled_command'] for q in experiments_per_queue]))
