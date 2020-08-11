root_experiments = [
    {
        'dataset_name': 'MNIST',
        'consistency_type': 1,
        'used_rows': n,
    } for n in range(28)
] + [
    {
        'dataset_name': 'MNIST',
        'consistency_type': 2,
        'used_rows': n,
    } for n in range(28)
] + [
    {
        'dataset_name': 'MNIST',
        'consistency_type': 3,
        'used_rows': n,
    } for n in range(28)
] + [
    {
        'dataset_name': 'cifar10',
        'consistency_type': 1,
        'used_rows': n,
    } for n in range(32)
] + [
    {
        'dataset_name': 'cifar10',
        'consistency_type': 2,
        'used_rows': n,
    } for n in range(32)
] + [
    {
        'dataset_name': 'cifar10',
        'consistency_type': 3,
        'used_rows': n,
    } for n in range(32)
]

experiments = []
for i in range(10):
    for loss in ('smile', ):
#     for loss in ('mine', 'smile', 'imine'):
        for e in root_experiments:
            e = e.copy()
            e['loss'] = loss
            e['iteration'] = i
            experiments.append(e)


gpu_ids = (0, 1, 2, 3, 4, 5, 6, 7)
process_per_gpus = 7

experiments_per_queue = [
    {'commands': [], 'gpu_id': gpu_id, 'process_id': process_id, 'compiled_command': ''}
    for gpu_id in gpu_ids
    for process_id in range(process_per_gpus)
]
for index, experiment in enumerate(experiments):
    target_queue_index = index % len(experiments_per_queue)
    gpu_id = experiments_per_queue[target_queue_index]['gpu_id']
    experiments_per_queue[target_queue_index]['commands'].append(
        f'CUDA_VISIBLE_DEVICES={gpu_id} python3 consistency_experiment.py ' + ' '.join(f'--{k} {v}' for k, v in experiment.items())
    )

for q in experiments_per_queue:
    q['compiled_command'] = '(%s)&' % '; '.join(q['commands'])

print('\n'.join([q['compiled_command'] for q in experiments_per_queue]))
