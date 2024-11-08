# python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

parser_choices = {
    'dataset': ['gtsrb', 'cifar10', 'cifar100', 'imagenette', 'ember', 'imagenet'],
    'poison_type': [  # Poisoning attacks
        'basic', 'badnet', 'blend', 'dynamic', 'clean_label', 'TaCT', 'SIG', 'WaNet', 'refool', 'ISSBA',
        'adaptive_blend', 'adaptive_patch', 'adaptive_k_way', 'none', 'badnet_all_to_all', 'trojan', 'SleeperAgent',
        # Other attacks
        'trojannn', 'BadEncoder', 'SRA', 'bpp', 'WB'],
    # 'poison_rate': [0, 0.001, 0.002, 0.004, 0.005, 0.008, 0.01, 0.015, 0.02, 0.05, 0.1],
    # 'cover_rate': [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
    'poison_rate': [i / 1000.0 for i in range(0, 500)],
    'cover_rate': [i / 1000.0 for i in range(0, 500)],
    'cleanser': ['SCAn', 'AC', 'SS', 'Strip', 'CT', 'SPECTRE', 'SentiNet', 'Frequency'],
    'defense': ['ABL', 'NC', 'STRIP', 'FP', 'NAD', 'SentiNet', 'ScaleUp', 'SEAM', 'SFT', 'NONE', 'Frequency', 'AC', 'moth'
                'FeatureRE'],
}

# parser_default = {
#     'dataset': 'cifar10',
#     'poison_type': 'badnet',
#     'poison_rate': 0.003,
#     'cover_rate': 0.003,
#     'alpha': 0.15,
#     'defense': 'badnet'
# }

parser_default = {
    'dataset': 'imagenet',
    'poison_type': 'badnet',
    'poison_rate': 0.003,
    'cover_rate': 0.003,
    'alpha': 0.5,
    # 'test_alpha': 0.2,
    'defense': 'badnet',
    'cleanser': 'SCAn',
    'trigger': 'firefox',
}


attr_parser_choices = {
    'attr_method': ['InputGrad', 'IntGrad', 'ExpGrad', 'IG_SG', 'IG_SQ', 'IG_Uniform', 'AGI', 'FullGrad',
                    'SmoothGrad', 'Random', 'GradCAM', 'InputxGrad', 'GuidedGradCAM', 'LPI'],
    'metric': ['DiffID', 'visualize'],
    'est_method': ['vanilla', 'valid_ip', 'valid_ref'],
    'exp_obj': ['logit', 'prob', 'contrast'],
    'post_process': ['absolute', 'origin'],
}

attr_parser_default = {
    'attr_method': 'GradCAM',
    'metric': 'visualize',
    'k': 5,  # 5
    'bg_size': 10,  # 10
    'est_method': 'vanilla',
    'exp_obj': 'prob',
    'post_process': 'absolute',
}

seed = 2333  # 999, 999, 666 (1234, 5555, 777)
