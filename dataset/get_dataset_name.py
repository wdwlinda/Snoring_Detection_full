from utils.configuration import load_config


def get_dataset():
    name_to_path = load_config('dataset/dataset.yml')['data_pre_root']

    datasets = get_dataset_names()
    dataset_pair = []
    for dataset in datasets:
        pair = {}
        for dataset_name in dataset:
            pair[dataset_name] = name_to_path[dataset_name]
        dataset_pair.append(pair)
    return dataset_pair


def get_dataset_names():
    dataset1 = [
        'ASUS_snoring',
    ]
    dataset2 = [
        'Kaggle_snoring_pad'
    ]
    dataset3 = [
        'ESC50'
    ]
    dataset4 = [
        'Kaggle_snoring_pad',
        'ASUS_snoring',
    ]
    dataset5 = [
        'Kaggle_snoring_pad',
        'ASUS_snoring',
        'ESC50'
    ]
    dataset6 = [
        'Kaggle_snoring_pad',
        'ASUS_snoring',
        'ESC50',
        'redmi',
        'pixel',
        'iphone',
        'Mi11_office',
        'Mi11_night',
        'Samsung_Note10Plus_night',
        'Redmi_Note8_night',
    ]
    dataset7 = ['Audioset_snoring_strong_repeat']
    dataset8 = [
        'Audioset_snoring_strong_repeat',
        'Kaggle_snoring_pad',
        'ASUS_snoring',
        'ESC50',
        'redmi',
        'pixel',
        'iphone',
        'Mi11_office',
        'Mi11_night',
        'Samsung_Note10Plus_night',
        'Redmi_Note8_night',
    ]

    datasets = [
        # dataset4, dataset5, dataset6,
        # dataset1, 
        # dataset2, dataset3, 
        dataset6, 
        dataset7, 
        dataset8,
    ]
    # datasets = [
    #     # ['Audioset_snoring_strong'],
    #     ['Audioset_snoring_strong_0.8'],
    #     ['Audioset_snoring_strong_0.6'],
    #     ['Audioset_snoring_strong_0.4'],
    #     ['Audioset_snoring_strong_0.2'],
    #     ['Audioset_snoring_strong_repeat'],
    # ]
    return datasets

    