from utils.configuration import load_config


def get_dataset():
    name_to_path = load_config('dataset/dataset.yml')['dataset']
    dataset1 = {
        'train': ['ASUS_snoring_train'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset2 = {
        'train': ['ASUS_snoring_train', 'ESC50'],
        'valid': ['ASUS_snoring_test'],
    }
    
    dataset3 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset4 = {
        'train': ['ASUS_snoring_train', 'Mi11_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset5 = {
        'train': ['ASUS_snoring_train', 'Samsung_Note10Plus_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset6 = {
        'train': ['ASUS_snoring_train', 'ESC50', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset7 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset8 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test',
                  ],
        'valid': ['ASUS_snoring_test'],
    }

    dataset9 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test',
                  'pixel_0908', 'iphone11_0908'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset10 = {
        'train': ['ASUS_snoring_train', 'Mi11_office'],
        'valid': ['ASUS_snoring_test'],
    }

    datasets = [dataset9, dataset8, dataset4]
    # datasets = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]
    dataset_pair = []
    for dataset in datasets:
        dataset_path_format = {}
        for state, names in dataset.items():
            pair = {}
            for data_name in names:
                pair[data_name] = name_to_path[data_name]
            dataset_path_format[state] = pair
        dataset_pair.append(dataset_path_format)
    return dataset_pair


def get_dataset_wav():
    name_to_path = load_config('dataset/dataset.yml')['dataset_wav']
    dataset1 = {
        'train': ['ASUS_snoring_train'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset2 = {
        'train': ['ASUS_snoring_train', 'ESC50'],
        'valid': ['ASUS_snoring_test'],
    }
    
    dataset3 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset4 = {
        'train': ['ASUS_snoring_train', 'Mi11_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset5 = {
        'train': ['ASUS_snoring_train', 'Samsung_Note10Plus_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset6 = {
        'train': ['ASUS_snoring_train', 'ESC50', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset7 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset8 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test',
                  ],
        'valid': ['ASUS_snoring_test'],
    }

    dataset9 = {
        'train': ['ASUS_snoring_train', 'pixel', 'iphone', 
                  'Mi11_night_test', 'Mi11_office', 
                  'Redmi_Note8_night', 'Samsung_Note10Plus_night_test',
                  'pixel_0908', 'iphone11_0908'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset10 = {
        'train': ['ASUS_snoring_train', 'Mi11_office'],
        'valid': ['ASUS_snoring_test'],
    }

    dataset11 = {
        'train': ['ASUS_snoring_train'],
        'valid': ['ASUS_snoring_test', 'ESC50'],
    }

    dataset12 = {
        'train': ['ASUS_snoring_train'],
        'valid': ['Kaggle_snoring'],
    }

    dataset13 = {
        'train': ['ASUS_snoring_train', 'ESC50'],
        'valid': ['Kaggle_snoring'],
    }

    dataset14 = {
        'train': ['yt_snoring'],
        'valid': ['yt_snoring'],
    }
    # datasets = [dataset14]
    datasets = [dataset8, dataset10, dataset2]
    dataset_pair = []
    for dataset in datasets:
        dataset_path_format = {}
        for state, names in dataset.items():
            pair = {}
            for data_name in names:
                pair[data_name] = name_to_path[data_name]
            dataset_path_format[state] = pair
        dataset_pair.append(dataset_path_format)
    return dataset_pair