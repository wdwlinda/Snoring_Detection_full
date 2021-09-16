
import importlib
import os



# TODO: General solution
# TODO: Can string_filtering be called by generate_filenames?
# TODO: output only benign file?
# TODO: think about input output design of generate_filenames
def generate_filename_list(path, file_key, dir_key='', only_filename=False):
    input_paths, gt_paths = [], []
    for root, dirs, files in os.walk(path):
        for f in files:
            if not only_filename:
                fullpath = os.path.join(root, f)
            else:
                fullpath = f
            if dir_key in fullpath:
                if file_key in fullpath:
                    gt_paths.append(fullpath)
                else:
                    input_paths.append(fullpath)
    return input_paths, gt_paths


def generate_filenames(path, keys=None, include_or_exclude=None, is_fullpath=True, loading_formats=None):
    """Get all the file name under the given path with assigned keys and including condition"""
    # TODO: index error when keys=['a','a','a'] include_or_exclude=['include', 'exclude', 'exclude']
    if len(keys) == 0: keys = None
    if len(include_or_exclude) == 0: include_or_exclude = None
    if keys is not None and include_or_exclude is not None:
        assert len(keys) == len(include_or_exclude)
        full_keys = [f'{condition}_{k}' for k, condition in zip(keys, include_or_exclude)]
        # TODO: logging instead of print for repeat key (think about raise error or not)
        if len(full_keys) != len(list(set(full_keys))):
            print('Warning: Repeat keys')
            full_keys = list(set(full_keys))
    if keys is not None:
        filenames = {k: [] for k in full_keys}
    else:
        filenames = []

    for root, dirs, files in os.walk(path):
        for f in files:
            if loading_formats is not None:
                process = False
                for format in loading_formats:
                    if format in f:
                        process = True
                        break
            else:
                process = True

            if process:
                if is_fullpath:
                    final_path = os.path.join(root, f)
                else:
                    final_path = f
                if keys is not None:
                    for idx, k in enumerate(keys):
                        if include_or_exclude[idx] == 'include':
                            if k in final_path:
                                filenames[full_keys[idx]].append(final_path)
                        elif include_or_exclude[idx] == 'exclude': 
                            if k not in final_path:
                                filenames[full_keys[idx]].append(final_path)
                        else:
                            raise ValueError('Unknown key condition')
                else:
                    filenames.append(final_path)
    return filenames

    
def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def load_content_from_txt(path, access_mode='r'):
    with open(path, access_mode) as fw:
        content = fw.read().splitlines()
    return content


def get_data_indices(data_name, data_path, save_path, data_split, mode, generate_index_func):
    """"Get dataset indices and create if not exist"""
    # # TODO: gt not exist condition
    # create index folder and sub-folder if not exist
    os.chdir(save_path)
    index_dir_name = f'{data_name}_data_index'
    sub_index_dir_name = f'{data_name}_{data_split[0]}_{data_split[1]}'
    input_data_path = os.path.join(save_path, index_dir_name, sub_index_dir_name)
    if not os.path.isdir(input_data_path): os.makedirs(input_data_path)
    
    # generate index list and save in txt file
    generate_index_func(data_path, data_split, input_data_path)
        
    # load index list from saved txt
    os.chdir(input_data_path)
    if os.path.isfile(f'{mode}.txt'):
        input_data_indices = load_content_from_txt(f'{mode}.txt')
        input_data_indices.sort()
    else:
        input_data_indices = None

    if os.path.isfile(f'{mode}_gt.txt'):
        ground_truth_indices = load_content_from_txt(f'{mode}_gt.txt')
        ground_truth_indices.sort()
    else:
        ground_truth_indices = None
    return input_data_indices, ground_truth_indices


def generate_kaggle_breast_ultrasound_index(data_path, save_path, data_split):
    data_keys = {'input': 'exclude_mask', 'ground_truth': 'include_mask'}
    save_input_and_label_index(data_path, save_path, data_split, data_keys, loading_format=['png', 'jpg'])



def generate_kaggle_snoring_index(data_path, save_path, data_split):
    # TODO: no ground truth case
    # TODO: keys=None, include_or_exclude=None case
    data_keys = {'input': None}
    save_input_and_label_index(data_path, save_path, data_split, data_keys, loading_format=['wav', 'm4a'])


def save_input_and_label_index(data_path, save_path, data_split, data_keys=None, loading_format=None):
    # TODO: test input output
    # assert 'input' in data_keys, 'Undefined input data key'
    # assert 'ground_truth' in data_keys, 'Undefined ground truth data key'
    class_name = os.listdir(data_path)
    os.chdir(save_path)
    include_or_exclude, keys = [], []
    if data_keys is not None:
        for v in data_keys.values():
            if v is not None:
                include_or_exclude.append(v.split('_')[0])
                keys.append(v.split('_')[1]) 
    data_dict = generate_filenames(
        data_path, keys=keys, include_or_exclude=include_or_exclude, is_fullpath=True, loading_formats=loading_format)

    def save_content_ops(data, train_name, valid_name):
        # TODO: Is this a solid solution?
        data.sort(key=len)
        split = int(len(data)*data_split[0])
        train_input_data, val_input_data = data[:split], data[split:]
        save_content_in_txt(train_input_data,  train_name, filter_bank=class_name, access_mode="w+", dir=save_path)
        save_content_in_txt(val_input_data, valid_name, filter_bank=class_name, access_mode="w+", dir=save_path)


    # if data_keys['input'] is not None:    
    #     input_data = data_dict[data_keys['input']]
    # else:
    #     input_data = data_dict
    
    if data_keys is not None:
        if 'input' in data_keys:
            if data_keys['input'] is not None:
                input_data = data_dict[data_keys['input']]
            else:
                input_data = data_dict
            save_content_ops(input_data, 'train.txt', 'valid.txt')
        if 'ground_truth' in data_keys:
            if data_keys['ground_truth'] is not None:
                ground_truth = data_dict[data_keys['ground_truth']]
            else:
                ground_truth = data_dict
            save_content_ops(ground_truth, 'train_gt.txt', 'valid_gt.txt')
    else:
        input_data = data_dict
        save_content_ops(input_data, 'train.txt', 'valid.txt')
        
    # input_data.sort()
    # split = int(len(input_data)*data_split[0])
    # train_input_data, val_input_data = input_data[:split], input_data[:split]
    # save_content_in_txt(train_input_data, 'train.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    # save_content_in_txt(val_input_data, 'valid.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    # if 'ground_truth' in data_keys:
    #     ground_truth = data_dict[data_keys['ground_truth']]
    #     ground_truth.sort()
    #     train_ground_truth, valid_ground_truth = ground_truth[split:], ground_truth[split:]
    #     save_content_in_txt(train_ground_truth, 'train_gt.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    #     save_content_in_txt(valid_ground_truth, 'valid_gt.txt', filter_bank=class_name, access_mode="w+", dir=save_path)


def string_filtering(s, filter):
    filtered_s = {}
    for f in filter:
        if f in s:
            filtered_s[f] = s
    if len(filtered_s) > 0:
        return filtered_s
    else:
        return None


def save_content_in_txt(content, path, filter_bank, access_mode='a+', dir=None):
    assert isinstance(content, (str, list, tuple, dict))
    # TODO: overwrite warning
    with open(path, access_mode) as fw:
        def string_ops(s, dir, filter):
            pair = string_filtering(s, filter)
            return os.path.join(dir, list(pair.keys())[0], list(pair.values())[0])

        if isinstance(content, str):
            if dir is not None:
                content = string_ops(content, dir, filter=filter_bank)
                # content = os.path.join(dir, content)
            fw.write(content)
        else:
            for c in content:
                if dir is not None:
                    c = string_ops(c, dir, filter=filter_bank)
                    # c = os.path.join(dir, c)
                fw.write(f'{c}\n')


if __name__ == "__main__":
    # generate_kaggle_breast_ultrasound_index(
    #     data_path=rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\archive\Dataset_BUSI_with_GT',
    #     data_split=(0.7, 0.3),
    #     save_path=rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\dataset\index\test')

    # get_data_indices(
    #     data_name='BU', 
    #     data_path=rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\archive\Dataset_BUSI_with_GT', 
    #     save_path=rf'C:\Users\test\Desktop\Leon\Projects\Breast_Ultrasound\dataset\index\test',
    #     mode='valid',
    #     data_split=(0.7, 0.3), 
    #     generate_index_func=generate_kaggle_breast_ultrasound_index)


    # filenames = generate_filenames(path=rf'C:\Users\test\Desktop\Leon\Datasets\Kaggle_Breast_Ultraound\archive\Dataset_BUSI_with_GT',
    #                                keys=['mask', 'png'], 
    #                                include_or_exclude=['include', 'include'], 
    #                                is_fullpath=True)
    # a = 3

    generate_kaggle_snoring_index(
        data_path=rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset\1',
        save_path=rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\dataset',
        data_split=(0.7, 0.3))

    # get_data_indices(
    #     data_name='Snoring', 
    #     data_path=rf'C:\Users\test\Desktop\Leon\Datasets\Snoring_Detection\Snoring Dataset', 
    #     save_path=rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\dataset',
    #     mode='valid',
    #     data_split=(0.7, 0.3), 
    #     generate_index_func=generate_kaggle_snoring_index)