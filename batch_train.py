# TODO: Aggregate evalution result
import copy

from numpy.lib.arraysetops import isin
import train
from utils import configuration
from utils import train_utils
from pprint import pprint
# import collections.abc


# TODO: Understand collections.abc.Mapping
def update(d, u):
    for k, v in u.items():
        # if isinstance(v, collections.abc.Mapping):
        if isinstance(v, dict):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def test():
    exp_config_path = rf'config/experiment.yml'
    train_config_path = rf'config/_cnn_train_config.yml'

    base_exp_config = configuration._load_config_yaml(exp_config_path)
    base_train_config = configuration._load_config_yaml(train_config_path)

    # pprint(base_exp_config)


    def xx(d, replace_key=None):
        stack = [d]
        keys = []
        while stack:
            node = stack.pop(0)
            for n in node:
                if isinstance(n, dict):
                    stack.append(node[n])
                else:
                    keys.append(n)
        return keys
                

    def iterate_dict(d, replace_key=None):
        for k in d:
            print(d, k)
            if isinstance(d[k], list):
                if replace_key: 
                    d[k] = replace_key
                else:
                    return d[k]
            else:
                iterate_dict(d[k], replace_key)


    # k = xx(base_exp_config, replace_key=None)
    dict_list = []
    params_list = []
    for exp in base_exp_config:
        for factor in base_exp_config[exp]:
            # pprint(factor)
            params_list.append(iterate_dict(base_exp_config[exp][factor]))
            for p in params_list:
                # print(p)
                iterate_dict(base_exp_config[exp][factor], p)
                # dict_list.append(copy.deepcopy(update(base_train_config, base_exp_config['exp2'])))
                exp_copy = copy.deepcopy(base_exp_config)
                train_copy = copy.deepcopy(base_train_config)
                train_copy.update(exp_copy[exp])
                dict_list.append(train_copy)
    print(params_list)
    # pprint(dict_list)
    # for i, d in enumerate(dict_list):
    #     print(30*str(i))
    #     pprint(d)
    return dict_list


        # pprint(base_exp_config)
        # pprint(base_train_config)
        # update(base_train_config, base_exp_config['exp2'])
        # pprint(base_train_config)


def main2():
    exp_config_path = rf'config/experiment.yml'
    train_config_path = rf'config/_cnn_train_config.yml'

    base_exp_config = configuration._load_config_yaml(exp_config_path)
    base_train_config = configuration._load_config_yaml(train_config_path)

    def iterate_dict(d, replace_key=None):
        for k in d:
            if isinstance(d[k], list):
                if replace_key: 
                    d[k] = replace_key
                else:
                    return d[k]
            else:
                iterate_dict(d[k], replace_key)

    if base_exp_config:
        dict_list = []
        params_list = []
        for exp in base_exp_config:
            for factor in base_exp_config[exp]:
                
                # pprint(factor)
                params_list.append(iterate_dict(base_exp_config[exp][factor]))
                for p in params_list:
                    # print(p)
                    iterate_dict(base_exp_config[exp][factor], p)
                    # dict_list.append(copy.deepcopy(update(base_train_config, base_exp_config['exp2'])))
                    exp_copy = copy.deepcopy(base_exp_config)
                    train_copy = copy.deepcopy(base_train_config)
                    train_copy.update(exp_copy[exp])
                    # train
                    train.main(train_copy)
    else:
        train.main(base_train_config)
    return dict_list


def main():
    exp_config_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\experiment.yml'
    train_config_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_train_config.yml'
    eval_config_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_valid_config.yml'

    base_exp_config = configuration._load_config_yaml(exp_config_path)
    base_train_config = configuration._load_config_yaml(train_config_path)
    base_eval_config = configuration._load_config_yaml(eval_config_path)

    exp_config = copy.deepcopy(base_exp_config)
    
    if base_exp_config:
        for exp, params in exp_config.items():
            # modify configuration
            var1 = list(params.keys())[0]
            var2 = list(params[var1].keys())[0]
            for var3 in params[var1][var2]:
                print(f'Running {exp} {var3}')
                train_config = copy.deepcopy(base_train_config)
                train_config[var1][var2] = var3
                
                # train
                train.main(train_config)

                # # evaluation
                # eval.main(eval_config)
    else:
        train.main(base_train_config)


if __name__ == '__main__':
    # test()
    main()
    # main2()