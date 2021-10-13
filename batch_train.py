# TODO: Aggregate evalution result
import copy
import train
from utils import configuration
from utils import train_utils


def modify_config():
    pass


def main2():
    exp_config_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\experiment.yml'
    train_config_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_train_config.yml'
    eval_config_path = rf'C:\Users\test\Desktop\Leon\Projects\Snoring_Detection\config\_cnn_valid_config.yml'

    base_exp_config = configuration._load_config_yaml(exp_config_path)
    base_train_config = configuration._load_config_yaml(train_config_path)
    base_eval_config = configuration._load_config_yaml(eval_config_path)

    for exp, factors in exp_config.items():
        for f in factors:
            temp = factors[f]
            while isinstance(temp, dict):
                temp = temp[f]

        train_utils.replace_item(obj, key, replace_value)


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
    main()