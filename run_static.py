import argparse

import torch

from sgnnbenchmark.utils import create_log_file, configs
from sgnnbenchmark.trainer import Exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='./configs/static/gc_snn.yaml')
    args = parser.parse_args()
    configs = configs.load_config_from_file(args.config_dir)
    
    device = torch.device(f'cuda:{configs.shared.gpu}' if torch.cuda.is_available() else 'cpu')
    configs.shared['device'] = device

    create_log_file(configs.shared.log_dir)
    
    Exp(configs).run()