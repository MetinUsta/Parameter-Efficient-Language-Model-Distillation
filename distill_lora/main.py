import os
import sys

# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distill_lora.train import distill_lora
from distill_lora.train.preprocess_dataset import preprocess_dataset
from distill_lora.utils.parse_configs import TrainConfig

if __name__ == "__main__":
    # parse cli arguments
    config_path = sys.argv[1]

    # parse config
    train_config = TrainConfig.from_yaml(config_path)

    # preprocess dataset
    dataset = preprocess_dataset(train_config)

    # distill lora
    distill_lora(train_config, dataset)
