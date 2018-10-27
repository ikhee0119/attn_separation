import argparse
from data_loader import get_loader
from Trainer import Trainer

def train(config):

    train_loader, valid_loader = get_loader(config.data_path, config.input_length,
                                              config.batch_size)

    trainer = Trainer(train_loader, config)
    trainer.train()

def test(config):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--data_path', default='../dataset/musdb18')

    parser.add_argument('--checkpoint_path', default='../checkpoint')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--input_length', type=int, default=16384)

    config = parser.parse_args()

    if config.mode == 'train':
        train(config)

    elif config.mode == 'test':
        test(config)