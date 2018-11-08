import argparse
import os

from data_loader import get_loader
from Trainer import Trainer
from Tester import Tester
from Utils import compute_input_length

os.environ["CUDA_VISIBLE_DEVICES"]="1"


def train(config):

    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    exp_name = config.model_name + config.exp_number
    exp_path = os.path.join(config.log_path, exp_name)
    config.exp_path = exp_path

    if os.path.exists(exp_path):
        print('exp already exists! use other name')
        # return
    else:
        os.makedirs(exp_path)

    if config.context:
        # for context mode, previous and future information is needed.

        input_length, output_length = compute_input_length(config.input_length, config.num_layers,
                                                           config.enc_filter_size, config.dec_filter_size)

        config.input_length = input_length
        config.output_length = output_length

    train_loader, valid_loader = get_loader(config.data_path, config.batch_size,
                                            config.input_length, config.output_length)

    trainer = Trainer(train_loader, valid_loader, config)
    trainer.train()


def test(config):

    exp_name = config.model_name + config.exp_number
    exp_path = os.path.join(config.log_path, exp_name)
    config.exp_path = exp_path

    tester = Tester(config, config.target_wav)
    tester.predict()


def evaluate(config):

    exp_name = config.model_name + config.exp_number
    exp_path = os.path.join(config.log_path, exp_name)
    config.exp_path = exp_path

    tester = Tester(config, None)
    tester.produce_musdb_source_estimates()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model name

    parser.add_argument('--mode', choices=['train', 'test', 'eval'], default='train')
    parser.add_argument('--data_path', default='../dataset/musdb18')
    parser.add_argument('--log_path', default='../log/attn_separation')

    parser.add_argument('--model_name', choices=['attention'], default='attention')
    parser.add_argument('--exp_number', default='10')

    # architecture parameters

    parser.add_argument('--attention_fn', choices=['self_attention, cross_attention'], default='cross_attention')

    parser.add_argument('--num_sources', type=int, default=2)
    parser.add_argument('--context', type=bool, default=True)
    parser.add_argument('--input_length', type=int, default=16384)
    parser.add_argument('--output_length', type=int, default=16384)

    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_filters', type=int, default=24)
    parser.add_argument('--enc_filter_size', type=int, default=15)
    parser.add_argument('--dec_filter_size', type=int, default=5)

    # training setting

    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=bool, default=True)

    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--valid_step', type=int, default=1000)

    parser.add_argument('--num_iters', type=int, default=200000)

    parser.add_argument('--num_iters_decay', type=int, default=100000)

    # test setting

    parser.add_argument('--saved_model', default='132000-AttnNet.ckpt')
    parser.add_argument('--target_wav', default="../example/mallory.wav")
    # parser.add_argument('--target_wav', default="../example/Angels In Amplifiers - I'm Alright.stem_mix.wav")
    # parser.add_argument('--target_wav', default="../example/The Mountaineering Club - Mallory.stem_mix.wav")
    # parser.add_argument('--target_wav', default="../example/mallory_acc.wav")
    # parser.add_argument('--target_wav', default="../example/jamesmay_ifyousay_mix.wav")

    config = parser.parse_args()

    if config.mode == 'train':
        train(config)

    elif config.mode == 'test':
        test(config)

    elif config.mode == 'eval':
        evaluate(config)