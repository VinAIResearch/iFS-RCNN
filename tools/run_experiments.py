import argparse
import os
import yaml
from ast import literal_eval as make_tuple
from subprocess import PIPE, STDOUT, Popen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10, 30],
                        help='Shots to run experiments over')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 10],
                        help='Range of seeds to run')
    parser.add_argument('--root', type=str, default='./', help='Root of data')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of path')
    parser.add_argument('--suffix2', type=str, default='', help='Suffix of path')
    parser.add_argument('--base_config', type=str, default='', help='The config file to start from')
    parser.add_argument('--weights_file', type=str, default='', help='The weights file to load from')
    parser.add_argument('--network', type=str, default='mask_rcnn', help='faster_rcnn or mask_rcnn')
    parser.add_argument('--arch', type=str, default='50', help='50 or 101')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ckpt-freq', type=int, default=10,
                        help='Frequency of saving checkpoints')
    # Model
    parser.add_argument('--fc', action='store_true',
                        help='Model uses FC instead of cosine')
    parser.add_argument('--two-stage', action='store_true',
                        help='Two-stage fine-tuning')
    parser.add_argument('--novel-finetune', action='store_true',
                        help='Fine-tune novel weights first')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze feature extractor')
    # PASCAL arguments
    parser.add_argument('--split', '-s', type=int, default=1, help='Data split')
    # COCO arguments
    parser.add_argument('--coco', action='store_true', help='Use COCO dataset')

    args = parser.parse_args()
    return args


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line)


def run_exp(cfg, configs, seed, shot):
    """
    Run training and evaluation scripts based on given config files.
    """
    # Train
    output_dir = configs['OUTPUT_DIR']
    model_path = os.path.join(args.root, output_dir, 'model_final.pth')

    if 'test_all' in output_dir:
        seed_str = f'seed{seed}/' if seed != 0 else ''
        combine_model = f'checkpoints/coco/{args.network}/{seed_str}{args.network}_R_{args.arch}_FPN_all_final_{shot}shot{args.suffix}{args.suffix2}'

        test_cmd =  f'python3 -m tools.ckpt_surgery {"--coco" if args.coco else ""} ' \
                    f'--src1 checkpoints/coco/{args.network}/{args.network}_R_{args.arch}_FPN_base{args.suffix}/model_final_early.pth ' \
                    f'--src2 checkpoints/coco/{args.network}/{seed_str}{args.network}_R_{args.arch}_FPN_ft_novel_{shot}shot{args.suffix}{args.suffix2}/model_final.pth ' \
                    f'--method combine --save-dir {combine_model}'

        run_cmd(test_cmd)

        test_cmd =  f'python3 -m tools.test_net --num-gpus {args.num_gpus} ' \
                    f'--config-file {cfg} --eval-only --resume ' \
                    f'--opts MODEL.WEIGHTS {combine_model}/model_reset_combine.pth'

        run_cmd(test_cmd)

        return

    if not os.path.exists(model_path):
        train_cmd = f'python3 -m tools.train_net --dist-url auto --num-gpus {args.num_gpus} ' \
                    f'--config-file {cfg} --resume --opts MODEL.WEIGHTS {args.weights_file}'
        run_cmd(train_cmd)

    # Test
    res_path = os.path.join(args.root, output_dir, 'inference',
                            'res_final.json')
    if not os.path.exists(res_path):
        test_cmd = f'python3 -m tools.test_net --dist-url auto --num-gpus {args.num_gpus} ' \
                    f'--config-file {cfg} --resume --eval-only --opts MODEL.WEIGHTS {model_path}'

        run_cmd(test_cmd)


def get_config(seed, shot):
    """
    For a given seed and shot, generate a config file based on a template
    config file that is used for training/evaluation.
    You can extend/modify this function to fit your use-case.
    """
    if args.coco:
        # COCO
        # assert args.two_stage, 'Only supports novel weights for COCO now'
        if args.novel_finetune:
            # Fine-tune novel classifier
            mode = 'novel'
        else:
            # Fine-tune entire classifier
            mode = 'all'
        split = temp_split = ''
        temp_mode = mode

        config_dir = f'configs/COCO-detection/{args.network}'
        ckpt_dir = f'checkpoints/coco/{args.network}'
        base_cfg = '../../../Base-RCNN-FPN.yaml'
    else:
        # PASCAL VOC
        assert not args.two_stage, 'Only supports random weights for PASCAL now'
        split = 'split{}'.format(args.split)
        mode = 'all{}'.format(args.split)
        temp_split = 'split1'
        temp_mode = 'all1'

        config_dir = f'configs/PascalVOC-detection/{args.network}'
        ckpt_dir = f'checkpoints/voc/{args.network}'
        base_cfg = '../../../Base-RCNN-FPN.yaml'

    seed_str = 'seed{}'.format(seed) if seed != 0 else ''

    seed_exp = f'_{seed_str}' if seed_str else ''

    parts = args.base_config.split('/')[-1].split('1shot')
    prefix = f'{parts[0]}{shot}shot{parts[1]}'

    output_dir = os.path.join(args.root, ckpt_dir, seed_str)
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(
        args.root, config_dir, split, seed_str,
    )
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, prefix)

    configs = load_yaml_file(f'{args.root}{config_dir}/{prefix}')

    parts = configs['DATASETS']['TRAIN'].split("',")
    configs['DATASETS']['TRAIN'] = f"{parts[0]}{seed_exp}',{parts[1]}"
    configs['OUTPUT_DIR'] = os.path.join(output_dir, prefix.replace('.yaml', ''))
    configs['_BASE_'] = base_cfg

    if seed != 0:
        with open(save_file, 'w') as fp:
            yaml.dump(configs, fp, sort_keys=False)

    return save_file, configs


def main(args):
    for shot in args.shots:
        for seed in range(args.seeds[0], args.seeds[1]):
            print('Split: {}, Seed: {}, Shot: {}'.format(args.split, seed, shot))
            cfg, configs = get_config(seed, shot)
            run_exp(cfg, configs, seed, shot)


if __name__ == '__main__':
    args = parse_args()
    main(args)
