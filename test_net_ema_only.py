import os
import glob
from xmodaler.config import get_cfg
from xmodaler.engine import default_argument_parser, default_setup, launch, build_engine
from xmodaler.modeling import add_config
from xmodaler.engine.hooks import EvalHook
from xmodaler.utils.events import EventStorage

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.debug:
        cfg.DEBUG = True
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.DATALOADER.TRAIN_BATCH_SIZE = 2
        cfg.DATALOADER.TEST_BATCH_SIZE = 4
        cfg.DATALOADER.FEATS_FOLDER = '../open_source_dataset/mscoco_dataset/features/up_down_100'
        if args.eval_only:
            cfg.DATALOADER.TEST_BATCH_SIZE = 4

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def test_one(trainer, weights_path, args):
    trainer.checkpointer.load(weights_path) # checkpointables is None in order to use default checkpointables

    next_iter = trainer.iter + 1
    epoch = int(os.path.basename(weights_path).split('_')[2])

    with EventStorage(next_iter) as trainer.storage:

        if trainer.val_data_loader is not None:
            for hook in trainer._hooks:
                if isinstance(hook, EvalHook) and hook._stage == 'val_ema':
                    hook._do_eval(epoch=epoch)

        if trainer.test_data_loader is not None:
            for hook in trainer._hooks:
                if isinstance(hook, EvalHook) and hook._stage == 'test_ema':
                    hook._do_eval(epoch=epoch)

def main(args):
    cfg = setup(args)
    trainer = build_engine(cfg)

    weights_path = cfg.MODEL.WEIGHTS
    if os.path.isdir(weights_path):
        weights_paths = glob.glob(os.path.join(weights_path, "*.pth"))
        weights_paths = sorted(weights_paths)
        for weights_path in weights_paths:
            test_one(trainer, weights_path, args)

    elif os.path.isfile(weights_path): 
        test_one(trainer, weights_path, args)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )