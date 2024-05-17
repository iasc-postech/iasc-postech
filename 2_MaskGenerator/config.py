import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


def config_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default_no_attn_feature_condition_false.yaml", help="path to config file")
    parser.add_argument(
            "--train",  
            type=str2bool,
            nargs="?",
            const=True, 
            default=False, 
            help="run with train mode"
        ) 

    parser.add_argument(
            "--test",  
            type=str2bool,
            nargs="?",
            const=True, 
            default=False, 
            help="run with test mode"
        )    
        
    parser.add_argument(
            "--debug", 
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="run with debug mode"
        )
    parser.add_argument(
            "--basedir", type=str, default="./logs",
            help="where to store ckpts and logs"
        )
    parser.add_argument(
            "--i_print", type=int, default=50,
            help="frequency of console printout and metric logging",
        )
    parser.add_argument(
            "--seed", type=int, default=0, help="seed to fix"
        )
    parser.add_argument(
            "--coco_path", type=str, default="../coco",
        )
    parser.add_argument(
            "--modified_coco_path", type=str, default="../modified_coco",
        )

    return parser.parse_known_args()[0], parser