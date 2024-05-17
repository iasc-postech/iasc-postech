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
    parser.add_argument("--config", type=str, default="configs/type_1_abs4c_no_classifier_deep.yaml", help="path to config file")
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
    # parser.add_argument(
    #         "--num_classes", type=int, default=20, help="class num"
    #     )
    
    # parser.add_argument(
    #         "--lr_init", type=float, default=1.0e-5, help="transformer opt"
    #     )
    # parser.add_argument(
    #         "--enc_layers", type=int, default=2, help="transformer opt"
    #     )
    # parser.add_argument(
    #         "--nhead", type=int, default=8, help="transformer opt"
    #     )
    # parser.add_argument(
    #         "--optimizer", type=str, default='adam', help="transformer opt"
    #     )
    # parser.add_argument(
    #         "--scheduler", type=str, default='lwca', help="transformer opt"
    #     )
    # parser.add_argument(
    #         "--pre_norm", type=bool, default=True, help="transformer opt"
    #     )
    # parser.add_argument(
    #         "--dropout", type=float, default='0.1', help="transformer opt"
    #     )

    return parser.parse_known_args()[0], parser