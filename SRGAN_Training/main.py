from mode import *
import argparse


parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("true")

# Default configuration & parsing command line
parser.add_argument("--LR_path", type=str, default="Dataset/train/LowRes")
parser.add_argument("--GT_path", type=str, default="Dataset/train/HighRes")
parser.add_argument("--res_num", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--L2_coeff", type=float, default=1.0)
parser.add_argument("--adv_coeff", type=float, default=1e-3)
parser.add_argument("--tv_loss_coeff", type=float, default=0.0)
parser.add_argument("--pre_train_epoch", type=int, default=4000)
parser.add_argument("--fine_train_epoch", type=int, default=0)
parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--patch_size", type=int, default=24)
parser.add_argument("--feat_layer", type=str, default="relu5_4")
parser.add_argument("--vgg_rescale_coeff", type=float, default=0.006)
parser.add_argument("--fine_tuning", type=str2bool, default=False)
parser.add_argument("--in_memory", type=str2bool, default=True)
parser.add_argument("--generator_path", type=str)
parser.add_argument("--mode", type=str, default="train")

args = parser.parse_args()

# Executing the corresponding mode based on the argument
if args.mode == "train":
    train(args)

elif args.mode == "valid":
    valid(args)

elif args.mode == "test":
    test(args)
