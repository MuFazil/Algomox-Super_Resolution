from mode import *
import argparse

# Get tuning parameters from NNI
tuner_params = nni.get_next_parameter()

# Extracting variable hyper parameters from the tuner_params
batch_size = tuner_params["batch_size"]
pre_train_epoch = tuner_params["pre_train_epoch"]
learning_rate = tuner_params["learning_rate"]

# Parsing command-line arguments
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("true")


# Default configuration
parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch Size")
parser.add_argument(
    "--pre_train_epoch", type=int, default=pre_train_epoch, help="Number of Epochs"
)
parser.add_argument(
    "--learning_rate", type=float, default=learning_rate, help="Learning Rate"
)
parser.add_argument("--LR_path", type=str, default="custom_dataset/train_LR")
parser.add_argument("--GT_path", type=str, default="custom_dataset/train_HR")
parser.add_argument("--res_num", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--L2_coeff", type=float, default=1.0)
parser.add_argument("--adv_coeff", type=float, default=1e-3)
parser.add_argument("--tv_loss_coeff", type=float, default=0.0)
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

elif args.mode == "test":
    test(args)

elif args.mode == "test_only":
    test_only(args)
