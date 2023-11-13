from mode import *
import argparse


parser = argparse.ArgumentParser()

# Get tuning parameters from NNI
tuner_params = nni.get_next_parameter()

# Extracting variable parameters from the tuner_params
num_block = tuner_params["num_block"]
num_discriminator_blocks = tuner_params["num_discriminator_blocks"]
conv_kernel_size = tuner_params["conv_kernel_size"]
generator_depth = tuner_params["generator_depth"]
discriminator_depth = tuner_params["discriminator_depth"]
generator_activation = tuner_params["generator_activation"]
discriminator_activation = tuner_params["discriminator_activation"]


def str2bool(v):
    return v.lower() in ("true")


parser.add_argument("--num_block", type=int, default=num_block, help="Number of blocks")
parser.add_argument(
    "--num_discriminator_blocks",
    type=int,
    default=num_discriminator_blocks,
    help="Number of Descriminator blocks",
)

parser.add_argument(
    "--conv_kernel_size", type=int, default=conv_kernel_size, help="conv_kernel_size"
)
parser.add_argument(
    "--generator_depth", type=int, default=generator_depth, help="generator_depth"
)
parser.add_argument(
    "--discriminator_depth",
    type=int,
    default=discriminator_depth,
    help="discriminator_depth",
)

parser.add_argument(
    "--generator_activation",
    type=str,
    default=generator_activation,
    help="Generator activation",
)
parser.add_argument(
    "--discriminator_activation",
    type=str,
    default=discriminator_activation,
    help="Discriminator activation",
)

# Default configuration
parser.add_argument("--LR_path", type=str, default="custom_dataset/train_LR")
parser.add_argument("--GT_path", type=str, default="custom_dataset/train_HR")
parser.add_argument("--res_num", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--L2_coeff", type=float, default=1.0)
parser.add_argument("--adv_coeff", type=float, default=1e-3)
parser.add_argument("--tv_loss_coeff", type=float, default=0.0)
parser.add_argument("--pre_train_epoch", type=int, default=100)
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
