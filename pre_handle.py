import argparse
import os
from data.datasets import get_dataset
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='refall',
        required=True, 
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--dataset_script",
        type=str,
        default=None,
        help=(
            "Dataset script path to custom one dataset"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=['An astronaut is riding a horse.'],
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1337, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token", 
        type=str, 
        default=None, 
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--noise_offset", 
        type=float, 
        default=0, 
        help="The scale of noise offset."
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default='', 
        help="Run name"
    )    
    # DiffusionITM
    parser.add_argument(
        '--neg_prob', 
        type=float, 
        default=1.0, 
        help='The probability of sampling a negative image.'
    )
    parser.add_argument(
        '--img_root', 
        type=str, 
        default='../dataset/coco/images'
    )
    parser.add_argument(
        '--hard_neg', 
        action='store_true'
    )
    parser.add_argument(
        '--relativistic', 
        action='store_true'
    )
    parser.add_argument(
        '--unhinged', 
        action='store_true'
    )
    parser.add_argument(
        '--neg_img', 
        action='store_true'
    )
    parser.add_argument(
        '--mixed_neg', 
        action='store_true'
    )
    # DiffQformer
    # - Backbone
    parser.add_argument(
        '--position_embedding', 
        default='sine', 
        type=str, 
        choices=('sine', 'learned'),
        help="Type of positional embedding to use on top of the image features"
    )
    # - Transformer
    parser.add_argument(
        '--enc_layers', 
        default=1, 
        type=int, 
        help="Number of encoding layers in the transformer"
    )
    parser.add_argument(
        '--dec_layers', 
        default=1, 
        type=int, 
        help="Number of decoding layers in the transformer"
    )
    parser.add_argument(
        '--dim_feedforward', 
        default=2048, 
        type=int, 
        help="Intermediate size of the feedforward layers in the transformer blocks"
    )
    parser.add_argument(
        '--hidden_dim', 
        default=256, 
        type=int,                        
        help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument(
        '--dropout', 
        default=0.1, 
        type=float,                        
        help="Dropout applied in the transformer"
    )
    parser.add_argument(
        '--nheads', 
        default=8, 
        type=int, 
        help="Number of attention heads inside the transformer's attentions"
    )
    parser.add_argument(
        '--num_queries_matching', 
        default=10, 
        type=int
    )
    parser.add_argument(
        '--num_queries_rec', 
        default=100, 
        type=int
    )
    parser.add_argument(
        '--pre_norm', 
        action='store_true'
    )
    parser.add_argument(
        '--fix_timestep', 
        default=None, 
        type=int, 
        help="use one timestep to train the model"
    )
    parser.add_argument(
        '--transformer_decoder_only', 
        action='store_true'
    )
    parser.add_argument(
        '--reset_optimizer', 
        action='store_true'
    )
    # rec
    parser.add_argument(
        "--combine_datasets_val", 
        nargs="+", 
        help="List of datasets to combine for eval", 
        default=['refcoco', 'refcoco+', 'refcocog']
    )
    parser.add_argument(
        "--no_freeze_text_encoder", 
        dest="freeze_text_encoder", 
        action="store_false", 
        help="Whether to freeze the weights of the text encoder"
    )
    parser.add_argument(
        '--text_encoder_type', 
        type=str, 
        default='roberta-base'
    )
    parser.add_argument(
        "--no_contrastive_align_loss", 
        dest="contrastive_align_loss", 
        action="store_false",
        help="Whether to add contrastive alignment loss"
    )
    parser.add_argument(
        "--contrastive_loss_hdim", 
        type=int, 
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss"
    )
    parser.add_argument(
        "--masks", 
        action="store_true"
    ) 
    # for segmentation
    parser.add_argument(
        "--no_detection", 
        action="store_true", 
        help="Whether to train the detector"
    )
    parser.add_argument(
        "--val_fullset", 
        dest='val_subset', 
        action="store_false"
    )
    parser.add_argument(
        "--dataset_name_val", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "--dataset_name_matching", 
        type=str, 
        default='mscoco_hard_negative'
    )
    # Criterion
    parser.add_argument(
        "--set_loss", 
        default="hungarian", 
        type=str, 
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss"
    )
    parser.add_argument(
        "--temperature_NCE", 
        type=float, 
        default=0.07, 
        help="Temperature in the  temperature-scaled cross entropy loss"
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class", 
        default=1, 
        type=float,
        help="Class coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_bbox", 
        default=5, 
        type=float,
        help="L1 box coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_giou", 
        default=2, 
        type=float,
        help="giou box coefficient in the matching cost"
    )
    # Loss coefficients
    parser.add_argument(
        "--ce_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--bbox_loss_coef", 
        default=5, 
        type=float
    )
    parser.add_argument(
        "--giou_loss_coef", 
        default=2, 
        type=float
    )
    parser.add_argument(
        "--contrastive_align_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--contrastive_i2t_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--contrastive_t2i_loss_coef", 
        default=1, 
        type=float
    )
    parser.add_argument(
        "--eos_coef", 
        default=0.1, 
        type=float,
        help="Relative classification weight of the no-object class"
    )
    parser.add_argument(
        "--unet_feature", 
        type=str, 
        required=True, 
        default=None
    )
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    # 替换为直接设置参数
    args = argparse.Namespace()
    
    # 设置命令行中指定的所有参数
    args.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    args.enable_xformers_memory_efficient_attention = True
    args.gradient_checkpointing = True
    args.unet_feature = "up1"
    args.dataloader_num_workers = 4
    args.center_crop = True
    args.random_flip = True
    args.lr_scheduler = "constant"
    args.checkpointing_steps = 500
    args.train_batch_size = 48
    args.val_batch_size = 16
    args.gradient_accumulation_steps = 1
    args.max_train_steps = 60000
    args.learning_rate = 1e-4
    args.run_name = "dpt_stage1"
    args.report_to = "wandb"
    args.dataset_name = "refall"
    
    # 添加路径相关参数，确保指向正确的文件位置
    # 这些参数可能需要根据您的实际环境进行调整
    args.ref_root = "/absolute/path/to/coco/images"  # 替换为实际的绝对路径
    args.ref_ann_path = "/absolute/path/to/ReC/mdetr/OpenSource"  # 替换为实际的绝对路径
    
    # 如果您知道正确的dataset_dir路径，也可以直接设置
    args.dataset_dir = "/absolute/path/to/dataset/"  # 替换为实际的绝对路径
    
    # 添加resolution参数，可能在build_dataset_combined中使用
    args.resolution = 512
    
    # 添加text_encoder_type参数，用于tokenizer
    args.text_encoder_type = "roberta-base"
    
    # 添加revision参数，用于CLIPTokenizer
    args.revision = None
    
    # 添加num_queries相关参数
    args.num_queries_matching = 16
    args.num_queries_rec = 16
    
    # 添加contrastive相关参数
    args.contrastive_align_loss = True
    args.contrastive_loss_hdim = 64
    args.contrastive_align_loss_coef = 1.0
    args.contrastive_i2t_loss_coef = 1.0
    args.contrastive_t2i_loss_coef = 1.0
    
    # 添加bbox相关参数
    args.bbox_loss_coef = 5.0
    args.giou_loss_coef = 2.0
    args.ce_loss_coef = 1.0
    
    # 添加eos_coef参数
    args.eos_coef = 0.1
    
    # 添加temperature参数
    args.temperature_NCE = 0.07
    
    # 添加freeze_text_encoder参数
    args.freeze_text_encoder = True

    args.dataset_name = 'mscoco_hard_negative'
    args.img_root = '../dataset/coco/images'
    
    train_dataset = get_dataset(args.dataset_name, args.img_root, args=args, split='train', combined=True)
    print(train_dataset)


if __name__ == "__main__":
    main()