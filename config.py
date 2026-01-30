import torch
import argparse

# vpt tuning
def vpt_option():
    # init
    parser = argparse.ArgumentParser('Visual Prompt Tuning')
    parser.add_argument('--dataset', type=str, default='fgvc_cub', help='datasets: cifar10, cifar100 or svhn')
    parser.add_argument('--task_classes', type=int, default=200, help='task_classes: 10 100 or 10')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--batch_size_train', type=int, default=64, help='train batch_size')
    parser.add_argument('--batch_size_test', type=int, default=64, help='test batch_size')
    parser.add_argument('--vpt_num', type=int, default=10, help='vpt prompt number')
    parser.add_argument('--vpt_type', type=str, default='Deep', help='vpt type: Deep or Shallow')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--patch_size', type=int, default=32, help='pretrained vit model patch size')
    parser.add_argument('--use_scheduler', default=True, action="store_true",
                        help='use scheduler for train vpt')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--ddp', default=False,
                        action="store_true",
                        help='use ddp for training')
    args = parser.parse_args()
    return args


# visual prompt
def visual_prompt_option():
    # init
    parser = argparse.ArgumentParser('Visual Prompting for Vision Models')
    parser.add_argument('--dataset', type=str, default='fgvc_cub', help='datasets: cifar10, cifar100 or svhn')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size_train', type=int, default=64, help='train batch_size')
    parser.add_argument('--batch_size_test', type=int, default=64, help='test batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    # optimization
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--patience', type=int, default=1000)
    # prompt
    parser.add_argument('--prompt_method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch', 'mypad1', 'mypad2', 'mypad3'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--patch_size', type=int, default=32, help='pretrained vit model patch size')
    parser.add_argument('--model_dir', type=str, default='./save/models', 
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images', 
                        help='path to save images')
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for initializing training')
    parser.add_argument('--ddp', default=False,
                        action="store_true",
                        help='use ddp for training')
    args = parser.parse_args()
    return args


# prompt CDF option
def prompt_cdf_option():
    # init
    parser = argparse.ArgumentParser('Visual Prompting with Context-aware Diffusion Features')
    parser.add_argument('--dataset', type=str, default='fgvc_cub', help='datasets')
    parser.add_argument('--task_classes', type=int, default=200, help='task_classes: 10 100 or 10')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--batch_size_train', type=int, default=64, help='train batch_size')
    parser.add_argument('--batch_size_test', type=int, default=64, help='test batch_size')
    parser.add_argument('--vpt_num', type=int, default=1, help='vpt prompt number')
    parser.add_argument('--vpt_type', type=str, default='Shallow', help='vpt type: Deep or Shallow')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--patch_size', type=int, default=32, help='pretrained vit model patch size')
    parser.add_argument('--use_scheduler', default=True, action="store_true",
                        help='use scheduler for train vpt') 
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--prompt_method', type=str, default='mypad1',
                        choices=['padding', 'random_patch', 'fixed_patch', 'center_patch', 'full_patch', 'mypad1', 'mypad2', 'mypad3'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30, 
                        help='size for visual prompts')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000, help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--temp', type=float, default=1.0, help='temperature for KL loss')
    parser.add_argument('--dataset_title', type=str, default='bird', help='title of each dataset')
    # DDP
    parser.add_argument('--ddp', default=False,
                        action="store_true",
                        help='use ddp for training')
    # parser.add_argument("--local_rank", type=int, default=-1)
    # parser.add_argument("--gpu_num", type=int, default=2) 
    args = parser.parse_args()
    return args