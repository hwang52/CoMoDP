import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from pathlib import Path
import sys
import time, logging
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
from models.vpt_prompt import build_vpt_promptmodel
from models import visual_prompt
from config import visual_prompt_option
from utils.dataset import get_dataset
from utils.util import AverageMeterVisual, setup_seed, refine_classname, cosine_lr, accuracy, ProgressMeter
from utils.logger import Logger
import torch.backends.cudnn as cudnn


model_dir = (Path(__file__).parent / "models").resolve()
if str(model_dir) not in sys.path: sys.path.insert(0, str(model_dir))
utils_dir = (Path(__file__).parent / "utils").resolve()
if str(utils_dir) not in sys.path: sys.path.insert(0, str(utils_dir))


def train_epoch(indices, dataloader, model, prompter, optimizer, scheduler, criterion, epoch, args):
    batch_time = AverageMeterVisual('Time', ':6.3f')
    data_time = AverageMeterVisual('Data', ':6.3f')
    losses = AverageMeterVisual('Loss', ':.4e')
    top1 = AverageMeterVisual('Acc@1', ':6.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    prompter.train()
    num_batches_per_epoch = len(dataloader)
    end = time.time()
    for idx, (images, labels) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        # adjust learning rate
        step = num_batches_per_epoch * epoch + idx
        scheduler(step)
        # training
        images, labels = images.to(args.device), labels.to(args.device)
        prompted_images = prompter(images)
        prompt_output = model(prompted_images)
        if indices: output = prompt_output[:, indices]
        loss = criterion(output, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure accuracy
        acc_prompt = accuracy(output, labels, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc_prompt[0].item(), images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)
            
    return losses.avg, top1.avg


def test_model(model, dataloader, indices, prompter, criterion, args):
    batch_time = AverageMeterVisual('Time', ':6.3f')
    losses = AverageMeterVisual('Loss', ':.4e')
    top1_org = AverageMeterVisual('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeterVisual('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, losses, top1_org, top1_prompt], prefix='Validate: ')
    
    prompter.eval()
    total_loss = 0
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            prompted_images = prompter(images)
            output_prompt = model(prompted_images)
            output_org = model(images)
            if indices:
                output_prompt = output_prompt[:, indices]
                output_org = output_org[:, indices]
            loss = criterion(output_prompt, labels)
            # test loss info
            total_loss += loss.item()
            # measure accuracy and record loss
            acc_org = accuracy(output_org, labels, topk=(1,))
            acc_prompt = accuracy(output_prompt, labels, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_org.update(acc_org[0].item(), images.size(0))
            top1_prompt.update(acc_prompt[0].item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # show
            if idx % args.print_freq == 0: progress.display(idx)

        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))
        
    return top1_prompt.avg, top1_org.avg


if __name__=="__main__":
    args = visual_prompt_option()
    setup_seed(args.seed)
    all_dataset = get_dataset(args)
    train_loader, test_loader, testset_num, class_names = getattr(all_dataset, 'get_{}_dataset'.format(args.dataset), 'dataset is none!')()
    # define pretrained model
    import timm
    from pprint import pprint
    model_names = timm.list_models('*vit*')
    model = timm.create_model('vit_base_patch' + str(32) + '_' + str(args.image_size), pretrained=True)
    model = model.to(args.device)
    model.eval()
    # visual prompt
    prompter = visual_prompt.__dict__[args.prompt_method](args).to(args.device)
    # define criterion and optimizer
    optimizer = torch.optim.SGD(prompter.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    # define classes indices
    class_names = refine_classname(class_names)
    indices = list(range(len(class_names)))
    # define scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)
    cudnn.benchmark = True


    if args.evaluate:
        print("=> evaluate visual prompt")
        prompter_path = os.path.join('./save/', 'fgvc_cub_vp_result.pt')
        print("=> loading checkpoint '{}'".format(prompter_path))
        prompter.load_state_dict(torch.load(prompter_path))
        prompter.to(args.device)
        acc_prompt_avg, acc_org_avg = test_model(model, test_loader, indices, prompter, criterion, args)
        print('---------------- evaluate prompt image acc {prompt_acc}, evaluate org image acc {org_acc}'
              .format(prompt_acc=acc_prompt_avg, org_acc=acc_org_avg))
    else:
        print("=> training visual prompt")
        epochs_since_improvement = 0
        best_acc = 0
        best_epoch = 0
        test_acc_list = []

        for epoch in tqdm(range(args.epochs), desc='Visual Prompt-training'):
            # train for single epoch
            train_loss, train_acc = train_epoch(indices, train_loader, model, prompter, optimizer, scheduler, criterion, epoch, args)
            # evaluate on validation set
            test_prompt_acc, test_org_acc = test_model(model, test_loader, indices, prompter, criterion, args)
            test_acc_list.append(test_prompt_acc)
            # remember best acc@1 and save checkpoint
            is_best = test_prompt_acc > best_acc
            best_acc = max(test_prompt_acc, best_acc)

            if is_best:
                # save result
                best_epoch = epoch
                state = {'best_epoch':best_epoch, 'best_acc':best_acc, 'test_acc_list':test_acc_list}
                json.dump(state, open(os.path.join('./save/', f'{args.dataset}_vp_result.json'), 'w'), indent=4)
                # save model
                torch.save(prompter.state_dict(), os.path.join('./save/', f'{args.dataset}_vp_result.pt'))
            else:
                state = {'best_epoch':best_epoch, 'best_acc':best_acc, 'test_acc_list':test_acc_list}
                json.dump(state, open(os.path.join('./save/', f'{args.dataset}_vp_result.json'), 'w'), indent=4)
            if is_best:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                print(f"There's no improvement for {epochs_since_improvement} epochs.")
                if epochs_since_improvement >= args.patience:
                    print("The training halted by early stopping criterion.")
                    break
            
            print('\nepoch [{epoch}/{epoches}], train_acc {train_acc:.4f}, train_loss {train_loss:.3f}, \
                test_prompt_acc {test_prompt_acc:.4f}, test_org_acc {test_org_acc:.4f}'.format(
                epoch=epoch+1, epoches=args.epochs, train_acc=train_acc, train_loss=train_loss, 
                test_prompt_acc=test_prompt_acc, test_org_acc=test_org_acc
            ))
        
        print("=> Visual Prompting test accuracy: ", test_acc_list, " | Visual Prompting max accuracy: ", max(test_acc_list))
        # python3 main_visual.py --dataset fgvc_cub --device 'cuda:0'