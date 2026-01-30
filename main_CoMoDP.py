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
import torch.nn.functional as F
from models import visual_prompt
from models.procdf_tuning import build_prosdg_model
from config import prompt_cdf_option
from utils.dataset import get_dataset
from utils.util import AverageMeterSDF, setup_seed, refine_classname, cosine_lr, ProgressMeter, accuracy
from utils.logger import Logger
import torch.backends.cudnn as cudnn
import timm, copy


_logger = Logger('train_procdf.log', logging.ERROR, logging.DEBUG)
model_dir = (Path(__file__).parent / "models").resolve()
if str(model_dir) not in sys.path: sys.path.insert(0, str(model_dir))
utils_dir = (Path(__file__).parent / "utils").resolve()
if str(utils_dir) not in sys.path: sys.path.insert(0, str(utils_dir))


def train_epoch_1step(model, basic_model, dataloader, prompter, prompter_opt, prompter_sch, 
                      procdf_opt, criterion, text_names, epoch, args):
    batch_time = AverageMeterSDF('Time', ':6.3f')
    data_time = AverageMeterSDF('Data', ':6.3f')
    losses = AverageMeterSDF('Loss', ':.4e')
    top1 = AverageMeterSDF('Acc@1', ':6.2f')
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses, top1], prefix="Epoch: [{}]".format(epoch))

    model.train()
    prompter.train()
    num_batches_per_epoch = len(dataloader)
    end = time.time()

    vm_params = copy.deepcopy(model.head.state_dict())
    vm_head = copy.deepcopy(model.head)
    for idx, (images, labels) in enumerate(dataloader):
        text_prompts = []
        for label in labels:
            text_prompts.append(f'a photo of a {text_names[int(label)]}')
        # measure data loading time
        data_time.update(time.time() - end)
        # adjust learning rate
        step = num_batches_per_epoch * epoch + idx
        prompter_sch(step)
        # compute losses
        images, labels = images.to(args.device), labels.to(args.device)
        prompt_images = prompter(images)
        sdf_fea, sdf_out = model(prompt_images, text_prompts) # sdf_fea.shape = [b，768]
        sdf_fea = sdf_fea.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            org_fea = basic_model.forward_features(images)[:, 0, :]
            for name_param in vm_params:
                vm_params[name_param] = model.head.state_dict()[name_param] * 0.99 + vm_params[name_param] * (1.0-0.99)
            vm_head.load_state_dict(vm_params)
            org_out = vm_head(model.fc_norm(org_fea))
            org_fea = org_fea.norm(dim=-1, keepdim=True)
        loss_ce = criterion(sdf_out, labels)
        loss_sim = F.l1_loss(sdf_fea, org_fea, reduction='mean')
        loss_dis = F.kl_div(F.log_softmax(sdf_out, dim=1), F.log_softmax(org_out, dim=1), reduction='sum', log_target=True) / sdf_out.numel()
        loss = loss_ce + loss_sim + loss_dis
        # compute gradient and do SGD step
        procdf_opt.zero_grad()
        prompter_opt.zero_grad()
        loss.backward()
        procdf_opt.step()
        prompter_opt.step()
        # measure accuracy
        acc_prompt = accuracy(sdf_out, labels, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc_prompt[0].item(), images.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (idx % args.print_freq == 0):
            progress.display(idx)
    return losses.avg, top1.avg


def test_epoch_1step(model, test_loader, prompter, criterion, text_names, args):
    batch_time = AverageMeterSDF('Time', ':6.3f')
    losses = AverageMeterSDF('Loss', ':.4e')
    top1_sdf = AverageMeterSDF('ProCDF Acc@1', ':6.2f')
    progress = ProgressMeter(len(test_loader), [batch_time, losses, top1_sdf], prefix='Validate: ')
    
    model.eval()
    prompter.eval()
    total_loss = 0
    with torch.no_grad():
        end = time.time()
        num_corrects = 0
        num_samples = 0
        for idx, (images, labels) in enumerate(test_loader):
            num_samples += images.size(0)
            text_prompts = []
            for label in labels:
                text_prompts.append(f'a photo of a {args.dataset_title}')
            images, labels = images.to(args.device), labels.to(args.device)
            prompt_images = prompter(images)
            _, output = model(prompt_images, text_prompts)
            loss = criterion(output, labels)
            total_loss += loss.item()
            # test acc
            _, predicts = torch.max(output, -1)
            num_corrects += sum(torch.eq(predicts.cpu(), labels.cpu())).item()
            # measure accuracy and record loss
            acc_sdf = accuracy(output, labels, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_sdf.update(acc_sdf[0].item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # show information
            if (idx % args.print_freq == 0): 
                progress.display(idx)
        t_acc = num_corrects / num_samples

        print('ProCDF Validate [{step}/{steps}]: ProCDF Test Loss: {loss:.3f} | ProCDF Test Acc: {acc:.3f}% | {t_acc:.3f}% [{num}/{nums}]'.format(
            step=idx+1, steps=len(test_loader), loss=total_loss/(idx+1), acc=top1_sdf.avg, t_acc=t_acc*100.0, num=num_corrects, nums=num_samples))
    return t_acc*100.0


def ProCDF_1step():
    args = prompt_cdf_option()
    setup_seed(args.seed)
    # 1.load dataset
    all_dataset = get_dataset(args)
    train_loader, test_loader, testset_num, class_names = getattr(all_dataset, 'get_{}_dataset'.format(args.dataset), 'dataset is none!')()
    text_names = class_names
    # 2.define unified visual prompt
    prompter = visual_prompt.__dict__[args.prompt_method](args).to(args.device)
    # 3.define basic model
    vis_model = timm.create_model('vit_base_patch' + str(32) + '_' + str(args.image_size), pretrained=True)
    for param in vis_model.parameters(): 
        param.requires_grad = False
    vis_model = vis_model.to(args.device)
    vis_model.eval()
    # 4.define our model
    procdf_model = build_prosdg_model(num_classes=args.task_classes, img_size=args.image_size, 
                                      model_idx='ViT', patch_size=args.patch_size,
                                      Prompt_Token_num=args.vpt_num, VPT_type=args.vpt_type, dift_input_dim=1280, args=args).to(args.device)
    # 5.define training
    prompter_opt = optim.SGD(prompter.parameters(), lr=40.0, momentum=0.9, weight_decay=0)
    procdf_params = [param for param in procdf_model.parameters() if param.requires_grad]
    procdf_opt = optim.Adam(params=procdf_params, lr=1e-3, weight_decay=0)
    criterion = nn.CrossEntropyLoss().to(args.device)
    total_steps = len(train_loader) * args.epochs
    prompter_sch = cosine_lr(prompter_opt, args.learning_rate, args.warmup, total_steps)
    procdf_sch = optim.lr_scheduler.CosineAnnealingLR(procdf_opt, T_max=args.epochs)
    # 6.define classes indices
    class_names = refine_classname(class_names)
    indices = list(range(len(class_names)))


    if args.evaluate:
        print("=> Evaluate ProCDF ******")
        procdf_prompter_path = os.path.join('./save/', 'fgvc_flower_prompter_padding.pt')
        print("=> loading ProCDF prompter checkpoint '{}'".format(procdf_prompter_path))
        prompter.load_state_dict(torch.load(procdf_prompter_path))
        prompter.to(args.device)
        procdf_meta_path = os.path.join('./save/', 'fgvc_flower_meta_padding.pt')
        print("=> loading ProCDF meta checkpoint '{}'".format(procdf_meta_path))
        procdf_model.load_state_dict(torch.load(procdf_meta_path))
        procdf_model.to(args.device)
        eval_acc = test_epoch_1step(procdf_model, test_loader, prompter, criterion, text_names, args)
        print('=> Evaluate ProCDF acc {eval_acc}'.format(eval_acc=eval_acc))
    else:
        print("=> Training ProCDF ******")
        # pre_acc = test_model(model, test_loader, indices, criterion, args)
        # print('=> previous PromptSDF acc {promptsdf_acc}'.format(promptsdf_acc=pre_acc))
        best_acc = 0
        best_epoch = 0
        test_acc_list = []

        for epoch in tqdm(range(args.epochs), desc='PromptCDF-training'):
            train_loss, train_acc = train_epoch_1step(procdf_model, vis_model, train_loader, prompter, prompter_opt, 
                                                prompter_sch, procdf_opt, criterion, text_names, epoch, args)
            procdf_sch.step()
            test_acc = test_epoch_1step(procdf_model, test_loader, prompter, criterion, text_names, args)
            test_acc_list.append(round(test_acc, 3))

            print('ProCDF Validate [{epoch}/{epochs}]: train Loss: {loss:.3f} | train Acc: {train_acc:.3f}% | test Acc: {test_acc:.3f}%'.format(
                    epoch=epoch, epochs=args.epochs, loss=train_loss, train_acc=train_acc, test_acc=test_acc))
        
            if test_acc > best_acc:
                best_acc = round(test_acc, 3)
                best_epoch = epoch
                # save result
                state = {'best_epoch':best_epoch, 'best_acc':best_acc, 'test_acc_list':test_acc_list}
                json.dump(state, open(os.path.join('./save/', f'{args.dataset}_{args.prompt_method}_result.json'), 'w'), indent=4) 
                # save model
                torch.save(prompter.state_dict(), os.path.join('./save/', f'{args.dataset}_prompter_{args.prompt_method}.pt'))
                torch.save(procdf_model.state_dict(), os.path.join('./save/', f'{args.dataset}_meta_{args.prompt_method}.pt'))
            else:
                # save result
                state = {'best_epoch':best_epoch, 'best_acc':best_acc, 'test_acc_list':test_acc_list}
                json.dump(state, open(os.path.join('./save/', f'{args.dataset}_{args.prompt_method}_result.json'), 'w'), indent=4) 

            # if ((epoch+1) % 10 == 0):
            print('ProCDF Validate [{epoch}/{epochs}] ProCDF Test Acc List: {acc_list}'.format(
                epoch=epoch, epochs=args.epochs, acc_list=test_acc_list)) 
        print("=> ProCDF Test Acc List: ", test_acc_list, " | ProCDF Max Test Acc: ", max(test_acc_list))


if __name__=='__main__':
    cudnn.benchmark = True
    ProCDF_1step()
    # python3 main_procdf.py --dataset fgvc_flower --task_classes 102 --device 'cuda:0'