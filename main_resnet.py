import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from pathlib import Path
import sys
import time, logging
from tqdm import tqdm
from collections import OrderedDict
from config import vpt_option
from utils.dataset import get_dataset
from utils.util import AverageMeterVPT, setup_seed
import torchvision.models as models


model_dir = (Path(__file__).parent / "models").resolve()
if str(model_dir) not in sys.path: sys.path.insert(0, str(model_dir))
utils_dir = (Path(__file__).parent / "utils").resolve()
if str(utils_dir) not in sys.path: sys.path.insert(0, str(utils_dir))


def train_epoch(model, dataloader, criterion, optimizer, device):
    batch_time_m = AverageMeterVPT()
    data_time_m = AverageMeterVPT()
    acc_m = AverageMeterVPT()
    losses_m = AverageMeterVPT()
    end = time.time()

    model.train()
    for idx, data_batch in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        # training
        images, labels = data_batch
        optimizer.zero_grad()
        preds = model(images.to(device))
        # update
        loss = criterion(preds, labels.to(device))
        loss.backward()
        optimizer.step()
        losses_m.update(loss.item())
        # train acc 
        acc_m.update(labels.to(device).eq(preds.argmax(dim=1)).sum().item()/labels.size(0), n=labels.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()
    return OrderedDict([('train_acc', acc_m.avg), ('train_loss', losses_m.avg)])


def test_model(model, dataloader, device):
    total_loss = 0
    num_corrects = 0
    num_total = 0

    model.eval()
    with torch.no_grad():
        for idx, data_batch in enumerate(dataloader):
            images, labels = data_batch
            outputs = model(images.to(device))
            loss = nn.CrossEntropyLoss()(outputs, labels.to(device))
            total_loss += loss.item()
            _, predicts = torch.max(outputs, -1)
            num_corrects += sum(torch.eq(predicts.cpu(), labels.cpu())).item()
            num_total += labels.size(0)
        accuracy = num_corrects / num_total
    return accuracy


if __name__=="__main__":
    args = vpt_option()
    setup_seed(args.seed)
    all_dataset = get_dataset(args)
    train_loader, test_loader, testset_num, class_names = getattr(all_dataset, 'get_{}_dataset'.format(args.dataset), 'dataset is none!')()
    # model
    model = models.__dict__['resnet50'](pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.task_classes)
    model = model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss().to(args.device)

    # total_params = sum(p.numel() for p in model.parameters())
    # print("total: ", total_params)

    if args.evaluate:
        print("=> evaluate ResNet full tuning")
        vpt_prompt_path = os.path.join('./save/', 'fgvc_cub_resnet_result.pt')
        print("=> loading checkpoint '{}'".format(vpt_prompt_path))
        checkpoint_dict = torch.load(vpt_prompt_path)
        model.load_state_dict(checkpoint_dict)
        model.to(args.device)
        evaluate_acc = test_model(model, test_loader, args.device)
        print('---------------- evaluate ResNet acc {vpt_acc}'.format(vpt_acc=evaluate_acc))
    else:
        print("=> training ResNet full tuning")
        # previous_acc = test_model(model, test_loader, args.device)
        # print('---------------- previous acc {acc}'.format(acc=previous_acc))

        best_acc = 0
        step = 0
        best_epoch = 0
        test_acc_list = []
        for epoch in tqdm(range(args.epochs), desc='ResNet-training'):
            # train acc + train losses
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device)
            # test acc
            test_acc = test_model(model, test_loader, args.device)
            test_acc_list.append(test_acc)
            if scheduler: scheduler.step()
            # checkpoint
            step += 1
            if best_acc < test_acc:
                # save result
                best_epoch = epoch
                best_acc = test_acc
                state = {'best_epoch':best_epoch, 'best_acc':best_acc, 'test_acc_list':test_acc_list}
                json.dump(state, open(os.path.join('./save/', f'{args.dataset}_resnet_result.json'), 'w'), indent=4)
                # save model
                torch.save(model.state_dict(), os.path.join('./save/', f'{args.dataset}_resnet_result.pt'))
            else:
                # save result
                state = {'best_epoch':best_epoch, 'best_acc':best_acc, 'test_acc_list':test_acc_list}
                json.dump(state, open(os.path.join('./save/', f'{args.dataset}_resnet_result.json'), 'w'), indent=4)

            print('\nepoch [{epoch}/{epoches}], train_acc {train_acc:.4f}, train_loss {train_loss:.3f}, test_acc {test_acc:.4f}'.format(
                epoch=epoch+1, epoches=args.epochs, train_acc=train_metrics['train_acc'], 
                train_loss=train_metrics['train_loss'], test_acc=test_acc
            ))
        # resnet_acc = test_model(model, test_loader, args.device)
        # print('---------------- after resnet acc {acc}'.format(acc=resnet_acc))
        print("=> ResNet test accuracy: ", test_acc_list, " | ResNet max accuracy: ", max(test_acc_list))
        # python3 main_resnet.py --dataset fgvc_cub --task_classes 200 --device 'cuda:0'