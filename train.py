import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from evaluate import validation  # renamed test.py to evaluate.py to avoid circular import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    print('Filtering is turned off, using full dataset')
    opt.select_data = ['']  # use all LMDB dataset
    opt.batch_ratio = ['1']
    opt.data_filtering_off = True

    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt, select_data=[''])
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,  # safer for Colab
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverterForBaiduWarpctc(opt.character) if opt.baiduCTC else CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception:
            if 'weight' in name:
                param.data.fill_(1)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.train()

    if opt.saved_model != '':
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    criterion = (torch.nn.CTCLoss(zero_infinity=True).to(device) if 'CTC' in opt.Prediction and not opt.baiduCTC 
                 else torch.nn.CrossEntropyLoss(ignore_index=0).to(device))

    if 'CTC' in opt.Prediction and opt.baiduCTC:
        from warpctc_pytorch import CTCLoss 
        criterion = CTCLoss()

    loss_avg = Averager()

    filtered_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = (optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999)) if opt.adam 
                 else optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps))

    start_iter = 0
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    start_time = time.time()

    while iteration < opt.num_iter:
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)
        else:
            text_input = text[:, :-1].to(device)
            target = text[:, 1:].to(device)
            preds = model(image, text_input)
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        loss_avg.add(cost)

        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')

                print(loss_log)

        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--valid_data', required=True)
    parser.add_argument('--manualSeed', type=int, default=1111)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_iter', type=int, default=300000)
    parser.add_argument('--valInterval', type=int, default=2000)
    parser.add_argument('--saved_model', default='')
    parser.add_argument('--FT', action='store_true')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--rho', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--baiduCTC', action='store_true')
    parser.add_argument('--select_data', type=str, default='')
    parser.add_argument('--batch_ratio', type=str, default='1')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0')
    parser.add_argument('--batch_max_length', type=int, default=12)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--character', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    parser.add_argument('--sensitive', action='store_true')
    parser.add_argument('--PAD', action='store_true')
    parser.add_argument('--data_filtering_off', action='store_true')
    parser.add_argument('--Transformation', type=str, required=True)
    parser.add_argument('--FeatureExtraction', type=str, required=True)
    parser.add_argument('--SequenceModeling', type=str, required=True)
    parser.add_argument('--Prediction', type=str, required=True)
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=256)

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}-Seed{opt.manualSeed}'

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    if opt.sensitive:
        opt.character = string.printable[:-6]

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    train(opt)
