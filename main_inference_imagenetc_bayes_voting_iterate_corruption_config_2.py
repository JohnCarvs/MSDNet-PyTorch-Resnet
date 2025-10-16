#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from sklearn.cluster import KMeans

import os
import sys
import math
import time
import shutil

import numpy as np
# import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict

np.set_printoptions(threshold=np.inf)

from dataloader import get_dataloaders
from args import arg_parser
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from tqdm import tqdm
import csv
import glob

# csv file writter
def get_new_csv_filename():
    existing = glob.glob("output_*.csv")
    nums = [int(f.split("_")[-1].split(".")[0]) for f in existing if f.split("_")[-1].split(".")[0].isdigit()]
    n = max(nums) + 1 if nums else 1
    return f"output_{n}.csv"
def log_result(function_called, exit_layer_number, corruption, severity, prec1, prec5):
    global csv_writer
    csv_writer.writerow([function_called, exit_layer_number, corruption, severity, prec1, prec5])
    csv_file.flush()  # write on disk imediatelly
csv_filename = get_new_csv_filename()
write_header = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
if write_header:
    csv_writer.writerow(["function_called", "exit_layer_number", "corruption", "severity", "prec1", "prec5"])


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

torch.manual_seed(args.seed)


def main():
    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del (model)

    model = getattr(models, args.arch)(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # if args.resume:
    checkpoint = load_checkpoint(args)
    if checkpoint is not None:
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        if 'best_prec1' in checkpoint:
            best_prec1 = checkpoint['best_prec1']
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(args)

    test_dict = {(c, s): l for c, s, l in test_loader} if isinstance(test_loader, list) else {}

    for corruption, severity, loader in val_loader:
        print(f"\n=== Processing {corruption} severity {severity} ===")
        test_loader_single = test_dict.get((corruption, severity), None)
        if test_loader_single is None:
            print(f"Warning: No matching test_loader for {corruption} severity {severity}")
            continue

        # 1. Dynamic evaluate (gera/usa logits)
        dynamic_evaluate(model, loader, loader, args, corruption=corruption, severity=severity)

        # 2. Ensemble
        validate_ensemble(test_loader_single, model, criterion, corruption, severity)

        # 3. Validação padrão
        validate(test_loader_single, model, criterion, corruption, severity)

        # 4. Matriz bayesiana simples
        bayes_matrix = np.zeros((7, args.num_classes, args.num_classes))
        bayes_matrix = bayes_matrix + 0.5
        bayes_matrix = validate_bayes_matrix(loader, model, criterion, bayes_matrix)
        validate_with_bayes_matrix(test_loader_single, model, criterion, bayes_matrix, corruption, severity)

        # 5. Matriz bayesiana com conformal prediction (v2)
        bayes_matrix = initialize_bayes_matrix(7, args.num_classes, 0.5)
        best_threshold = 0.8904  # ou calcule dinamicamente se quiser
        bayes_matrix = validate_bayes_matrix_with_conformal_prediction_fixing_target_not_most_v2(
            loader, model, criterion, bayes_matrix, threshold=0.95
        )
        validate_with_bayes_matrix_conformal_prediction(
            test_loader_single, model, criterion, bayes_matrix, threshold=0.95, corruption=corruption, severity=severity
        )
    print('********** Final prediction results **********')
    # validate(test_loader, model, criterion)

    return


def calculate_accuracy_and_proportion(conformal_set_counter, min_count=5):
    """
    计算Conformal set长度为1时每个输出层的平均准确率和每个Conformal set的准确率
    以及Conformal set长度为1的占比

    参数：
        conformal_set_counter: Conformal sets 计数器
        min_count: 过滤长度为2的conformal set的最小出现次数

    返回：
        average_accuracy: 每个输出层的平均准确率
        set_accuracy: 每个Conformal set的准确率
        proportion_length_1: Conformal set长度为1的占比
    """
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    set_accuracy = defaultdict(lambda: defaultdict(float))
    length_1_counts = defaultdict(int)
    total_predictions = defaultdict(int)
    length_1_accuracy = defaultdict(lambda: defaultdict(float))
    length_2_accuracy = defaultdict(lambda: defaultdict(float))

    for j, sets in conformal_set_counter.items():
        for conformal_set, targets in sets.items():
            if conformal_set == ('other',):
                continue
            total_count = sum(targets.values())
            total_predictions[j] += total_count

            if len(conformal_set) == 1:
                correct_count = targets[conformal_set[0]] if conformal_set[0] in targets else 0
                correct_counts[j] += correct_count
                total_counts[j] += total_count
                length_1_counts[j] += total_count
                length_1_accuracy[j][conformal_set] = correct_count / total_count if total_count > 0 else 0
            elif len(conformal_set) == 2 and total_count > min_count:
                correct_count = sum(targets[label] for label in conformal_set if label in targets)
                length_2_accuracy[j][conformal_set] = correct_count / total_count if total_count > 0 else 0

    # 计算每个输出层的平均准确率
    average_accuracy = {j: (correct_counts[j] / total_counts[j]) if total_counts[j] > 0 else 0
                        for j in correct_counts}

    # 计算Conformal set长度为1的占比
    proportion_length_1 = {j: (length_1_counts[j] / total_predictions[j]) if total_predictions[j] > 0 else 0
                           for j in length_1_counts}

    return average_accuracy, set_accuracy, proportion_length_1, length_1_accuracy, length_2_accuracy


def filter_and_display_conformal_sets(conformal_set_counter, average_accuracy, set_accuracy, min_count=5):
    """
    过滤并显示出现次数超过 min_count 次且Conformal set长度为2的具体标签分布情况
    以及每个输出层的平均准确率和每个Conformal set的准确率

    参数：
        conformal_set_counter: Conformal sets 计数器
        average_accuracy: 平均准确率
        set_accuracy: 每个Conformal set的准确率
        min_count: 最小出现次数
    """
    print("Average Accuracy for each layer when Conformal set length is 1:")
    for j, accuracy in average_accuracy.items():
        print(f"  Output Head {j}: {accuracy:.2f}")

    print("\nConformal Sets with more than min_count occurrences (length=2):")
    for j, sets in conformal_set_counter.items():
        for conformal_set, targets in sets.items():
            if len(conformal_set) == 2:
                total_count = sum(targets.values())
                if total_count > min_count:
                    print(f"\nOutput Head {j}: Conformal Set {conformal_set}")
                    for label, count in targets.items():
                        print(f"  Label {label}: {count} times")

    print("\nAccuracy for each Conformal set:")
    for j, sets in set_accuracy.items():
        for conformal_set, accuracy in sets.items():
            print(f"\nOutput Head {j}: Conformal Set {conformal_set} Accuracy: {accuracy:.2f}")




def filter_conformal_sets(conformal_set_counter, min_count=5):
    """
    过滤出出现次数超过 min_count 次的 Conformal sets

    参数：
        conformal_set_counter: Conformal sets 计数器
        min_count: 最小出现次数

    返回：
        filtered_sets: 过滤后的 Conformal sets
    """
    filtered_sets = defaultdict(lambda: defaultdict(int))

    for j, sets in conformal_set_counter.items():
        for conformal_set, targets in sets.items():
            total_count = sum(targets.values())
            if total_count > min_count:
                filtered_sets[j][conformal_set] = total_count

    return filtered_sets


#感觉不是有用的版本
def validate_with_conformal_set_bayes(val_loader, model, criterion, conformal_set_counter, threshold, min_count=5):
    """
    使用Conformal set统计信息进行模型验证

    参数：
        val_loader: 验证集的数据加载器
        model: 训练好的模型
        criterion: 损失函数
        conformal_set_counter: 收集的Conformal sets及其对应的目标标签计数
        threshold: 计算得到的阈值
        min_count: 仅使用出现次数大于该值的Conformal set进行贝叶斯计算

    返回：
        准确率等验证结果
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(len(conformal_set_counter)):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            batch_size = input.size(0)
            for j, out in enumerate(output):
                softmax_output = F.softmax(out, dim=1)
                _, pred = softmax_output.topk(3, 1, True, True)
                # print('pred',pred)
                pred = pred.t()
                # print('pred_t',pred)

                for k in range(batch_size):
                    conformal_set = []
                    cumulative_score = 0.0
                    for idx in range(3):
                        cumulative_score += softmax_output[k, pred[idx, k].item()].item()
                        conformal_set.append(pred[idx, k].item())
                        if cumulative_score >= threshold:
                            break

                    # 调整Conformal set的大小
                    if len(conformal_set) == 3:
                        cumulative_score_2 = softmax_output[k, pred[0, k].item()].item() + softmax_output[k, pred[1, k].item()].item()
                        if abs(threshold - cumulative_score_2) <= abs(cumulative_score - threshold):
                            conformal_set = conformal_set[:2]
                        else:
                            conformal_set = conformal_set[:3]

                    if len(conformal_set) == 2:
                        if abs(threshold - softmax_output[k, pred[0, k].item()].item()) <= abs(cumulative_score - threshold):
                            conformal_set = conformal_set[:1]

                    conformal_set_tuple = tuple(conformal_set)
                    if conformal_set_tuple in conformal_set_counter[j] and sum(conformal_set_counter[j][conformal_set_tuple].values()) > min_count:
                        bayes_probs = np.ones_like(softmax_output[k].cpu().numpy())  # Initialize with 1 for smoothing
                        for predicted_class in conformal_set_tuple:
                            for target_class, count in conformal_set_counter[j][conformal_set_tuple].items():
                                bayes_probs[target_class] += count
                        bayes_probs /= bayes_probs.sum()

                        bayes_probs = torch.tensor(bayes_probs).cuda()
                    else:
                        bayes_probs = softmax_output[k]

                    loss = criterion(bayes_probs.unsqueeze(0), target_var[k].unsqueeze(0))
                    losses.update(loss.item(), input.size(0))

                    prec1, prec5 = accuracy(bayes_probs.unsqueeze(0), target_var[k].unsqueeze(0), topk=(1, 5))
                    top1[j].update(prec1.item(), input.size(0))
                    top5[j].update(prec5.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {losses.val:.4f}\t'
                      'Acc@1 {top1_val:.4f}\t'
                      'Acc@5 {top5_val:.4f}'.format(
                    i + 1, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    losses=losses, top1_val=top1[j].val, top5_val=top5[j].val))

    for j in range(len(conformal_set_counter)):
        print(' * Layer {0} prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(j, top1=top1[j], top5=top5[j]))

    return losses.avg, [top1[j].avg for j in range(len(conformal_set_counter))], [top5[j].avg for j in range(len(conformal_set_counter))]




def update_bayes_matrix_with_conformal_set(val_loader, model, threshold):
    """
    使用Conformal set更新贝叶斯矩阵

    参数：
        val_loader: 验证集的数据加载器
        model: 训练好的模型
        criterion: 损失函数
        matrix: 贝叶斯矩阵
        threshold: 计算得到的阈值

    返回：
        更新后的贝叶斯矩阵
    """
    model.eval()
    conformal_set_counter = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # 第一遍遍历验证集，收集Conformal sets
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j, out in enumerate(output):
                softmax_output = F.softmax(out, dim=1)
                _, pred = softmax_output.topk(3, 1, True, True) #【batch, 3】

                for k in range(input.size(0)):  # input.size(0) is batch_size
                    conformal_set = []
                    cumulative_score = 0.0
                    for idx in range(3):
                        cumulative_score += softmax_output[k, pred[k, idx].item()].item()
                        conformal_set.append(pred[k, idx].item())
                        if cumulative_score >= threshold:
                            break

                    # 比较两个softmax值和三个softmax值的累积与阈值的差距
                    if len(conformal_set) == 3:
                        cumulative_score_2 = softmax_output[k, pred[k, 0].item()].item() + softmax_output[
                            k, pred[k, 1].item()].item()
                        if abs(threshold - cumulative_score_2) <= abs(cumulative_score - threshold):
                            conformal_set = conformal_set[:2]
                        else:
                            conformal_set = ('other',)

                    # 检查是否应该只包含一个类别
                    if len(conformal_set) == 2:
                        if abs(threshold - softmax_output[k, pred[k, 0].item()].item()) <= abs(
                                cumulative_score - threshold):
                            conformal_set = conformal_set[:1]

                    # 记录Conformal set对应的目标标签计数
                    conformal_set_tuple = tuple(conformal_set)
                    conformal_set_counter[j][conformal_set_tuple][target[k].item()] += 1

    return conformal_set_counter


#计算conformal prediction的阈值
def determine_threshold(val_loader, model):
    """
    计算验证集中样本的Nonconformity score并确定最佳阈值

    参数：
        val_loader: 验证集的数据加载器
        model: 训练好的模型

    返回：
        best_threshold: 使前两个类别的累积softmax值达到最佳阈值
    """
    model.eval()
    top3_scores_list = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                # 计算softmax输出
                softmax_output = F.softmax(output[j], dim=1)
                for k in range(input.size(0)):
                    # 获取前三个最大softmax值
                    top3_scores, _ = softmax_output[k].topk(3)
                    top3_scores_list.append(top3_scores.cpu().numpy())

    # 将前三个最大softmax值转换为numpy数组
    top3_scores_array = np.array(top3_scores_list)

    # 确定一个阈值，使得前两个类别的累积softmax值尽量使conformal set大小为2
    thresholds = np.linspace(0.5, 1.0, 100)  # 调整阈值范围
    best_threshold = 0
    best_coverage = 0

    # 遍历每个阈值，计算覆盖率
    for threshold in thresholds:
        print('Threshold:', threshold)
        # 计算当前阈值的覆盖率
        conformal_set_sizes = []
        for scores in top3_scores_array:
            if scores[0] >= threshold:
                conformal_set_sizes.append(1)
            elif scores[0] + scores[1] >= threshold:
                conformal_set_sizes.append(2)
            elif scores[0] + scores[1] + scores[2] >= threshold:
                conformal_set_sizes.append(3)
            else:
                conformal_set_sizes.append(4)  # 这实际上不应该发生

        coverage = np.mean(np.array(conformal_set_sizes) == 2)

        # 更新最佳阈值和最高覆盖率
        if coverage > best_coverage:
            best_coverage = coverage
            best_threshold = threshold

    return best_threshold

def validate(val_loader, model, criterion, corruption, severity):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            # if i == 0:
            #     print('target',target)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # print("output: ", output[0].size())
            # print("target: ", target_var.size())

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                    i + 1, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))
    for j in range(args.nBlocks):
        # PRINT 2 #####################################################################################
        log_result("validate", j, corruption, severity, top1[j].avg, top5[j].avg)
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg


def validate_weight(val_loader, model, criterion, weights=None):
    if weights is None:
        # weights = [1.514, 2.551, 3.71, 4.633, 5.734, 6.696, 7.594]  # 如果没有提供权重，默认为1（相等权重）
        weights = [89.22, 92.62, 94.88, 96.18, 96.62, 96.76, 96.92]

    # 确保提供的权重数量和classifier数量一致
    assert len(weights) == args.nBlocks, "weights长度应与classifier数量一致"

    # 对权重进行归一化
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # 累积变量，用于保存加权ensemble输出
            weighted_ensemble_output = torch.zeros_like(output[0])

            loss = 0.0
            for j in range(len(output)):
                # 对每个classifier的输出进行加权累加
                weighted_ensemble_output += normalized_weights[j] * output[j]

                # 计算ensemble的损失
                loss += criterion(weighted_ensemble_output, target)

                # 使用ensemble结果计算当前classifier的精度
                prec1, prec5 = accuracy(weighted_ensemble_output.data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1[0].val:.4f}\t'
                      'Acc@5 {top5[0].val:.4f}'.format(
                    i + 1, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

    # 输出每个classifier的最终平均准确率
    for j in range(args.nBlocks):
        print(' * Classifier {0} -> prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(j, top1=top1[j], top5=top5[j]))

    return losses.avg, top1[-1].avg, top5[-1].avg


def validate_ensemble(val_loader, model, criterion, corruption, severity):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # 累积变量，用于保存前面所有classifier的输出
            cumulative_output = torch.zeros_like(output[0])
            # print(output[0].shape)

            # ensemble_output = (output[0] + output[1]) / 2

            loss = 0.0
            for j in range(len(output)):
                # 累加前面所有classifier的输出
                cumulative_output += output[j]
                # 计算平均ensemble输出
                ensemble_output = cumulative_output / (j + 1)

                # 计算ensemble的损失
                loss += criterion(ensemble_output, target)

                # 使用ensemble结果计算当前classifier的精度
                prec1, prec5 = accuracy(ensemble_output.data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1[0].val:.4f}\t'
                      'Acc@5 {top5[0].val:.4f}'.format(
                    i + 1, len(val_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

    # 输出每个classifier的最终平均准确率
    for j in range(args.nBlocks):
        # PRINT 1 ####################################################################################################
        log_result("validate_ensemble", j, corruption, severity, top1[j].avg, top5[j].avg)
        print(' * Classifier {0} -> prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(j, top1=top1[j], top5=top5[j]))

    return losses.avg, top1[-1].avg, top5[-1].avg


def validate_bayes_matrix_with_conformal_prediction(val_loader, model, criterion, bayes_matrix, threshold):
    """
    通过验证集计算贝叶斯矩阵

    参数：
        val_loader: 验证集的数据加载器
        model: 训练好的模型
        criterion: 损失函数
        bayes_matrix: 贝叶斯矩阵
        threshold: 计算得到的阈值

    返回：
        bayes_matrix: 更新后的贝叶斯矩阵
    """
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                bayes_matrix[j] = cal_fault_case_with_conformal_set(output[j].data, target, bayes_matrix[j], threshold)
    return bayes_matrix

def validate_bayes_matrix_with_conformal_prediction_fixing_target_not_most(val_loader, model, criterion, bayes_matrix, threshold):
    # 通过Val set计算bayes矩阵

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                softmax_output = F.softmax(output[j].data, dim=1)
                _, pred_index = softmax_output.topk(3, 1, True, True)

                for k in range(input.size(0)):  # input.size(0) is batch_size
                    conformal_set = []
                    cumulative_score = 0.0
                    for idx in range(2):  # 只考虑前两个最大值的情况
                        cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                        conformal_set.append(pred_index[k, idx].item())
                        if cumulative_score >= threshold:
                            break

                    # 比较两个softmax值的累积与阈值的差距
                    if len(conformal_set) == 2:
                        cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                        if abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):
                            conformal_set = conformal_set[:1]

                    condition_index = (pred_index[k, 0].item(), len(conformal_set))

                    # 记录Conformal set对应的目标标签计数
                    conformal_set_tuple = tuple(conformal_set)
                    bayes_matrix[j][condition_index][target[k].item()] += 1

                    # 统计当前condition_index对应的分布
                    current_distribution = bayes_matrix[j][condition_index]

                    # 找到出现次数最多的target
                    most_common_target = max(current_distribution, key=current_distribution.get)
                    most_common_count = current_distribution[most_common_target]

                    # 如果most_common_target不是softmax最大值对应的target，进行调整
                    if most_common_target != pred_index[k, 0].item():
                        max_softmax_target = pred_index[k, 0].item()
                        current_distribution[max_softmax_target] = most_common_count + 1
                        total_count = sum(current_distribution.values())
                        for target_class in range(args.num_classes):
                            current_distribution[target_class] = (current_distribution[target_class] / total_count) * total_count

    return bayes_matrix

def validate_bayes_matrix_with_conformal_prediction_fixing_target_not_most_v2(val_loader, model, criterion, bayes_matrix, threshold):
    # 通过Val set计算bayes矩阵

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                softmax_output = F.softmax(output[j].data, dim=1)
                _, pred_index = softmax_output.topk(3, 1, True, True)
                _, pred_index_max = softmax_output.topk(1, 1, True, True)
                # print(pred_index)

                for k in range(input.size(0)):  # input.size(0) is batch_size
                    conformal_set = []
                    cumulative_score = 0.0
                    for idx in range(2):  # 只考虑前两个最大值的情况
                        cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                        conformal_set.append(pred_index[k, idx].item())
                        if cumulative_score >= threshold:
                            break

                    # 比较两个softmax值的累积与阈值的差距
                    if len(conformal_set) == 2:
                        cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                        if abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):
                            conformal_set = conformal_set[:1]

                    condition_index = (pred_index[k, 0].item(), len(conformal_set))
                    # print('condition_index',condition_index)

                    # 记录Conformal set对应的目标标签计数
                    conformal_set_tuple = tuple(conformal_set)
                    bayes_matrix[j][condition_index][target[k].item()] += 1

    # 对每个分布进行检查和修正
        for j in range(len(output)):
            for condition_index in bayes_matrix[j]:
                current_distribution = bayes_matrix[j][condition_index]
                # print('current_distribution',current_distribution)
                most_common_target = max(current_distribution, key=current_distribution.get)
                # print('most_common_target',most_common_target)
                most_common_count = current_distribution[most_common_target]
                # print('most_common_count',most_common_count)
                max_softmax_target = condition_index[0]
                # print('max_softmax_target',max_softmax_target)
                # print(condition_index)

                # 如果most_common_target不是softmax最大值对应的target，进行调整
                bayes_matrix[j][condition_index][max_softmax_target] += 0.1
                if most_common_target != max_softmax_target:
                    # print(bayes_matrix[j][condition_index][max_softmax_target])

                    bayes_matrix[j][condition_index][max_softmax_target] = most_common_count + 1
                    # print(bayes_matrix[j][condition_index][max_softmax_target])
                # total_count = sum(current_distribution.values())
                # for target_class in current_distribution:
                #     current_distribution[target_class] = (current_distribution[target_class] / total_count) * total_count

    return bayes_matrix

def cal_fault_case_with_conformal_set(output, target, matrix, threshold):
    """
    计算错误的情况，统计模型容易犯哪些错误，包括符合Conformal set长度为1和不符合的情况

    参数：
        output: 模型的输出
        target: 目标标签
        matrix: 贝叶斯矩阵
        threshold: 计算得到的阈值

    返回：
        matrix: 更新后的贝叶斯矩阵
    """
    batch_size = target.size(0)
    softmax_output = F.softmax(output, dim=1)
    _, pred = softmax_output.topk(3, 1, True, True)

    for i in range(batch_size):
        conformal_set = []
        cumulative_score = 0.0
        for idx in range(2):  # 只考虑前两个最大值的情况
            cumulative_score += softmax_output[i, pred[i, idx].item()].item()
            conformal_set.append(pred[i, idx].item())
            if cumulative_score >= threshold:
                break

        # 比较两个softmax值的累积与阈值的差距
        if len(conformal_set) == 2:
            cumulative_score_1 = softmax_output[i, pred[i, 0].item()].item()
            if abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):
                conformal_set = conformal_set[:1]

        condition_index = (pred[i, 0].item(), len(conformal_set))
        matrix[condition_index][target[i].item()] += 1

    return matrix

def validate_bayes_matrix(val_loader, model, criterion, bayes_matrix):
    # 通过Val set计算bayes矩阵

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                # prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                bayes_matrix[j] = cal_fault_case(output[j].data, target, bayes_matrix[j])
    return bayes_matrix

def validate_with_bayes_matrix(val_loader, model, criterion, bayes_matrix, corruption, severity):
    # 使用贝叶斯矩阵进行迭代概率更新
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            batch_size = input.shape[0]

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # print("output: ", output[0].size())



            for j in range(len(output)):
                if j == 0:
                    # print('j=0')
                    prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                    # print("output: ", output[j].size()) torch.Size([64, 100])
                else:
                    # _, pred_indix = output.topk(1, 1, True, True)
                    # output_bayes = bayes_matrix[i][pred_indix]
                    output_bayes = np.zeros((batch_size, args.num_classes))
                    # print('output_bayes', output_bayes)

                    for i in range(j+1):
                        if i == 0:
                            # print('i=0')
                            _, pred_indix = output[i].topk(1, 1, True, True)
                            for batch_num in range(batch_size):
                                output_bayes[batch_num] = bayes_matrix[i][pred_indix[batch_num]]

                            # print('output_bayes', output_bayes.size())





                            # pred_indix.cpu()
                            # print(pred_indix.size())
                            # output_bayes = bayes_matrix[i][pred_indix]
                        else:
                            _, pred_indix = output[i].topk(1, 1, True, True)
                            output_bayes_temp = np.zeros((batch_size, args.num_classes))
                            for batch_num in range(batch_size):
                                output_bayes_temp[batch_num] = bayes_matrix[i][pred_indix[batch_num]]

                            output_bayes = output_bayes * output_bayes_temp

                    # print(output_bayes)

                    output_bayes = torch.tensor(output_bayes).cuda()
                    # print('output_bayes',output_bayes.size()) torch.Size([64, 100])

                    # output_bayes = output_bayes.tensor().cuda()

                    prec1, prec5 = accuracy(output_bayes, target, topk=(1, 5))


                        # output_bayes = output[i].data
                # prec1 = accuracy_with_bayes_matrix(output[j].data, target, bayes_matrix[j])
                # prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                # top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    for j in range(args.nBlocks):
        # PRINT 3 ########################################################################################
        log_result("validate_with_bayes_matrix", j, corruption, severity, top1[j].avg, top5[j].avg)
        print(' * prec@1 {top1.avg:.3f} '.format(top1=top1[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return


def validate_bayes_matrix_enselble(val_loader, model, criterion, bayes_matrix, weights=None):
    # 通过Val set计算bayes矩阵
    if weights is None:
        weights = [1.0] * args.nBlocks  # 默认使用等权重

    # total_weight = sum(weights)
    # normalized_weights = [w / total_weight for w in weights]

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # 计算ensemble结果
            weighted_ensemble_output = torch.zeros_like(output[0])
            for j in range(len(output)):
                # 加权ensemble
                weighted_ensemble_output += output[j]
                bayes_matrix[j] = cal_fault_case(weighted_ensemble_output.data, target, bayes_matrix[j])

            # 使用ensemble的结果来更新贝叶斯矩阵
            # bayes_matrix = cal_fault_case(weighted_ensemble_output.data, target, bayes_matrix)

    return bayes_matrix
def validate_with_bayes_matrix_enselble(val_loader, model, criterion, bayes_matrix):
    # 使用贝叶斯矩阵进行迭代概率更新
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            batch_size = input.shape[0]

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # print("output: ", output[0].size())




            for j in range(len(output)):
                if j == 0:
                    # print('j=0')

                    prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                    # print("output: ", output[j].size()) torch.Size([64, 100])
                else:
                    # _, pred_indix = output.topk(1, 1, True, True)
                    # output_bayes = bayes_matrix[i][pred_indix]

                    output_bayes = np.zeros((batch_size, args.num_classes))
                    ensemble_output = torch.zeros_like(output[0])
                    # print('output_bayes', output_bayes)

                    for i in range(j+1):
                        if i == 0:
                            # print('i=0')
                            ensemble_output += output[i]

                            _, pred_indix = ensemble_output.topk(1, 1, True, True)
                            for batch_num in range(batch_size):
                                output_bayes[batch_num] = bayes_matrix[i][pred_indix[batch_num]]

                            # print('output_bayes', output_bayes.size())





                            # pred_indix.cpu()
                            # print(pred_indix.size())
                            # output_bayes = bayes_matrix[i][pred_indix]
                        else:
                            ensemble_output += output[i]

                            _, pred_indix = ensemble_output.topk(1, 1, True, True)
                            output_bayes_temp = np.zeros((batch_size, args.num_classes))
                            for batch_num in range(batch_size):
                                output_bayes_temp[batch_num] = bayes_matrix[i][pred_indix[batch_num]]

                            output_bayes = output_bayes * output_bayes_temp

                    # print(output_bayes)

                    output_bayes = torch.tensor(output_bayes).cuda()
                    # print('output_bayes',output_bayes.size()) torch.Size([64, 100])

                    # output_bayes = output_bayes.tensor().cuda()

                    prec1, prec5 = accuracy(output_bayes, target, topk=(1, 5))


                        # output_bayes = output[i].data
                # prec1 = accuracy_with_bayes_matrix(output[j].data, target, bayes_matrix[j])
                # prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                # top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} '.format(top1=top1[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return



def validate_with_bayes_matrix_fixing_previous_distribution(val_loader, model, criterion, bayes_matrix):
    # 使用贝叶斯矩阵进行迭代概率更新
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(device=None)
            input = input.cuda()

            batch_size = input.shape[0]

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            # print("output: ", output[0].size())



            for j in range(len(output)):
                if j == 0:
                    # print('j=0')
                    prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                    # print("output: ", output[j].size()) torch.Size([64, 100])
                else:
                    # _, pred_indix = output.topk(1, 1, True, True)
                    # output_bayes = bayes_matrix[i][pred_indix]
                    output_bayes = np.zeros((batch_size, args.num_classes))
                    # print('output_bayes', output_bayes)

                    for i in range(j+1):
                        if i == 0:
                            # print('i=0')
                            _, pred_indix = output[i].topk(1, 1, True, True)
                            for batch_num in range(batch_size):
                                output_bayes[batch_num] = bayes_matrix[i][pred_indix[batch_num]]

                            # print('output_bayes', output_bayes.size())

                            output_bayes = torch.tensor(output_bayes).cuda()
                            output_bayes = F.softmax(output_bayes, dim=1)






                            # pred_indix.cpu()
                            # print(pred_indix.size())
                            # output_bayes = bayes_matrix[i][pred_indix]
                        else:
                            _, pred_indix = output[i].topk(1, 1, True, True)
                            output_bayes_temp = np.zeros((batch_size, args.num_classes))
                            for batch_num in range(batch_size):
                                output_bayes_temp[batch_num] = bayes_matrix[i][pred_indix[batch_num]]

                            output_bayes_temp = torch.tensor(output_bayes_temp).cuda()
                            output_bayes_temp = F.softmax(output_bayes_temp, dim=1)

                            # Element-wise multiplication of the softmaxed Bayesian matrix
                            output_bayes = output_bayes * output_bayes_temp  ##!!!!!
                            # output_bayes = output_bayes_temp

                            # output_bayes = output_bayes * output_bayes_temp

                    # print(output_bayes)

                    output_bayes = torch.tensor(output_bayes).cuda()
                    # print('output_bayes',output_bayes.size()) torch.Size([64, 100])

                    # output_bayes = output_bayes.tensor().cuda()

                    prec1, prec5 = accuracy(output_bayes, target, topk=(1, 5))


                        # output_bayes = output[i].data
                # prec1 = accuracy_with_bayes_matrix(output[j].data, target, bayes_matrix[j])
                # prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                # top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} '.format(top1=top1[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return


def initialize_bayes_matrix(num_layers, num_labels, initial_value=0.5):
    """
    初始化贝叶斯矩阵，每种可能的情况赋予初始值

    参数：
        num_layers: 输出层的数量
        num_labels: 每个输出层的标签数量
        initial_value: 初始化值

    返回：
        bayes_matrix: 初始化后的贝叶斯矩阵
    """
    bayes_matrix = {
        layer: defaultdict(lambda: defaultdict(lambda: initial_value))
        for layer in range(num_layers)
    }
    for layer in range(num_layers):
        for label in range(num_labels):
            for length in [1, 2]:
                for target in range(num_labels):
                    bayes_matrix[layer][(label, length)][target] = initial_value
    return bayes_matrix

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    model_filename = os.path.join(model_dir, 'msdnet-step=4-block=5.pth.tar')

    # if os.path.exists(latest_filename):
    #     with open(latest_filename, 'r') as fin:
    #         model_filename = fin.readlines()[0].strip()
    # else:
    #     return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    # print("maxk: ",maxk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print("pred: ", pred.size())
    pred = pred.t()
    # print("pred: ", pred.size())
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def cal_fault_case(output, target, matrix):
    #计算错误的case, 统计模型容易犯哪些错误，比如容易把class A 错分为class B
    #只考虑结果，即模型输出的标签，暂不考虑softmax的数值

    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    # print("pred: ", pred)
    # print("target: ", target)
    for i in range(batch_size):
        matrix[pred[i]][target[i]] += 1

    return matrix
    # matrix

def cal_fault_case_with_norm(output, target, matrix):
    #计算错误的case, 统计模型容易犯哪些错误，比如容易把class A 错分为class B
    #只考虑结果，即模型输出的标签，暂不考虑softmax的数值

    for i in range(100):
        for j in range(100):
            matrix[i][j] = 0.5

    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    # print("pred: ", pred)
    # print("target: ", target)
    for i in range(batch_size):
        matrix[pred[i]][target[i]] += 1

    return matrix



def count_labels_in_val_set(val_loader):
    label_count = defaultdict(int)

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cpu().numpy()
            for label in target:
                label_count[label] += 1

    return label_count


def print_max_min_labels(label_count):
    if not label_count:
        print("No labels found.")
        return

    max_label = max(label_count, key=label_count.get)
    min_label = min(label_count, key=label_count.get)

    print(f"Label with max count: {max_label} ({label_count[max_label]} samples)")
    print(f"Label with min count: {min_label} ({label_count[min_label]} samples)")



def validate_with_bayes_matrix_conformal_prediction(val_loader, model, criterion, bayes_matrix, threshold, corruption, severity):
    # 使用贝叶斯矩阵进行迭代概率更新

    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            batch_size = input.shape[0]

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            output_bayes = np.zeros((batch_size, args.num_classes))

            for j in range(len(output)):
                if j == 0:
                    softmax_output = F.softmax(output[j].data, dim=1)
                    # print("softmax_output: ", softmax_output)
                    _, pred_index = softmax_output.topk(3, 1, True, True)

                    for k in range(batch_size):
                        conformal_set = []
                        cumulative_score = 0.0
                        for idx in range(2):  # 只考虑前两个最大值的情况
                            cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                            conformal_set.append(pred_index[k, idx].item())
                            if cumulative_score >= threshold:
                                break

                        # 比较两个softmax值的累积与阈值的差距
                        if len(conformal_set) == 2:
                            cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                            if abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):
                                conformal_set = conformal_set[:1]

                        condition_index = (pred_index[k, 0].item(), len(conformal_set))

                        for target_class in range(args.num_classes):
                            output_bayes[k][target_class] += bayes_matrix[j][condition_index][target_class]

                    output_bayes = torch.tensor(output_bayes).cuda()

                if j > 0:
                    output_bayes_temp = np.zeros((batch_size, args.num_classes))

                    softmax_output = F.softmax(output[j].data, dim=1)

                    _, pred_index = softmax_output.topk(3, 1, True, True)

                    for k in range(batch_size):
                        conformal_set = []
                        cumulative_score = 0.0
                        for idx in range(2):
                            cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                            conformal_set.append(pred_index[k, idx].item())
                            if cumulative_score >= threshold:
                                break

                        if len(conformal_set) == 2:
                            cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                            if (threshold - cumulative_score_1) > 0:
                                conformal_set = conformal_set
                            elif abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):

                                conformal_set = conformal_set[:1]


                        condition_index = (pred_index[k, 0].item(), len(conformal_set))
                        # print("condition_index: ", j, condition_index)
                        for target_class in range(args.num_classes):
                            output_bayes_temp[k][target_class] += bayes_matrix[j][condition_index][target_class]

                    # Apply softmax to the temporary Bayesian matrix
                    output_bayes_temp = torch.tensor(output_bayes_temp).cuda()
                    # output_bayes_temp = F.softmax(output_bayes_temp, dim=1)
                    # print("output_bayes_temp: ", output_bayes_temp)
                    output_bayes_temp /= output_bayes_temp.sum(axis=1, keepdims=True)
                    # print("output_bayes_temp_after: ", output_bayes_temp)
                    # print(j)
                    # print("output_bayes_temp: ", output_bayes_temp)
                    # print("output_bayes: ", output_bayes)

                    # output_bayes = torch.tensor(output_bayes).cuda()

                    output_bayes = output_bayes * output_bayes_temp #!!!!
                    # print("output_bayes_after: ", output_bayes)
                
                # output_bayes_tensor = F.softmax(output_bayes_tensor, dim=1)
                prec1, prec5 = accuracy(output_bayes, target, topk=(1, 5)) ###!!!
                output_bayes /= output_bayes.sum(axis=1, keepdims=True)
                output_bayes += 0.002
                output_bayes /= output_bayes.sum(axis=1, keepdims=True)
                # print("output_bayes_after_2: ", output_bayes)
                # output_bayes = F.softmax(output_bayes, dim=1)

                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))



    for j in range(args.nBlocks):
        # PRINT 4 #################################################################################################################
        log_result("validate_with_bayes_matrix_conformal_prediction", j, corruption, severity, top1[j].avg, top5[j].avg)
        print(' * Block {j} prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(j=j, top1=top1[j], top5=top5[j]))
    return


def validate_with_bayes_matrix_conformal_prediction_cluster(val_loader, model, criterion, bayes_matrix, threshold, cluster):
    # 使用贝叶斯矩阵进行迭代概率更新

    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            batch_size = input.shape[0]

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            output_bayes = np.zeros((batch_size, args.num_classes))

            for j in range(len(output)):
                if j == 0:
                    softmax_output = F.softmax(output[j].data, dim=1)
                    # print("softmax_output: ", softmax_output)
                    _, pred_index = softmax_output.topk(3, 1, True, True)

                    for k in range(batch_size):
                        conformal_set = []
                        cumulative_score = 0.0
                        for idx in range(2):  # 只考虑前两个最大值的情况
                            cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                            conformal_set.append(pred_index[k, idx].item())
                            if cumulative_score >= threshold:
                                break

                        # 比较两个softmax值的累积与阈值的差距
                        if len(conformal_set) == 2:
                            cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                            if abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):
                                conformal_set = conformal_set[:1]

                        condition_index = (pred_index[k, 0].item(), len(conformal_set))

                        if len(conformal_set) == 2:
                            for target_class in range(args.num_classes):
                                if target_class in cluster:
                                    output_bayes[k][target_class] += bayes_matrix[j][condition_index][target_class]
                                output_bayes[k][target_class] += bayes_matrix[j][condition_index][target_class]

                        for target_class in range(args.num_classes):
                            output_bayes[k][target_class] += bayes_matrix[j][condition_index][target_class]

                    output_bayes = torch.tensor(output_bayes).cuda()

                if j > 0:
                    output_bayes_temp = np.zeros((batch_size, args.num_classes))

                    softmax_output = F.softmax(output[j].data, dim=1)

                    _, pred_index = softmax_output.topk(3, 1, True, True)

                    for k in range(batch_size):
                        conformal_set = []
                        cumulative_score = 0.0
                        for idx in range(2):
                            cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                            conformal_set.append(pred_index[k, idx].item())
                            if cumulative_score >= threshold:
                                break

                        if len(conformal_set) == 2:
                            cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                            if (threshold - cumulative_score_1) > 0:
                                conformal_set = conformal_set
                            elif abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):

                                conformal_set = conformal_set[:1]

                        condition_index = (pred_index[k, 0].item(), len(conformal_set))
                        # print("condition_index: ", j, condition_index)
                        for target_class in range(args.num_classes):
                            output_bayes_temp[k][target_class] += bayes_matrix[j][condition_index][target_class]

                    # Apply softmax to the temporary Bayesian matrix
                    output_bayes_temp = torch.tensor(output_bayes_temp).cuda()
                    # output_bayes_temp = F.softmax(output_bayes_temp, dim=1)
                    # print("output_bayes_temp: ", output_bayes_temp)
                    output_bayes_temp /= output_bayes_temp.sum(axis=1, keepdims=True)
                    # print("output_bayes_temp_after: ", output_bayes_temp)
                    # print(j)
                    # print("output_bayes_temp: ", output_bayes_temp)
                    # print("output_bayes: ", output_bayes)

                    # output_bayes = torch.tensor(output_bayes).cuda()

                    output_bayes = output_bayes * output_bayes_temp  # !!!!
                    # print("output_bayes_after: ", output_bayes)

                # output_bayes_tensor = F.softmax(output_bayes_tensor, dim=1)
                prec1, prec5 = accuracy(output_bayes, target, topk=(1, 5))  ###!!!
                output_bayes /= output_bayes.sum(axis=1, keepdims=True)
                output_bayes += 0.002
                output_bayes /= output_bayes.sum(axis=1, keepdims=True)
                # print("output_bayes_after_2: ", output_bayes)
                # output_bayes = F.softmax(output_bayes, dim=1)

                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

    for j in range(args.nBlocks):
        print(' * Block {j} prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(j=j, top1=top1[j], top5=top5[j]))
    return



def validate_bayes_matrix_with_conformal_prediction_fixing_target_not_most_v2_for_len1(val_loader, model, criterion, bayes_matrix, threshold):
    # 通过Val set计算bayes矩阵

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                softmax_output = F.softmax(output[j].data, dim=1)
                _, pred_index = softmax_output.topk(3, 1, True, True)
                # print(pred_index)

                for k in range(input.size(0)):  # input.size(0) is batch_size
                    conformal_set = []
                    cumulative_score = 0.0
                    for idx in range(2):  # 只考虑前两个最大值的情况
                        cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                        conformal_set.append(pred_index[k, idx].item())
                        if cumulative_score >= threshold:
                            break

                    # 比较两个softmax值的累积与阈值的差距
                    if len(conformal_set) == 2:
                        cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                        if abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):
                            conformal_set = conformal_set[:1]

                    condition_index = (pred_index[k, 0].item(), len(conformal_set))
                    # print('condition_index',condition_index)

                    # 记录Conformal set对应的目标标签计数
                    if len(conformal_set) == 1:
                        conformal_set_tuple = tuple(conformal_set)
                        bayes_matrix[j][condition_index][target[k].item()] += 1

    # 对每个分布进行检查和修正
        for j in range(len(output)):
            for condition_index in bayes_matrix[j]:
                if condition_index[1] == 1:
                    current_distribution = bayes_matrix[j][condition_index]
                    # print('current_distribution',current_distribution)
                    most_common_target = max(current_distribution, key=current_distribution.get)
                    # print('most_common_target',most_common_target)
                    most_common_count = current_distribution[most_common_target]
                    # print('most_common_count',most_common_count)
                    max_softmax_target = condition_index[0]
                    # print('max_softmax_target',max_softmax_target)
                    # print(condition_index)

                    # 如果most_common_target不是softmax最大值对应的target，进行调整
                    bayes_matrix[j][condition_index][max_softmax_target] += 0.1
                    if most_common_target != max_softmax_target:
                        # print(bayes_matrix[j][condition_index][max_softmax_target])

                        bayes_matrix[j][condition_index][max_softmax_target] = most_common_count + 1
                    # print(bayes_matrix[j][condition_index][max_softmax_target])
                # total_count = sum(current_distribution.values())
                # for target_class in current_distribution:
                #     current_distribution[target_class] = (current_distribution[target_class] / total_count) * total_count

    return bayes_matrix



def validate_bayes_matrix_with_conformal_prediction_fixing_target_not_most_v2_for_len2_cluster(val_loader, model, criterion, bayes_matrix, threshold, cluster_mapping):
    # 通过Val set计算bayes矩阵

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                softmax_output = F.softmax(output[j].data, dim=1)
                _, pred_index = softmax_output.topk(3, 1, True, True)
                # print(pred_index)

                for k in range(input.size(0)):  # input.size(0) is batch_size
                    conformal_set = []
                    cumulative_score = 0.0
                    for idx in range(2):  # 只考虑前两个最大值的情况
                        cumulative_score += softmax_output[k, pred_index[k, idx].item()].item()
                        conformal_set.append(pred_index[k, idx].item())
                        if cumulative_score >= threshold:
                            break

                    # 比较两个softmax值的累积与阈值的差距
                    if len(conformal_set) == 2:
                        cumulative_score_1 = softmax_output[k, pred_index[k, 0].item()].item()
                        if abs(threshold - cumulative_score_1) <= abs(threshold - cumulative_score):
                            conformal_set = conformal_set[:1]

                        condition_index = (pred_index[k, 0].item(), len(conformal_set))
                        # print('condition_index',condition_index)

                        pred_cluster = cluster_mapping[pred_index[k, 0].item()]
                        conformal_clusters = [cluster_mapping[class_id] for class_id in conformal_set]
                        target_cluster = cluster_mapping[target[k].item()]

                        condition_index = (pred_cluster, len(conformal_clusters))

                        # 记录Conformal set对应的目标标签计数
                        if len(conformal_clusters) == 2:
                            conformal_set_tuple = tuple(conformal_clusters)
                            bayes_matrix[j][condition_index][target_cluster] += 1

                    # # 记录Conformal set对应的目标标签计数
                    # if len(conformal_set) == 2:
                    #     conformal_set_tuple = tuple(conformal_set)
                    #     bayes_matrix[j][condition_index][target[k].item()] += 1

    # 对每个分布进行检查和修正
        for j in range(len(output)):
            for condition_index in bayes_matrix[j]:
                if condition_index[1] == 2:
                    current_distribution = bayes_matrix[j][condition_index]
                    # print('current_distribution',current_distribution)
                    most_common_target = max(current_distribution, key=current_distribution.get)
                    # print('most_common_target',most_common_target)
                    most_common_count = current_distribution[most_common_target]
                    # print('most_common_count',most_common_count)
                    max_softmax_target = condition_index[0]
                    # print('max_softmax_target',max_softmax_target)
                    # print(condition_index)

                    # 如果most_common_target不是softmax最大值对应的target，进行调整
                    bayes_matrix[j][condition_index][max_softmax_target] += 0.1
                    if most_common_target != max_softmax_target:
                        # print(bayes_matrix[j][condition_index][max_softmax_target])

                        bayes_matrix[j][condition_index][max_softmax_target] = most_common_count + 1
                        # print(bayes_matrix[j][condition_index][max_softmax_target])
                    # total_count = sum(current_distribution.values())
                    # for target_class in current_distribution:
                    #     current_distribution[target_class] = (current_distribution[target_class] / total_count) * total_count

    return bayes_matrix


def determine_threshold_with_accuracy(val_loader, model, desired_accuracy=0.8):
    """
    计算验证集中样本的Nonconformity score并确定最佳阈值

    参数：
        val_loader: 验证集的数据加载器
        model: 训练好的模型
        desired_accuracy: 期望的准确率

    返回：
        best_thresholds: 每个classifier的最佳阈值
    """
    model.eval()
    top_scores_dict = defaultdict(list)
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                # 计算softmax输出
                softmax_output = F.softmax(output[j], dim=1)
                for k in range(input.size(0)):
                    # 获取最大的softmax值
                    top_score, top_index = softmax_output[k].max(0)
                    top_scores_dict[j].append((top_score.item(), top_index.item(), target[k].item()))

                    # 更新总数统计
                    # total_counts[j] += 1
                    # # 更新正确预测计数
                    # if target[k] == top_index:
                    #     correct_counts[j] += 1
    print('done softmax')
    best_thresholds = {}
    for j in top_scores_dict:
        # 将最大的softmax值转换为numpy数组
        top_scores_array = np.array([score for score, _, _ in top_scores_dict[j]])
        top_indices_array = np.array([index for _, index, _ in top_scores_dict[j]])
        target = np.array([target for _, _, target in top_scores_dict[j]])

        # print(top_scores_array, top_indices_array, target)

        # 确定一个阈值，使得最大类别的softmax值尽量接近预设准确率
        thresholds = np.linspace(0, 1.0, 200)  # 调整阈值范围
        best_threshold = 0
        best_accuracy_diff = float('inf')

        for threshold in thresholds:
            correct = 0
            total = 0
            for i in range(len(top_scores_array)):
                score = top_scores_array[i]
                index = top_indices_array[i]
                if score >= threshold:
                    total += 1
                    if target[i] == index:
                        correct += 1


            accuracy = correct / total

            accuracy_diff = abs(accuracy - desired_accuracy)
            # print(best_accuracy_diff, accuracy)
            if accuracy_diff < best_accuracy_diff:
                best_accuracy_diff = accuracy_diff
                best_threshold = threshold

        best_thresholds[j] = best_threshold
        print(f"Classifier {j}: Best threshold = {best_threshold}, Accuracy Difference = {best_accuracy_diff}")

    return best_thresholds

def determine_threshold_with_accuracy_per_class(val_loader, model, desired_accuracy=0.8, n_clusters=10):
    """
    计算验证集中样本的Nonconformity score并确定每个类的最佳阈值

    参数：
        val_loader: 验证集的数据加载器
        model: 训练好的模型
        desired_accuracy: 期望的准确率

    返回：
        best_thresholds: 每个classifier的每个类的最佳阈值
    """
    model.eval()
    top_scores_dict = defaultdict(lambda: defaultdict(list))
    class_counts = defaultdict(int)

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for j in range(len(output)):
                # 计算softmax输出
                softmax_output = F.softmax(output[j], dim=1)
                for k in range(input.size(0)):
                    # 获取最大的softmax值
                    top_score, top_index = softmax_output[k].max(0)
                    top_scores_dict[j][target[k].item()].append((top_score.item(), top_index.item()))
                    class_counts[target[k].item()] += 1

    best_thresholds = {}
    for j in top_scores_dict:
        best_thresholds[j] = {}
        for target_class in top_scores_dict[j]:
            # 将最大的softmax值转换为numpy数组
            top_scores_array = np.array([score for score, _ in top_scores_dict[j][target_class]])
            top_indices_array = np.array([index for _, index in top_scores_dict[j][target_class]])

            # 确定一个阈值，使得最大类别的softmax值尽量接近预设准确率
            thresholds = np.linspace(0.0, 1.0, 100)  # 调整阈值范围
            best_threshold = 0
            best_acc = 0
            best_accuracy_diff = float('inf')

            for threshold in thresholds:
                correct = 0
                total = 0.01
                for i in range(len(top_scores_array)):
                    score = top_scores_array[i]
                    index = top_indices_array[i]
                    if score >= threshold:
                        total += 1
                        if index == target_class:
                            correct += 1

                accuracy = correct / total
                accuracy_diff = abs(accuracy - desired_accuracy)
                if accuracy_diff < best_accuracy_diff:
                    best_accuracy_diff = accuracy_diff
                    best_threshold = threshold
                    best_acc = accuracy

            best_thresholds[j][target_class] = best_threshold
            print(f"Classifier {j}: Best threshold = {best_threshold}, Accuracy Difference = {best_accuracy_diff}", f"Accuracy = {best_acc}", {target_class})

    # all_thresholds = []
    # for j in best_thresholds:
    #     for target_class in best_thresholds[j]:
    #         # print(f"Classifier {j}, Class {target_class}: Best threshold = {best_thresholds[j][target_class]}")
    #         all_thresholds.append(best_thresholds[j][target_class])
    #
    # all_thresholds = np.array(all_thresholds).reshape(-1, 1)

    # 使用KMeans进行聚类
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_thresholds)
    # clustered_thresholds = kmeans.cluster_centers_
    #
    # # 映射每个类到其对应的聚类结果
    # threshold_clusters = {}
    # for j in best_thresholds:
    #     threshold_clusters[j] = {}
    #     for target_class in best_thresholds[j]:
    #         threshold = best_thresholds[j][target_class]
    #         cluster_index = kmeans.predict([[threshold]])[0]
    #         threshold_clusters[j][target_class] = clustered_thresholds[cluster_index][0]
    #
    #         print(f"Classifier {j}, Target Class {target_class}: Best threshold = {best_threshold}, Accuracy Difference = {best_accuracy_diff}")

    return best_thresholds

def validate_thresholds_on_testset(test_loader, model, best_thresholds):
    """
    在测试集中验证每个类的最佳阈值

    参数：
        test_loader: 测试集的数据加载器
        model: 训练好的模型
        best_thresholds: 每个classifier的每个类的最佳阈值

    返回：
        results: 每个类在测试集上的准确率
    """
    model.eval()
    # correct_counts = defaultdict(int)
    # total_counts = defaultdict(int)
    correct_counts = np.zeros((7, 100))
    total_counts = np.zeros((7, 100))
    class_accuracy = np.zeros((7, 100))
    print(total_counts.shape)
    # for p in range(7):
    #     correct_counts[p] = {}
    #     total_counts[p] = {}

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]



            for j in range(len(output)):
                # 计算softmax输出
                softmax_output = F.softmax(output[j], dim=1)
                for k in range(input.size(0)):
                    # 获取最大softmax值及其对应的索引
                    top_score, top_index = softmax_output[k].max(0)
                    true_class = target[k].item()
                    threshold = best_thresholds[j][true_class]  # 使用默认值0.5避免没有阈值的情况

                    # 判断预测是否正确
                    if top_score.item() >= threshold:
                        total_counts[j][true_class] += 1

                        if top_index.item() == true_class:
                            correct_counts[j][true_class] += 1


    # 计算每个类的准确率
    for i in range(7):

        for class_label in range(100):
            accuracy = correct_counts[i][class_label] / total_counts[i][class_label] if total_counts[i][class_label] > 0 else 0
            class_accuracy[i][class_label] = accuracy
            print(f"{i},Class {class_label}: Accuracy = {accuracy:.3f}")

    return class_accuracy

if __name__ == '__main__':
    main()
