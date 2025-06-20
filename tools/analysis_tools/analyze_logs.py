# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()

def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[args.interval - 1]]:
                raise KeyError(f'{args.json_logs[i]} does not contain metric {metric}')
            if args.mode == 'eval':
                if min(epochs) == args.interval:
                    x0 = args.interval
                else:
                    if min(epochs) % args.interval == 0:
                        x0 = min(epochs)
                    else:
                        x0 = min(epochs) + args.interval - min(epochs) % args.interval
                xs = np.arange(x0, max(epochs) + 1, args.interval)
                ys = []
                for epoch in epochs[args.interval - 1::args.interval]:
                    ys += log_dict[epoch][metric]
                if not log_dict[epoch][metric]:
                    xs = xs[:-1]
                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[args.interval-1]]['iter'][-1]
                for epoch in epochs[args.interval - 1::args.interval]:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()

def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser('plot_curve', help='绘制曲线的解析器')
    parser_plt.add_argument('json_logs', type=str, nargs='+', help='json格式训练日志的路径')
    parser_plt.add_argument('--keys', type=str, nargs='+', default=['mAP_0.25'], help='要绘制的指标')
    parser_plt.add_argument('--title', type=str, help='图表标题')
    parser_plt.add_argument('--legend', type=str, nargs='+', default=None, help='每个图表的图例')
    parser_plt.add_argument('--backend', type=str, default=None, help='plt后端')
    parser_plt.add_argument('--style', type=str, default='dark', help='plt样式')
    parser_plt.add_argument('--out', type=str, default=None)
    parser_plt.add_argument('--mode', type=str, default='train')
    parser_plt.add_argument('--interval', type=int, default=1)

def add_time_parser(subparsers):
    parser_time = subparsers.add_parser('cal_train_time', help='计算每次训练迭代的平均时间的解析器')
    parser_time.add_argument('json_logs', type=str, nargs='+', help='json格式训练日志的路径')
    parser_time.add_argument('--include-outliers', action='store_true', help='计算平均时间时包含每个epoch的第一个值')

def parse_args():
    parser = argparse.ArgumentParser(description='分析Json日志')
    subparsers = parser.add_subparsers(dest='task', help='任务解析器')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args

def load_json_logs(json_logs):
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts

def main():
    args = parse_args()
    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    eval(args.task)(log_dicts, args)

if __name__ == '__main__':
    main()
