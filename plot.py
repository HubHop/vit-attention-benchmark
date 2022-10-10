import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_cpu():
    data = {}
    with open('results/attention_benchmark.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                data[row[0]] = {
                    'params': row[1],
                    'flops': row[2],
                    'cpu': int(row[3]),
                    'gpu': int(row[4]),
                }

    gpu_speeds = np.array([v['cpu'] for k, v in data.items()])
    names = np.array(list(data.keys()))

    indices = np.argsort(gpu_speeds)[::-1]
    gpu_speeds = gpu_speeds[indices]
    names = names[indices]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=30)
    bars = ax.bar(names, gpu_speeds, width=0.6, color='#669DB3FF')
    ax.bar_label(bars)
    ax.set_axisbelow(True)
    fig.autofmt_xdate(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel('Throughput (imgs/s)')
    plt.grid(linewidth=0.3, axis='y', linestyle='--')

    plt.ylim([0, 1100])
    plt.title('Intel® Core™ i9-10900X CPU @ 3.70GHz')
    plt.savefig('/data1/vit-attention-projects/vit-attention-benchmark/res_cpu.pdf')
    print('..')

def plot_gpu():
    data = {}
    with open('results/attention_benchmark.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                data[row[0]] = {
                    'params': row[1],
                    'flops': row[2],
                    'cpu': row[3],
                    'gpu': int(row[4]),
                }

    gpu_speeds = np.array([v['gpu'] for k, v in data.items()])
    names = np.array(list(data.keys()))

    indices = np.argsort(gpu_speeds)[::-1]
    gpu_speeds = gpu_speeds[indices]
    names = names[indices]

    # fig, ax = plt.subplots(figsize=(6, 5), dpi=30)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=30)
    bars = ax.bar(names, gpu_speeds, width=0.6, color='#669DB3FF')
    ax.bar_label(bars)
    ax.set_axisbelow(True)
    fig.autofmt_xdate(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.ylabel('Throughput (imgs/s)')
    plt.grid(linewidth=0.3, axis='y', linestyle='--')

    plt.ylim([2500, 5300])
    plt.title('NVIDIA GeForce RTX 3090')
    plt.savefig('/data1/vit-attention-projects/vit-attention-benchmark/res_gpu.pdf')
    print('..')


if __name__ == '__main__':
    plot_cpu()
    plot_gpu()
