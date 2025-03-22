import json, os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictionary', default='./performance', type=str)
    parser.add_argument('--file', required=True)
    args = parser.parse_args()

    final_load_path = os.path.join(args.dictionary, args.file)
    data = json.load(open(final_load_path, 'r'))
    pair_data = data[0].keys()
    metrics = data[0][list(pair_data)[0]].keys()
    mean_base = len(list(pair_data))
    final_result = {}
    for metric in list(metrics):
        final_result[metric] = 0
    for each_pair in list(pair_data):
        for metric in list(metrics):
            final_result[metric] += data[0][each_pair][metric]
    for metric in list(metrics):
        final_result[metric] /= mean_base
    print(json.dumps(final_result, indent=4, ensure_ascii=False))

