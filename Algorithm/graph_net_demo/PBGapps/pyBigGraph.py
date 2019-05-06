#-*- encoding:utf-8 -*-

import argparse
import os
from itertools import chain

import attr

import torchbiggraph.converters.utils as utils
from torchbiggraph.config import parse_config
from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train

from Algorithm.graph_net_demo.utils.filtered_eval import *

FILENAMES = {
    'train': 'FB15k/freebase_mtr100_mte100-train.txt',
    'valid': 'FB15k/freebase_mtr100_mte100-valid.txt',
    'test': 'FB15k/freebase_mtr100_mte100-test.txt',
}
def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir
# E:\ZacharyChu\python\GraphEmbedding\GraphEmbedding\data\FB15k\freebase_mtr100_mte100-test.txt
def main():
    parser = argparse.ArgumentParser(description='Example on FB15k')
    parser.add_argument('--config', default='./fb15k_config.py',
                        help='Path to config file')
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('--data_dir', default='../../../data',
                        help='where to save processed data')
    parser.add_argument('--no-filtered', dest='filtered', action='store_false',
                        help='Run unfiltered eval')
    args = parser.parse_args()

    if args.param is not None:
        overrides = chain.from_iterable(args.param)  # flatten
    else:
        overrides = None

    # download data
    data_dir = args.data_dir
    #fpath = utils.download_url(FB15K_URL, data_dir)
    #utils.extract_tar(fpath)
    #print('Downloaded and extracted file.')

    edge_paths = [os.path.join(data_dir, name) for name in FILENAMES.values()]
    print('edge_paths',edge_paths)
    convert_input_data(
        args.config,
        edge_paths,
        lhs_col=0,
        rhs_col=2,
        rel_col=1,
    )

    config = parse_config(args.config, overrides)

    train_path = [convert_path(os.path.join(data_dir, FILENAMES['train']))]
    train_config = attr.evolve(config, edge_paths=train_path)

    train(train_config)

    eval_path = [convert_path(os.path.join(data_dir, FILENAMES['test']))]
    relations = [attr.evolve(r, all_negs=True) for r in config.relations]
    eval_config = attr.evolve(config, edge_paths=eval_path, relations=relations)
    if args.filtered:
        filter_paths = [
            convert_path(os.path.join(data_dir, FILENAMES['test'])),
            convert_path(os.path.join(data_dir, FILENAMES['valid'])),
            convert_path(os.path.join(data_dir, FILENAMES['train'])),
        ]
        do_eval(eval_config, FilteredRankingEvaluator(eval_config, filter_paths))
    else:
        do_eval(eval_config)


if __name__ == "__main__":
    #f = '../../../data/FB15k/freebase_mtr100_mte100-test.txt'
    #f = open(f)
    #f.close()
    main()
