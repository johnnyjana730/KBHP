import argparse
import numpy as np
from data_loader import load_data
from parser import parse_args
from train import train
import sys
import logging
import logging.handlers

np.random.seed(5)

def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def env_by_item(args, data_info):

	for epoch in range(1):
		args.epoch = epoch
		args.sw_stage = 0
		train(args, data_info, show_loss)

		args.load_pretrain_emb = True
		args.sw_stage += 1
		train(args, data_info, show_loss)


if __name__ == "__main__":
	args = parse_args()

	logger = get_logger(args.path.log_file + '.txt')
	args.logger = logger

	show_loss = False
	data_info = load_data(args)

	env_by_item(args, data_info)