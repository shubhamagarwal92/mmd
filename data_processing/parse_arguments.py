import argparse


def arg_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-dir_path', default='/v2', type=str, help='Data path')
	parser.add_argument('-context_size', default=2, type=int, help='context size')
	parser.add_argument('-max_images', default=5, type=int, help='max images in a turn')
	parser.add_argument('-max_len', default=40, type=int, help='max length of utterance')
	parser.add_argument('-pad_id', default=0, type=int, help='max_images')
	parser.add_argument('-unk_id', default=1, type=int, help='max_images')
	parser.add_argument('-user_start_id', default=2, type=int, help='user_start_id')
	parser.add_argument('-user_end_id', default=3, type=int, help='user_end_id')
	parser.add_argument('-sys_start_id', default=4, type=int, help='sys_start_id')
	parser.add_argument('-sys_end_id', default=5, type=int, help='sys_end_id')
	parser.add_argument('-cutoff', default=2, type=int, help='vocab cutoff')
	parser.add_argument('-start_id', default=2, type=int, help='start_id')
	parser.add_argument('-end_id', default=3, type=int, help='end_id')
	# parser.add_argument('-use_cuda', action='store_true', default=False, help='use gpu')
	parser.add_argument('-out_dir_path', type=str, help='out_dir_path')
	parser.add_argument('-vocab_pkl_path', default='/vocab.pkl', type=str, 
						help='vocab_pkl_path')	
	parser.add_argument('-vocab_stats_path', default='/vocab_stats.pkl', type=str, 
						help='vocab_stats_path')
	parser.add_argument('-annoy_path', default='/image_annoy_index', type=str, 
						help='path to the output dir path')
	parser.add_argument('-annoy_file', default='./annoy.ann', type=str, 
						help='path to the output dir path')
	parser.add_argument('-annoy_pkl', default='./FileNameMapToIndex.pkl', 
						type=str, help='path to the output dir path')
	parser.add_argument('-image_rep_size', default=4096, type=int, 
						help='path to the output dir path')
	args = parser.parse_args()
	return args