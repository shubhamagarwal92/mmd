import os
import json
import cPickle as pkl
import argparse

def main(args):
	# labels_file = os.path.join(args.data_dir, "test_state_labels.txt")
	# pred_file = args.pred
	# target_file = args.target
	
	with open(args.labels_file, 'r') as state_file, \
		open(args.pred, 'r') as pred_file, \
		open(args.target, 'r') as target_file, \
		open(args.context, 'r') as context_file:
		for state_line, pred, target, context in zip(state_file, pred_file, \
			target_file, context_file):
			# state = ','.join(state_line.split(',')[:-1])
			state_dir = str(state_line.strip())
			state_out_dir = os.path.join(args.results_dir, state_dir)
			if not os.path.exists(state_out_dir):
				os.makedirs(state_out_dir)
			state_pred_file = os.path.join(state_out_dir, "pred_"+ str(args.checkpoint)+"_"+args.beam+".txt")
			state_target_file = os.path.join(state_out_dir,"test_tokenized.txt")
			state_context_file = os.path.join(state_out_dir,"test_context_text.txt")
			with open(state_pred_file, 'a+') as fp:
				fp.write(pred)
			with open(state_target_file, 'a+') as fp:
				fp.write(target)
			with open(state_context_file, 'a+') as fp:
				fp.write(context)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-labels_file', type=str, default='./data/vocab.pkl')
	parser.add_argument('-results_dir', type=str, default='./data/vocab.pkl')
	parser.add_argument('-pred', type=str, help='prediction file path')
	parser.add_argument('-target', type=str, help='target file path')
	parser.add_argument('-context', type=str, help='target file path')
	parser.add_argument('-checkpoint', type=str, help='target file path')	
	parser.add_argument('-beam', type=str, help='target file path')		
	args = parser.parse_args()
	main(args)