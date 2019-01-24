import sys
sys.path.append('/home/sagarwal/projects/mmd/nlg-eval/')
from nlgeval import compute_metrics
import argparse

def main(args):
	metrics_dict = compute_metrics(hypothesis=args.pred_file, 
					references=[args.ref_file], 
					no_skipthoughts=True, no_glove=True)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-pred_file', type=str, help='hypothesis file')
	parser.add_argument('-ref_file', type=str, help='reference file')
	args = parser.parse_args()
	main(args)
