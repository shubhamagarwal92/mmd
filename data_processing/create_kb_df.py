import os
import pickle as pkl
import argparse
import pandas as pd

def main(args):
	data = pkl.load(open('search_criteria_val.pkl','r'))
	df = pd.DataFrame.from_dict(data, orient='index').reset_index()
	df.columns = ['kb','count']
	# temp_df = df[colname].str.split(delimiter, expand=True)
	temp_df = df['kb'].apply(lambda x: x.split('|||'))
	df3 = pd.DataFrame(temp_df.values.tolist(), columns=['a','b','c'])
	temp_df.columns = ['criteria','sub_criteria','val']
	df = df.join(temp_df).drop(colname, axis=1)
	with open(filename, 'wb') as f:
		pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, help='dir path')
	parser.add_argument('-data_type', type=str, help='dir path')
	args = parser.parse_args()
	data_dir = os.path.join(args.data_dir, args.data_type)
	main(args)