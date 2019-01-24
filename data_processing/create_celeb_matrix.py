import json
import argparse
import pandas as pd
import pickle as pkl
import os

def main(args):
	celeb_synset_file_path = os.path.join(data_dir, 'synset_distribution_over_celebrity_men.json')
	celeb_file = open(celeb_synset_file_path,'r')
	celeb_data = json.load(celeb_file)
	celebs = celeb_data.keys()
	celeb_dic={}
	for celeb in celebs:
		synsets = celeb_data[celeb].keys()
		for synset in synsets:
			attributes = celeb_data[celeb][synset].keys()
			for attribute in attributes:
				sub_attrs = celeb_data[celeb][synset][attribute].keys()
				for sub_attr in sub_attrs:
					value = celeb_data[celeb][synset][attribute][sub_attr]
					key = '|'.join([celeb, synset, attribute, sub_attr])
					celeb_dic[key] = value

	celeb_df = pd.DataFrame.from_dict(celeb_dic, orient='index').reset_index()
	celeb_df.columns = ['temp', 'values']
	temp_df = celeb_df['temp'].str.split('|',expand=True)
	temp_df.columns = ['celeb','synset','attribute','sub_attr']
	celeb_df = celeb_df.join(temp_df).drop('temp', axis=1)
	# df = pd.io.json.json_normalize(celeb_data)
	# df = pd.DataFrame(celeb_df.row.str.split('|',1).tolist(),
	#                                    columns = ['celeb','synset','attribute','sub_attr'])
	synset_celeb_file_path = os.path.join(data_dir, 'celebrity_distribution_over_synset_men.json')
	synset_file = open(synset_celeb_file_path,'r')
	synset_data = json.load(synset_file)
	synsets = synset_data.keys()
	synset_dic={}
	for synset in synsets:
		attributes = synset_data[synset].keys()
		for attribute in attributes:
			sub_attrs = synset_data[synset][attribute].keys()
			for sub_attr in sub_attrs:
				celebs = synset_data[synset][attribute][sub_attr].keys()
				for celeb in celebs:
					value = synset_data[synset][attribute][sub_attr][celeb]
					key = '|'.join([synset, attribute, sub_attr, celeb])
					synset_dic[key] = value

	synset_df = pd.DataFrame.from_dict(synset_dic, orient='index').reset_index()
	synset_df.columns = ['temp', 'values']
	temp_df = synset_df['temp'].str.split('|',expand=True)
	temp_df.columns = ['synset','attribute','sub_attr','celeb']
	synset_df = synset_df.join(temp_df).drop('temp', axis=1)

	celeb_df_file = os.path.join(out_dir, 'celeb_df.pkl')
	with open(celeb_df_file, 'wb') as f:
		pkl.dump(celeb_df, f, protocol=pkl.HIGHEST_PROTOCOL)
	synset_df_file = os.path.join(out_dir, 'synset_df.pkl')
	with open(synset_df_file, 'wb') as f:
		pkl.dump(synset_df, f, protocol=pkl.HIGHEST_PROTOCOL)
# synset_df[synset_df['celeb']=='cel_432']

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, default='./',
						help='data dir')
	parser.add_argument('-out_dir', type=str, help='out dir')
	args = parser.parse_args()
	main(args)