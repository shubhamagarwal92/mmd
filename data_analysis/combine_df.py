import re
import numpy as np
from pandas.io.json import json_normalize
import json
import pandas as pd
import time
import os
import glob
import argparse
import string
import unicodedata

def read_json_to_df(file_path):
	# df = pd.read_json(path_or_buf=file_path,orient='records',lines=True)
	df = pd.read_json(path_or_buf=file_path,orient='records')
	return df

def flatten_json_column(df,col_name='utterance'):
	temp_df = json_normalize(df[col_name].tolist())
	df.reset_index(drop=True,inplace=True)
	df = df.join(temp_df).drop(col_name, axis=1)
	return df

def get_column_stats(df,column_name,to_dict = False):
	if to_dict:
		return df[column_name].value_counts().to_dict()
	else: 
		return df[column_name].value_counts()

def findFiles(path): 
	return glob.glob(path)

def get_column_names(df):
	return df.columns.values

def get_value_row_column(df,index,column_name):
	return df.get_value(index,column_name)

def flatten_dic_column(df,col_name):
	df_new= pd.concat([df.drop([col_name], axis=1), df[col_name].apply(pd.Series)], axis=1)
	return df_new	

def append_df(df, df_to_append, ignore_index=True):
	new_df = df.append(df_to_append,ignore_index=ignore_index)
	return new_df

def write_df_to_csv(df,outputFilePath):
	df.to_csv(outputFilePath, sep=str('\t'),quotechar=str('"'), index=False, header=True)

def write_df_to_json(df,outputFilePath):
	df.to_json(path_or_buf=outputFilePath,orient='records',lines=True)

def save_df_pickle(df,output_file):
	df.to_pickle(output_file)

def get_unique_column_values(df,col_name):
	""" Returns unique values """
	return df[col_name].unique()

def count_unique(df, col_name):
	""" Count unique values in a df column """
	count = df[col_name].nunique()
	return count

def nlp_stats(nlg):
	nlg = nlg.encode('ascii',errors='ignore').lower().strip().strip('.?!') #Remove last '.?!'
	stats = {}
	stats['num_chars'] = len(nlg)
	sentences = nlg.replace('?','.').replace('!',' ').split('.') 
	#  replace '? and !' with '.'' ## [:-1] to discard last?? 
	stats['num_sent'] = len(sentences)
	words = nlg.replace('.','').replace('!','').replace('?','').split()
	stats['num_words'] = len(words)
	return stats


def main(args):
	all_files = glob.glob(args.file_dir + "/*.json")
	start = time.time()
	stats_df = pd.DataFrame()
	global_df = pd.DataFrame()
	global_user_df = pd.DataFrame()
	global_system_df = pd.DataFrame()
	print("Reading files")
	index = 0
	for dialogue_json in all_files:
		index+=1
		df = read_json_to_df(dialogue_json)
		df_flatten = flatten_json_column(df)
		df_flatten = df_flatten[['speaker','type', 'question-type','question-subtype','nlg','images']]
		# Assign filename
		df_flatten = df_flatten.assign(filename=dialogue_json)
		# Analysis
		df_flatten['num_images']=df_flatten['images'].apply(lambda x: len(x) if (type(x) is list) else None)
		# replace nans; create new df
		df = df_flatten.replace(np.nan, '', regex=True)
		# create state column
		df_flatten['state'] = df[['type', 'question-type','question-subtype']].apply(lambda x: ','.join(x), axis=1)
		df_flatten['nlp_stats'] = df_flatten['nlg'].apply(lambda x: nlp_stats(x) if (type(x) is unicode) else None)
		df_flatten = flatten_dic_column(df_flatten,'nlp_stats')
		# Flags
		df_flatten['is_image']=df_flatten['images'].apply(lambda x: 1 if (type(x) is list) else 0)
		df_flatten['is_nlg'] = df_flatten['nlg'].apply(lambda x: 1 if (type(x) is unicode) else 0)
		df_flatten['is_multimodal'] = df_flatten['is_nlg'] + df_flatten['is_image'] -1 # text + image -1
		# Subset
		user_df = df_flatten.loc[df_flatten['speaker'] == 'user']
		system_df = df_flatten.loc[df_flatten['speaker'] == 'system']
		# Analytics
		image_turns = df_flatten['is_image'].sum()
		nlg_turns = df_flatten['is_nlg'].sum()
		multimodal_turns = df_flatten['is_multimodal'].sum()
		total_turns = df_flatten.shape[0]
		user_turns = user_df.shape[0]
		sys_turns = system_df.shape[0]
		user_nlg_turns = user_df['is_nlg'].sum()
		sys_nlg_turns = system_df['is_nlg'].sum()
		# summarized utterance df
		local_data = {'filename':dialogue_json, 'total_turns':total_turns, 'image_turns':image_turns,
					'nlg_turns':nlg_turns, 'multimodal_turns':multimodal_turns, 'user_turns':user_turns,
					'sys_turns':sys_turns, 'user_nlg_turns':user_nlg_turns, 'sys_nlg_turns':sys_nlg_turns}
		local_df = pd.DataFrame(data=local_data, index=[index])
		# Append DF
		stats_df = append_df(stats_df,local_df,ignore_index=False)
		global_df = append_df(global_df, df_flatten)
		global_user_df = append_df(global_user_df, user_df)
		global_system_df = append_df(global_system_df, system_df)
	print("Writing files")
	write_df_to_json(global_df, args.output_file_json)
	save_df_pickle(global_df, args.output_file_pkl)
	write_df_to_json(global_user_df, args.output_user_file_json)
	save_df_pickle(global_user_df, args.output_user_file_pkl)
	write_df_to_json(global_system_df, args.output_sys_file_json)
	save_df_pickle(global_system_df, args.output_sys_file_pkl)
	write_df_to_json(stats_df, args.stats_file_json)
	save_df_pickle(stats_df, args.stats_file_pkl)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-file_dir', help='Input file directory path')
	parser.add_argument('-output_file_json', help='Output file path')
	parser.add_argument('-output_file_pkl', help='Output file path')
	parser.add_argument('-output_user_file_json', help='Output file path')
	parser.add_argument('-output_user_file_pkl', help='Output file path')
	parser.add_argument('-output_sys_file_json', help='Output file path')
	parser.add_argument('-output_sys_file_pkl', help='Output file path')
	parser.add_argument('-stats_file_json', help='Output file path')
	parser.add_argument('-stats_file_pkl', help='Output file path')	
	args = parser.parse_args()
	main(args)

