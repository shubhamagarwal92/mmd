import re
import numpy as np
import json
import pandas as pd
import time
import os
import glob
import argparse
# import openpyxl for xls writer

def read_json_to_df(file_path):
	# df = pd.read_json(path_or_buf=file_path,orient='records',lines=True)
	df = pd.read_json(path_or_buf=file_path,orient='records')
	return df

def read_df_pickle(file_path):
	df = pd.read_pickle(path=file_path)
	return df

def get_column_stats(df,column_name,to_dict = False):
	if to_dict:
		return df[column_name].value_counts().to_dict()
	else: 
		return df[column_name].value_counts()

def main(args):
	# Stats
	# stats_file_pkl = 'stats_df_all.pkl'
	stats_df = read_df_pickle(args.stats_file_pkl)
	stats_describe = stats_df.describe()
	stats_sum = pd.DataFrame(stats_df.sum(numeric_only=True))
	stats_sum.columns = ['total']
	total_turns = stats_sum.at['total_turns','total'] # Value 
	stats_sum['percent'] = stats_sum/total_turns*100
	stats_sum['percent'] = stats_sum['percent'].astype(int)
	# System utterances
	# sys_file_pkl = 'combined_df_sys.json'
	writer = pd.ExcelWriter(args.output_file_path)
	stats_describe.to_excel(writer, sheet_name='Stats_desc')
	stats_sum.to_excel(writer, sheet_name='Stats_sum')

	sys_df = read_df_pickle(args.sys_file_pkl)
	sys_df = sys_df.loc[sys_df['is_nlg']>0]
	sys_df_numeric = sys_df.loc[sys_df['is_nlg']>0][['state', 'num_chars','num_sent', 'num_words']]	
	sys_df_describe = sys_df_numeric.groupby('state').describe()
	unique_nlg = sys_df.groupby(['state']).nlg.nunique().to_frame()
	unique_nlg.columns = ['total']
	# states = get_column_stats(sys_df, 'state').to_frame()
	# Num unique
	num_unique_states = sys_df['state'].nunique()
	num_unique_type = sys_df['type'].nunique()
	print("Sys Unique states" + str(num_unique_states))
	print("Sys Unique type" + str(num_unique_states))
	nlg_state_table = sys_df.groupby(['state','nlg']).size().sort_values(ascending=False).to_frame()
	nlg_state_table.columns = ['count']
	largest_nlg = nlg_state_table.groupby(['state'])['count'].nlargest(10)

	sys_df_describe.to_excel(writer, sheet_name='Sys_describe')
	unique_nlg.to_excel(writer, sheet_name='Sys_unique_count_nlg')
	largest_nlg.to_excel(writer, sheet_name='Sys_state_nlg')

	# User utterances
	# user_file_pkl = 'combined_df_user.json'
	user_df = read_df_pickle(args.user_file_pkl)
	user_df = user_df.loc[user_df['is_nlg']>0]
	user_df_numeric = user_df.loc[user_df['is_nlg']>0][['state', 'num_chars','num_sent', 'num_words']]	
	user_df_describe = user_df_numeric.groupby('state').describe()
	unique_nlg_user = user_df.groupby(['state']).nlg.nunique().to_frame()
	unique_nlg_user.columns = ['total']
	# states = get_column_stats(sys_df, 'state').to_frame()
	# Num unique
	num_unique_states = user_df['state'].nunique()
	num_unique_type = user_df['type'].nunique()
	print("User Unique states" + str(num_unique_states))
	print("User Unique type" + str(num_unique_states))
	nlg_state_table = user_df.groupby(['state','nlg']).size().sort_values(ascending=False).to_frame()
	nlg_state_table.columns = ['count']
	largest_nlg_user = nlg_state_table.groupby(['state'])['count'].nlargest(10)

	user_df_describe.to_excel(writer, sheet_name='User_describe')
	unique_nlg_user.to_excel(writer, sheet_name='User_unique_count_nlg')
	largest_nlg_user.to_excel(writer, sheet_name='User_state_nlg')
	writer.save()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-user_file_pkl', help='User file path')
	parser.add_argument('-sys_file_pkl', help='Sys file path')
	parser.add_argument('-stats_file_pkl', help='Output file path')
	parser.add_argument('-output_file_path', help='Output file path')
	args = parser.parse_args()
	main(args)
