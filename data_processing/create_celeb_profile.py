import json
import argparse
from pandas.io.json import json_normalize

celeb_synset_file_path = 'synset_distribution_over_celebrity_men.json'
celeb_file = open(celeb_synset_file_path,'r')
celeb_data = json.load(celeb_file)

celebs = celeb_data.keys()
#[u'cel_183', u'cel_2170']
synset = celeb_data[celebs[0]].keys()
# [u'denim shorts', u'dress socks', u'fedora']
attributes = celeb_data[celebs[0]][synset[0]].keys()
# [u'style', u'rise', u'fit', u'color']
sub_attr = celeb_data[celebs[0]][synset[0]][attributes[0]].keys()
# [u'urban', u'tunic', u'peplum', u'gorgeous']
value = celeb_data[celebs[0]][synset[0]][attributes[0]][sub_attr[0]]
# 0.00013578585357177885

cel_data = json_normalize(celeb_data)


synset_celeb_file_path = 'celebrity_distribution_over_synset_men.json'
synset_file = open(synset_celeb_file_path,'r')
synset_data = json.load(synset_file)

synset = synset_data.keys()
# 116
#[u'cel_183', u'cel_2170']
attributes = synset_data[synset[0]].keys()
# [u'denim shorts', u'dress socks', u'fedora']
sub_attr = synset_data[synset[0]][attributes[0]].keys()
# [u'style', u'rise', u'fit', u'color']
celebs = synset_data[synset[0]][attributes[0]][sub_attr[0]].keys()
# [u'urban', u'tunic', u'peplum', u'gorgeous']
value = synset_data[synset[0]][attributes[0]][sub_attr[0]][celebs[0]]
# 0.00013578585357177885




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-celeb_synset_file_path', type=str, default='./models/',
						help='path for celeb')
	parser.add_argument('-out_file_path', type=str, help='annoy path')
	parser.add_argument('-annoy_pkl_path', type=str, help='annoy pkl')	
	args = parser.parse_args()
	main(args)