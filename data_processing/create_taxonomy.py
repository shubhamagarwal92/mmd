import json
import argparse


def main(args):





# attribute_lexicons.txt
# ls | wc -l 
# 48
# all_colors.txt     card slots.txt  cuff.txt                 fit.txt             lapel.txt          occasion.txt  shape.txt          toe.txt
# band color.txt     care.txt        cut.txt                  frame color.txt     length.txt         pleats.txt    size.txt           type.txt
# band material.txt  clean.txt       dial color.txt           frame material.txt  lens material.txt  pockets.txt   sleeves.txt        upper material.txt
# belt-loops.txt     closure.txt     dial glass material.txt  handloom.txt        material.txt       print.txt     sole material.txt  warranty.txt
# brand.txt          collar.txt      dimension.txt            heel.txt            movement type.txt  rim.txt       style.txt          water resistance.txt
# button.txt         color.txt       display type.txt         jeans style.txt     neck.txt           rise.txt      tip.txt            weave.txt

# all_colors.txt
# amber   200


# styletip values
goes_with_file_path = 'goes_with_synset_per_synset_men_final.json'
style_file = open(goes_with_file_path,'r')
style_data = json.load(style_file)
synset = style_data.keys()
# 112
# [u'brogues', u'sling bag', u'thermal', u'goggles']
product = style_data[synset[0]].keys()
# [u'tuxedo', u'jeans', u'accessories', u'messenger bag']
value = style_data[synset[0]][product[0]]
# 20/182/10

goes_with_file_path = 'goes_with_synset_attribute_per_synset_men_final.json'
style_file = open(goes_with_file_path,'r')
style_data = json.load(style_file)
synset = style_data.keys()
# 115
# [u'military jacket', u'brogues', u'sling bag', u'thermal', u'goggles']
product = style_data[synset[0]].keys()
# [u'sunglasses', u'accessories', u'goggles']
value = style_data[synset[0]][product[4]]
# {u'print': [u'gothic'], u'style': [u'gothic'], u'type': [u'gothic'], u'brand': [u'gothic']}

goes_with_file_path = 'goes_with_synset_attribute_per_synset_attribute_men_final.json'
style_file = open(goes_with_file_path,'r')
style_data = json.load(style_file)
synset = style_data.keys()
# 115
# [u'military jacket', u'brogues', u'sling bag', u'thermal', u'goggles']
product = style_data[synset[0]].keys()
# [u'color', u'style', u'material']
value = style_data[synset[0]][product[4]].keys()
# {u'black': {u'apparel': {u'print': [u'gothic'], u'style': [u'gothic'], u'type': [u'gothic'], u'brand': [u'gothic']}}}



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-celeb_synset_file_path', type=str, default='./models/',
						help='path for celeb')
	parser.add_argument('-out_file_path', type=str, help='annoy path')
	parser.add_argument('-annoy_pkl_path', type=str, help='annoy pkl')	
	args = parser.parse_args()
	main(args)