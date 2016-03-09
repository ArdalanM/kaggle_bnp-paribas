import pickle, argparse


parser = argparse.ArgumentParser()
parser.add_argument("--pickle",default="bloggender",help="pickle file")
parser.add_argument("--key",default="all",help="pickle key to select")
args = parser.parse_args()


# CODE_FOLDER = "/home/ardalan/Documents/iosquare/moderation_match_meetic/"


# filename = CODE_FOLDER + 'data/4_diclogs/meetic_best_NN_30000feats_CV10_(tr-acc|te-acc|roc)=(0.963|0.866|0.938).p'
filename = args.pickle
key = args.key

print(filename)
print(key)

diclog = pickle.load(open(filename, 'rb'))

if key == 'all': print(diclog)
else: print(diclog[key])
