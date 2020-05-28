import sys
import os.path
import loaders
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import entropy
import datetime
from scipy.spatial.distance import euclidean
import joblib
import wordcloud
import turkishnlptool as tnlpt

# Read Config YAML file.
cfg  = loaders.configloader()
datafolder = cfg['data']['datafolder']
rawcsvloc = cfg['data']['rawcsv']
processedcsvloc = cfg['data']['processedcsv']
linkscsvloc = cfg['data']['linkscsv']
n_topic = cfg['parameters']['n_topic']
n_gram = cfg['parameters']['n_gram']
name = cfg['data']['name']
# Generate other variables from config.
n_gram_str = '_'.join([str(i) for i in n_gram])
modelname = f'{name}_{n_topic}topic_{n_gram_str}gram'

# Get full path.
rawloc = os.path.join(datafolder,rawcsvloc)
nlploc = os.path.join(datafolder,processedcsvloc)
linkloc = os.path.join(datafolder,linkscsvloc)
# Load Data.ç
if os.path.exists(nlploc): # If processed file exist, no processing required.
    print('Processed data file found. No processing is required anymore.')
    df = pd.read_csv(nlploc,index_col='id') # Load data as dataframe
    df.dropna(inplace=True)
    isProcessed = True
elif os.path.exists(rawloc):
    print('Raw data file found only, it is being loaded. You should call preprocess method.')
    df = pd.read_csv(rawloc,index_col='id') # Load data as dataframe
    df.dropna(inplace=True)
    isProcessed = False
else:
    raise ValueError('File not found !')

def eval_(x):
    if isinstance(x,str):
        return eval(x)
    else:
        return []
try:
    links = pd.read_csv(linkloc,index_col='id')
    links['linking_pages'] = links['linking_pages'].apply(eval_)
except:
    links = None
    print('Link info file is not found.')
# Count Raw word number.
df['wordnumber'] = df['text'].apply(lambda x: len(x.split()))
word_threshold = np.quantile(df.wordnumber,0.2)
df_red = df[df['wordnumber']>word_threshold] # Remove empty pages.
print('--- NaN Values and too short documents are dropped ---')
# Plot word distribution.
sns.distplot(df['wordnumber'].apply(np.log2), bins=100, kde=True, norm_hist=False)
plt.xlabel('log2(word count)')
plt.ylabel('probability')
plt.savefig('word_dist.png')
plt.show()

# Processing.
if not isProcessed:
    zmbrk = tnlpt.Zemberek()
# Utility function for preprocessing..
def docprocess(paragraph,concat=False):
    '''This function performs required preprocessings to the document.'''
    # Remove newline chaacters from the document
    paraflat = tnlpt.removeNewLine(paragraph)
    # Divide the paragraph into sentences.
    sent_divd1 = zmbrk.sentenceBoundary(paraflat)
    #
    sent_divd = list(map(tnlpt.removePunc,sent_divd1))
    #
    sent_lem = [] # Lemmatized sentences list.
    for tmp_sent in sent_divd:
        tmp_sent_lem = []
        stems,lemmas = zmbrk.sentenceDisambugation(tmp_sent)
        for i in range(0,len(lemmas)):
            if lemmas[i][0].isdigit():
                continue
            elif lemmas[i][0] == 'UNK' and not zmbrk.isStopWord(stems[i][0]):
                tmp_sent_lem.append(stems[i][-1]) # -1 last stem
            elif lemmas[i][0] != 'UNK' and not zmbrk.isStopWord(lemmas[i][0]):
                tmp_sent_lem.append(lemmas[i][-1]) # -1 last lemma
            else:
                continue
        if concat:
            sent_lem.append(' '.join(tmp_sent_lem))
        else:
            sent_lem.append(tmp_sent_lem)
    return sent_lem

# Processing begins here.
if not isProcessed:
    tqdm.pandas(desc="Progress information: ")
    print('--- Lemmatization started ---')
    pcs_fcn = lambda x: ' '.join(docprocess(x,True)) # Zemberek preprocesser returns list of sentences
    df['nlp_text'] = df['text'].progress_apply(pcs_fcn)
    print('--- Lemmatization done ---')
    df.to_csv(nlploc,index=False)
    print('--- Preprocessed file is saved ---')