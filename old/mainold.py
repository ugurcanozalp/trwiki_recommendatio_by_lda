import sys
import os
import yaml
import loaders
import turkishnlptool as tnlpt

# Read Config YAML file.
cfg  = loaders.configloader()
datafolder = cfg['data']['datafolder']
rawcsvloc = cfg['data']['rawcsv']
processedcsvloc = cfg['data']['processedcsv']
n_topic = cfg['parameters']['n_topic']
n_gram = cfg['parameters']['n_gram']
word_doc_threshold = cfg['parameters']['word_doc_threshold']
n_gram_str = '_'.join([str(i) for i in n_gram])
name = cfg['data']['name']
modelname = f'{name}_{n_topic}topic_{n_gram_str}gram'

zmbrk = tnlpt.Zemberek()

import WikiSystem

MYWIKI = WikiSystem.Wiki(zmbrk, datafolder, rawcsvloc, processedcsvloc, word_doc_threshold, n_gram=n_gram, word_threshold=10)

# If processed file is not found, preprocess the data
if not MYWIKI.isProcessed:
    MYWIKI.preprocess()

# Now, either perform Topic Modeling via LDA or recover a model.

try:
    MYWIKI.recoverModel(modelname)
    print(str(n_topic)+' topic model is recovered..\n')
except:
    print(str(n_topic) + ' topic model is not available.\nNew model is being trained...\n')
    MYWIKI.count()
    MYWIKI.fitByLDA(n_topic=n_topic, alpha=1/n_topic, batch_size=1024, random_state=6, test_size=2)
    MYWIKI.saveModel(modelname)

# Recommender object.
MYRECOM = WikiSystem.Recommender(MYWIKI.df,MYWIKI.n_topic)

if __name__ == 'main':
    search_result = MYRECOM.searchPage('Orta Doğu Teknik Üniversitesi', True)
    MYRECOM.addToHistory(search_result.index)
    print('Lets get recommendations..')
    print(MYRECOM.recommendWrtHistory())
