import numpy as np
import handle_data as H
import train
import utils
from recommender import Recommender

try: # If model is already available in disk.
    model,countVec,countMatrix,n_topic = train.recoverModel(H.modelname,H.datafolder)
    beta_lda = model.components_/model.components_.sum(axis=1)[:, np.newaxis]
except: 
	n_topic = H.n_topic
	#train_idx, valid_idx = train.splitData(H.df_red,0.1)
	countVec, countMatrix = train.count(H.df_red,[1,2], 0.002,0.6)
	model,beta_lda = train.fitLDA(countMatrix,n_topic=H.n_topic,random_state=6,alpha=None)
	train.saveModel(H.modelname,H.datafolder,model,countVec,countMatrix,H.n_topic)

topic_dist, dominant_topic = train.transformer(countMatrix, model)



if __name__ == 'main':
    # Recommender object.
	R = Recommender(H.df_red,n_topic,topic_dist,H.links)
    R.clearHistory() # R.history = np.array([],dtype='int64')
    # search_result = R.searchPage('Orta Doğu Teknik Üniversitesi', True)
    # search_result = R.searchPage('Makine Öğrenimi', True)
    # search_result = R.searchPage('Suriye', True)
    # search_result = R.searchPage('Python (programlama dili)', True)
    # search_result = R.searchPage('General Dynamics F-16 Fighting Falcon', True)
	# search_result = R.searchPage('Kadeş Antlaşması', True)
	# search_result = R.searchPage('Cengiz Han', True)
    search_result = R.searchPage('Wolfgang Amadeus Mozart', True)
    R.addToHistory(search_result.index.to_numpy())
    print('Lets get recommendations..')
    rcm = R.recommendWrtHistory()
    R.evaluateRecommendation(rcm.head(20))
