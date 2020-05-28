import sys
import os.path
import loaders
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import entropy
from sklearn.model_selection import ShuffleSplit
import datetime
from scipy.spatial.distance import euclidean
import joblib
import wordcloud
# import turkishnlptool as tnlpt

class Wiki:
    '''This is main class for TR Wikipedia data. It performs text cleaning, numbering etc.
    Following creating the object, preprocess and cluster algorithms should be executed respectively.
    This class stores documents as data frame with 4 fields: id, title, url and text.
    '''
    # Text processing tool as static property.
    def __init__(self,zmbrk, datafolder, rawcsvloc, processedcsvloc, word_doc_threshold, n_gram=(1), word_threshold=10):
        '''TextData object is constructed by data file location. '''
        # Configuration
        self.zmbrk = zmbrk
        self.datafolder = datafolder
        self.rawloc = os.path.join(datafolder,rawcsvloc)
        self.nlploc = os.path.join(datafolder,processedcsvloc)
        self.word_doc_threshold = word_doc_threshold
        self.n_gram = n_gram
        self.isCounted = False
        # sys.path.append(cfg['libpath']) # Add extra libraries to the path
        # Check for preprocessed file.
        if os.path.exists(self.nlploc): # If processed file exist, no processing required.
            print('Processed data file found. No processing is required anymore.')
            self.df = pd.read_csv(self.nlploc) # Load data as dataframe
            self.df.dropna(inplace=True)
            self.isProcessed = True
        elif os.path.exists(self.rawloc):
            print('Raw data file found only, it is being loaded. You should call preprocess method.')
            self.df = pd.read_csv(self.rawloc) # Load data as dataframe
            self.df.dropna(inplace=True)
            self.isProcessed = False
        else:
            print('No file found.')
        #
        self.df['wordnumber'] = self.df['text'].apply(lambda x: len(x.split()))
        self.df = self.df[self.df['wordnumber']>word_threshold] # Remove empty pages.
        print('--- NaN Values and too short documents are dropped ---')
        # Plot word distribution.
        sns.distplot(self.df['wordnumber'].apply(np.log2), bins=100, kde=True, norm_hist=False)
        plt.xlabel('log2(word count)')
        plt.ylabel('probability')
        plt.savefig('word_dist.png')
        plt.show()
        return

    def preprocess(self):
        '''Preprocessing function.'''
        if self.isProcessed:
            key = input('Data seems to be preprocessed. If you want to continue, press y, else, press any other key.')
            if key != 'y':
                return
        # ELSE, PROCESS THE DATA.
        # Processing.
        tqdm.pandas(desc="Progress information: ")
        print('--- Lemmatization started ---')
        pcs_fcn = lambda x: ' '.join(self.zmbrk.preprocesser(x,True)) # Zemberek preprocesser returns list of sentences
        self.df['nlp_text'] = self.df['text'].progress_apply(pcs_fcn)
        print('--- Lemmatization done ---')
        self.df.to_csv(self.nlploc,index=False)
        print('--- Preprocessed file is saved ---')
        return

    def trainTestSplit(self, random_state=6,test_size=0.25):
        '''Perform Train Test Split on the data. '''
        if test_size == 0:
            self.train_idx = np.array(self.df.index)
            self.test_idx = np.array([])
        else:
            splitter = ShuffleSplit(n_splits=1, random_state=random_state, test_size=test_size, train_size=None)
            train_test_idx = list(splitter.split(self.df))
            self.train_idx = train_test_idx[0][0]
            self.valid_idx = train_test_idx[0][1]
        return

    def count(self):
        # Count Vectorizer for counting.
        min_word = self.word_doc_threshold
        ngram = tuple(self.n_gram)
        self.countVec = CountVectorizer(min_df=min_word, max_df=0.6,ngram_range = (1,2))
        self.countMatrix = self.countVec.fit_transform(self.df['nlp_text'])
        self.isCounted = True

    def fitByLDA(self,n_topic,alpha=None, batch_size = 2048, random_state=6, test_size=0):
        '''
        Perform Latent Dirichlet Allocation to cluster documents, and create topic distribution features.
        Call count method prior to this method
        '''
        if not self.isCounted: # are words counted ?
        	self.count()
        # Split data first..
        self.trainTestSplit(random_state=random_state,test_size=test_size)
        # Create history
        self.n_topic = n_topic
        # Create probabilistic model..
        if alpha is None: # If topic prior is not given, assign it as 1/k
            alpha = 1/n_topic
        print(f'Latent Dirichlet Allocation has been started for overall Corpus.')
        self.model = LatentDirichletAllocation(n_components=n_topic, random_state=random_state,doc_topic_prior=alpha, learning_method='online',batch_size=batch_size,verbose=1, n_jobs=-1)
        # Train the model
        t = time.time() # tic
        countMatrixTrain = self.countMatrix[self.train_idx]
        self.model.fit(countMatrixTrain)
        elapsed = time.time() - t # toc
        print(f'LDA is fit. Elapsed time: in {elapsed} seconds.')
        t = time.time()  # tic
        # Transform all data into topics
        print('Topics are being found for all pages and words...')
        Topics = self.model.transform(self.countMatrix)
        self.df['topic_dist'] = Topics.tolist() # DOCUMENT TOPIC DISTRIBUTIONS
        self.df['topic_dist'] = self.df['topic_dist'].apply(np.array)
        modelcomponents = self.model.components_
        self.beta_lda = modelcomponents/modelcomponents.sum(axis=1)[:, np.newaxis]
        elapsed = time.time() - t  # toc
        print(f'End topics transform: in {elapsed} seconds.')
        # Find dominant topics.
        print('Dominant topics are being found...')
        t = time.time()  # tic
        self.df['dominant_topic'] = self.df['topic_dist'].apply(np.argmax)
        elapsed = time.time() - t  # toc
        print(f'End find dominant topics: in {elapsed} seconds.')
        # Returning elements.
        return

    def saveModel(self,name):
        '''Save topic distribution if desired.'''
        print('Writing to Binary File on disk...')
        filename = os.path.join(self.datafolder,'models',f'{name}.joblib')
        joblib.dump([self.model, self.countVec, self.countMatrix, self.n_topic],filename)
        print(f'Saved to file: {filename}\n')
        return

    def recoverModel(self,name):
        '''Recover the model.'''
        filename = os.path.join(self.datafolder,'models',f'{name}.joblib')
        recoveredModel = joblib.load(filename)
        self.model = recoveredModel[0]
        self.countVec = recoveredModel[1]
        self.countMatrix = recoveredModel[2]
        self.n_topic = recoveredModel[3]
        #
        Topics = self.model.transform(self.countMatrix)
        self.df['topic_dist'] = Topics.tolist() #.apply(np.array)
        self.df['topic_dist'] = self.df['topic_dist'].apply(np.array)
        self.df['dominant_topic'] = self.df['topic_dist'].apply(np.argmax)
        self.beta_lda = self.model.components_/self.model.components_.sum(axis=1)[:, np.newaxis]

    def evalPerplexity(self,istest=False):
        '''Evaluate perplexity of the fit model by validation set or overall corpus.'''
        if istest:
            pplx = self.model.perplexity(self.countMatrix[self.valid_idx]) # Perplexity
        else:
            pplx = self.model.perplexity(self.countMatrix) # Perplexity
        return pplx

    def getEmbedding(self,paragraph):
        '''
        Get Embedding of a paragraph, and its sentences.
        '''
        sents = self.zmbrk.preprocesser(paragraph, True)
        sent_emb = self.countVec.transform(sents)
        #
        sent_vec = self.model.transform(sent_emb)
        return sent_vec

    # UTILITIES
    def wordDistByTopic(self):
        '''
        This function only works if LDA algorithm is run as priori.
        '''
        # Show words for each topics.
        # word indices with highest priority
        wi = list(map(np.argsort,-self.beta_lda))
        # wp = list(map(lambda x: , self.beta_lda))
        wordnum = []
        for i in range(0,len(self.beta_lda)):
            wordnum.append(sum(self.beta_lda[i]>0.005))

        for index, topic in enumerate(self.beta_lda) :
            print(f'MOST CHARACTERISTIC WORDS FOR TOPIC #{index}')
            # print([countVec_.get_feature_names()[i] for i in topic.argsort()[-10 :]])
            print([ self.countVec.get_feature_names()[i] for i in wi[index][0:wordnum[index]] ])
            print('\n')

    def topicWordCloud(self,topic_idx):
        '''
        This function creates wordcloud of a desired topic.
        '''
        # Show words for each topics.
        # word indices with highest priority
        wi = list(map(np.argsort,-self.beta_lda))
        # wp = list(map(lambda x: , self.beta_lda))
        wordnum = []
        for i in range(0,len(self.beta_lda)):
            wordnum.append(sum(self.beta_lda[i]>0.005))
        #
        keywords = ' '.join([self.countVec.get_feature_names()[i] for i in wi[topic_idx][0 :wordnum[topic_idx]]])
        print(keywords)
        topic_i_cloud = wordcloud.WordCloud(background_color="white").generate(keywords)
        # Display the generated image:
        plt.imshow(topic_i_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.savefig(os.path.append('wordclouds', f'wordcloud{topic_idx}.png'))
        plt.show()

    def somePages(self):
        '''Show some pages which is about mostly a specific topic.'''
        for i in range(0, self.n_topic) :
            print(f'SOME PAGES FOR TOPIC #{i}')
            topici = self.df['topic_dist'].apply(lambda x : x[i])
            pagesi = topici.sort_values(ascending=False)[:20]

            for k in pagesi.index :
                print(self.df['title'][k])
            print('\n')

    def searchPage(self,keyword,exact=False):
        '''Search for a page in page titles given keyword.'''
        isinfun = lambda title: keyword.lower() in title.lower()
        issame = lambda title: keyword.lower() == title.lower()
        if exact:
            results = self.df['title'].apply(issame)
        else:
            results = self.df['title'].apply(isinfun)
        out = self.df[results]
        
        for i,idx in enumerate(self.df[results].index):
            print(f'Page Number: {idx} - Page Name: {out["title"].loc[idx]}' )

        # print(f'Page Number: {self.df[results].index} ,\n Page Name: {out["title"].values}' )
        return out

    def visualize(self):
        '''Visualize word distribution on 2 dimensional space.'''
        from sklearn.manifold import TSNE
        mytsne = TSNE(n_components=2)
        embedded = mytsne.fit_transform(self.beta_lda.transpose())
        dominant_topic=self.beta_lda.argmax(axis=0).reshape([-1,1]).astype('float32')
        df_subset = pd.DataFrame(np.concatenate((embedded,dominant_topic),axis=1),columns=['dim1','dim2','dominant_topic'])

        plt.figure(figsize=(16,10))
        sns.scatterplot(
             x="dim1", y="dim2",
             hue="dominant_topic",
             palette=sns.color_palette("hls", 75),
             data=df_subset,
             legend=False,
             alpha=0.3,
         )
        plt.savefig('tsne_lda.png')
        return 0

class Recommender:    
    '''Main class for recommendations.'''
    def __init__(self,df,n_topic):
        '''
        Inputs	: 
        df 		: Must have fields : ['title','topic_dist','dominant_topic','Distance']
        '''
        self.df = df
        self.n_topic = n_topic
        self.clearHistory()
        return
    
    def searchPage(self,keyword,exact=False):
        '''Search for a page in page titles given keyword.'''
        isinfun = lambda title: keyword.lower() in title.lower()
        issame = lambda title: keyword.lower() == title.lower()
        if exact:
            results = self.df['title'].apply(issame)
        else:
            results = self.df['title'].apply(isinfun)
        out = self.df[results]
        
        for i,idx in enumerate(self.df[results].index):
            print(f'Page Number: {idx} - Page Name: {out["title"].loc[idx]}' )

        # print(f'Page Number: {self.df[results].index} ,\n Page Name: {out["title"].values}' )
        return out

    def addToHistory(self,pageIndex):
        '''Add a page to history.'''
        self.history.append(pageIndex)
        # Update history topic distribution.
        # self.history_dist = 0.5*self.history_dist + 0.5*np.array(self.df['topic_dist'].iloc[pageIndex])
        self.history_dist = self.df['topic_dist'].loc[pageIndex].mean()

    def clearHistory(self):
        '''This function clears visit history. For history distribution, equal probabilities are assigned to all topics.'''
        self.history = []
        self.history_dist = np.full(self.n_topic,1/self.n_topic) # Initial topic distribution.

    def recommendWrtHistory(self,distfun='KL'):
        '''Evaluate all pages distance to history. Pass distance function as string;
        KL: Kullback-Leibler Divergence.
        JS: Jensen-Shannon Divergence.
        HL: Hellinger Distance.
        '''
        if distfun=='KL':
            dist2hist = lambda p: entropy(self.history_dist,p)
        elif distfun=='JS':
            dist2hist = lambda p: entropy(self.history_dist,0.5*(self.history_dist + p)) + entropy(0.5*(self.history_dist + p),p)
        elif distfun=='HL':
            _SQRT2 = 1.41421356237
            dist2hist = lambda p: euclidean(np.sqrt(p),np.sqrt(self.history_dist))/ _SQRT2
        else:
            print('Wrong distance function')
            return

        recom_list = self.df
        recom_list['Distance'] = self.df['topic_dist'].apply(dist2hist)
        recom_list = recom_list[['title','topic_dist','dominant_topic','Distance']]
        recom_list.sort_values(by='Distance',inplace=True)
        return recom_list