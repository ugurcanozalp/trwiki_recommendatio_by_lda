import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

class Recommender:    
    '''Main class for recommendations.'''
    def __init__(self,df,n_topic,topic_dist,linkinfo=None):
        '''
        Inputs	: 
        df 		: Must have fields : ['title','topic_dist','dominant_topic','Distance']
        '''
        self.df = df
        self.n_topic = n_topic
        self.topic_dist_df = pd.DataFrame(topic_dist,index=self.df.index)
        self.topic_dist = topic_dist
        self.dominant_topic_df = pd.DataFrame(np.argmax(topic_dist,axis=1),index=self.df.index)
        self.linkinfo = linkinfo
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
        self.history = np.concatenate((self.history,pageIndex))
        # Update history topic distribution.
        # self.history_dist = 0.5*self.history_dist + 0.5*np.array(self.df['topic_dist'].iloc[pageIndex])
        self.history_dist = self.topic_dist_df.loc[pageIndex].mean(axis=0).to_numpy()

    def clearHistory(self):
        '''This function clears visit history. For history distribution, equal probabilities are assigned to all topics.'''
        self.history = np.array([], dtype='int64') 
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

        recom_list = self.df[['title']]
        distance = np.apply_along_axis(dist2hist,1,self.topic_dist)
        recom_list['distance'] = distance
        recom_list.sort_values(by='distance',inplace=True)
        return recom_list

    def evaluateRecommendation(self, firsts):
        if self.linkinfo is None:
            print('Since link info is not available, recommendations cannot be evaluated.')
            return
        rcm_idx = set(firsts.index)
        # First order desired pages
        des_idx = set(self.linkinfo.loc[self.history[0]]['linking_pages']) 
        # Second order desired pages
        des_idx2 = set.union(*self.linkinfo.loc[des_idx]['linking_pages'].apply(set)).difference(des_idx)
        # Third order desired pages
        des_idx3 = set.union(*self.linkinfo.loc[des_idx2]['linking_pages'].apply(set)).difference(des_idx2.union(des_idx))
        # Undesired pages (4th order and higher)
        #undes4 = des_idx.union(des_idx2).union(des_idx3) 
        firstorder = rcm_idx.intersection(des_idx) # How much of recommendations are in 1st order desired pages ?
        secondorder = rcm_idx.intersection(des_idx2) # How much of recommendations are in 2nd order desired pages ?
        thirdorder = rcm_idx.intersection(des_idx3) # How much of recommendations are in 3rd order desired pages ?
        #forthandhigherorder = rcm_idx.intersection(undes4) # How much of recommendations are in 4nd and more order desired pages ?
        #
        fo = [len(firstorder)/len(rcm_idx), len(des_idx)]        
        so = [len(secondorder)/len(rcm_idx), len(des_idx2)]
        to = [len(thirdorder)/len(rcm_idx), len(des_idx3)]
        other = [1-fo[0]-so[0]-to[0], len(self.linkinfo)-len(des_idx)-len(des_idx2)-len(des_idx3)]

        print(f'First order recommendation accuracy: {fo[0]}, from {fo[1]} linked pages.')
        print(f'Second order recommendation accuracy: {so[0]}, from {so[1]} linked pages.')
        print(f'Third order recommendation accuracy: {to[0]}, from {to[1]} linked pages.')
        print(f'Other Recommendation accuracy: {other[0]}, from {other[1]} linked pages.')
        return fo, so, to, other

