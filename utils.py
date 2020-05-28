import os
import wordcloud
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
# UTILITIES
def wordDistByTopic(beta_lda, countVec):
    '''
    This function only works if LDA algorithm is run as priori.
    '''
    # Show words for each topics.
    # word indices with highest priority
    wi = list(map(np.argsort,-beta_lda))
    # wp = list(map(lambda x: , self.beta_lda))
    wordnum = []
    for i in range(0,len(beta_lda)):
        wordnum.append(sum(beta_lda[i]>0.005))
    for index, topic in enumerate(beta_lda) :
        print(f'MOST CHARACTERISTIC WORDS FOR TOPIC #{index}')
        # print([countVec_.get_feature_names()[i] for i in topic.argsort()[-10 :]])
        print([ countVec.get_feature_names()[i] for i in wi[index][0:wordnum[index]] ])
        print('\n')

def topicWordCloud(beta_lda,countVec,topic_idx):
    '''
    This function creates wordcloud of a desired topic.
    '''
    # Show words for each topics.
    # word indices with highest priority
    wi = list(map(np.argsort,-beta_lda))
    # wp = list(map(lambda x: , self.beta_lda))
    wordnum = []
    for i in range(0,len(beta_lda)):
        wordnum.append(sum(beta_lda[i]>0.005))
    #
    keywords = ' '.join([countVec.get_feature_names()[i] for i in wi[topic_idx][0 :wordnum[topic_idx]]])
    print(keywords)
    topic_i_cloud = wordcloud.WordCloud(background_color="white").generate(keywords)
    # Display the generated image:
    plt.imshow(topic_i_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(os.path.join('wordclouds', f'wordcloud{topic_idx}.png'))
    plt.show()

def somePages(df, topic_dist, n_topic):
    '''Show some pages which is about mostly a specific topic.'''
    for i in range(0, n_topic) :
        print(f'SOME PAGES FOR TOPIC #{i}')
        topici = topic_dist[:,i]
        pagesi = np.argsort(-topici)[:20]
        print(df['title'].iloc[pagesi])

def visualize(beta_lda):
    '''Visualize word distribution on 2 dimensional space.'''
    mytsne = TSNE(n_components=2)
    embedded = mytsne.fit_transform(beta_lda.transpose())
    dominant_topic=beta_lda.argmax(axis=0).reshape([-1,1]).astype('float32')
    df_subset = pd.DataFrame(np.concatenate((embedded,dominant_topic),axis=1),columns=['dim1','dim2','dominant_topic'])
    plt.figure(figsize=(16,10))
    sns.scatterplot(x="dim1", y="dim2",hue="dominant_topic", palette=sns.color_palette("hls", beta_lda.shape[0]), data=df_subset, legend=False, alpha=0.3)
    plt.savefig('tsne_lda.png')
    return 0