import handle_data as H
import train
import utils
import matplotlib.pyplot as plt

train_idx, valid_idx = train.splitData(H.df_red,0.1)
countVec, countMatrix = train.count(H.df_red,[1,2], 0.002,0.6)
print(f'Shape of count matrix is: {countMatrix.shape}.')
test_pplx = []
n_topic_list = [20,40,60,80,100,120,140]

for n_topic in n_topic_list:
	model,beta_lda = train.fitLDA(countMatrix[train_idx],n_topic=n_topic,random_state=6,alpha=None)
	pplx = model.perplexity(countMatrix[valid_idx])
	print(f'Perplexity for {n_topic} topic model: {pplx}')
	test_pplx.append(pplx)

fig, axes = plt.subplots(1,1, figsize = (12,8))
axes.plot(n_topic_list[:10],test_pplx[:10],marker='o',ls='--',linewidth=2,markersize=5)
axes.grid(True)
axes.set_title('Perplexity vs Topic Number')
axes.set_xlabel('Topic Number')
axes.set_ylabel('Perplexity')
plt.savefig('pplx_vs_topic.png')
plt.show()