import numpy as np
from sklearn.manifold import TSNE

import plotly.offline as plt
import plotly.graph_objs as go

def example():
	X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
	X_embedded = TSNE(n_components=2).fit_transform(X)
	print(X_embedded.shape)
	print(X_embedded)

	import matplotlib.pyplot as plt
	import seaborn as sns
	palette = sns.color_palette("bright", 4)
	sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full', palette=palette)
	plt.show()

def tsne_plot(word_vectors):
	# PCA (optional)

	words = list(word_vectors.keys())
	X = np.array([word_vectors[word] for word in words])
	X_embedded = TSNE(n_components=2).fit_transform(X) # Nx2
	fig = go.Figure(data=go.Scatter(x=X_embedded[:,0],
									y=X_embedded[:,1],
									mode='markers+text',
									text=words,
									textposition='bottom center',
									hoverinfo="text")) # hover text goes here
	fig.update_layout(title='Evaluation Words')
	fig.write_image("fig1.png")

word_vectors = {"dog": np.array([1, 0, 0, 0]), 
"cat": np.array([0.9, 0.1, 0, 0]), 
"tree": np.array([0, 0, 1, 0]),
"human": np.array([1.5, 0.05, 0.03, 0.02])}
tsne_plot(word_vectors)


