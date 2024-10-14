# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:19:25 2024

@author: aniket 
"""
'''
1] Use PCA to compress images from the sklearn digits dataset. Reconstruct
   the images using fewer principal components and compare the results.
   Goals:
1. Apply PCA to reduce the dimensionality of the digit images.
2. Reconstruct the compressed images using a selected number of components.
3. Compare the visual quality of the reconstructed images with the original
   ones using a side-by-side plot.'''
  
  
 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np


digits = load_digits()

X = digits.data  

n_components = 20

pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X)

X_reconstructed = pca.inverse_transform(X_pca)
 

 
 
 
 
 
 
 
 
'''
 2] Perform PCA on the iris dataset to understand how much variance is
explained by each principal component. Additionally, you must determine
how many principal components are required to capture at least 95% of
the total variance in the data.
Goals:
1. Apply PCA to the standardized Iris dataset.
2. Calculate and plot the cumulative explained variance for each principal component.
3. Identify the minimum number of components needed to explain 95% of the variance.
4. Visualize this with a plot that shows the cumulative explained variance.

 '''
 
 
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.datasets import load_iris 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 

iris = load_iris()
X = iris.data


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_

cumulative_explained_variance = np.cumsum(explained_variance_ratio)

n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

plt.figure(figsize=(8,6))
plt.plot(np.arange(1, len(cumulative_explained_variance)+1), cumulative_explained_variance, marker='o', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.axvline(x=n_components_95, color='r', linestyle='-')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

n_components_95, cumulative_explained_variance

 

'''
3. Perform Singular Value Decomposition (SVD) on a randomly generated
matrix and verify that the original matrix can be reconstructed using the
product of the decomposed matrices.
Goals:
1. Generate a random matrix of size 5x5.
2. Perform SVD on this matrix to obtain the U, Σ (singular values), and Vᵀ matrices.
3. Reconstruct the original matrix using the decomposed matrices.
4. Compare the original and reconstructed matrices and compute the difference.

'''
import numpy as np

np.random.seed(42)
original_matrix = np.random.rand(5, 5)

U, S, Vt = np.linalg.svd(original_matrix)

S_matrix = np.diag(S)


reconstructed_matrix = np.dot(U, np.dot(S_matrix, Vt))

difference_matrix = original_matrix - reconstructed_matrix
difference_norm = np.linalg.norm(difference_matrix)

original_matrix, reconstructed_matrix, difference_matrix, difference_norm






'''
4. Apply both PCA and SVD for dimensionality reduction on the Breast
Cancer dataset, and compare the results in terms of variance explained
and reconstruction accuracy.
Goals:
1 Apply PCA to the standardized dataset and reduce it to 5 components.
2 Perform SVD on the same dataset, also reducing it to 5 components.
3 Compare the explained variance for PCA and the reconstruction accuracy 
for both methods.
4 Calculate and report the reconstruction error (mean squared error) for
 both methods.
'''




'''
5] You are given a small collection of text documents. Use TruncatedSVD on
a TF-IDF matrix derived from a small set of text documents to reduce the
dimensionality of the matrix and reconstruct it.
Goals:
1 Convert a set of text documents into a TF-IDF matrix using TfidfVectorizer.
2 Apply TruncatedSVD to reduce the dimensionality of the TF-IDF matrix to 2 components.
3 Reconstruct the original TF-IDF matrix from the reduced representation.
4 Compare the reconstructed matrix with the original one to assess how much information was
retained.


'''





import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

doc = ["The quick brown fox jumps over the lazy dog",
       "Never jump over the lazy dog quickly",
       "Brown foxes are quick and dogs are lazy",
       "The dog is quick and brown"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(doc)

svd = TruncatedSVD(n_components=2)
reduced_matrix = svd.fit_transform(tfidf_matrix)

reconstructed_matrix = svd.inverse_transform(reduced_matrix)

mse = mean_squared_error(tfidf_matrix.toarray(), reconstructed_matrix)
print("Mean Squared Error between original and reconstructed matrix:", mse)

