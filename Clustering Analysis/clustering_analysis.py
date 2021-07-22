#!/usr/bin/env python
# coding: utf-8

# ## Customer segmentation

# A cluster is a collection of points in a dataset. These points are more similar between them than they are to points belonging to other clusters.
# Distance-based clustering groups the points into some number of clusters such that distances within the cluster should be small while distances between clusters should be large.

# ### Import modules requiered

# First of all, we need to import the required module. 

# In[10]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[11]:


import pandas as pd
import numpy as np
import sklearn


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# We'll work with two `scikit-learn` modules: `Kmeans` and `PCA`. They will allow us to perform a clustering algorithm and dimensionality reduction.

# In[13]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[14]:


from sklearn.preprocessing import MinMaxScaler


# In[15]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio


# In[8]:


import extra_graphs


# ### Read data into a DataFrame

# We read the basic data stored in the [customers.csv](https://www.kaggle.com/akram24/mall-customers) file into a `DataFrame` using pandas. 

# In[9]:


customers = pd.read_csv("customers.csv")


# We check the first five rows of the DataFrame. We can see that we have: CustumerID, Gender, Age, Annual Income expressed as price x1000, and the spending score as we expected.

# In[10]:


customers.head()


# ### Exploring the data

# First, we check that if there is any missing value in the dataset. K-means algorithm is not able to deal with missing values. 

# In[11]:


print(f"Missing values in each variable: \n{customers.isnull().sum()}")


# Fortunately, there is no missing data. We can also check if there are duplicated rows.

# In[12]:


print(f"Duplicated rows: {customers.duplicated().sum()}")


# Finally, we check how each variable is presented in the DataFrame. Categorical variables cannot be handled directly. K-means is based on distances. The approach for converting those variables depend on the type of categorical variables. 

# In[13]:


print(f"Variable:                  Type: \n{customers.dtypes}") 


# #### Descriptive statistics and Distribution.

# For the descriptive statistcs, we'll get mean, standard deviation, median and variance. If the variable is not numeric, we'll get the counts in each category.

# In[14]:


def statistics(variable):
    if variable.dtype == "int64" or variable.dtype == "float64":
        return pd.DataFrame([[variable.name, np.mean(variable), np.std(variable), np.median(variable), np.var(variable)]], 
                            columns = ["Variable", "Mean", "Standard Deviation", "Median", "Variance"]).set_index("Variable")
    else:
        return pd.DataFrame(variable.value_counts())


# In[15]:


def graph_histo(x):
    if x.dtype == "int64" or x.dtype == "float64":
        # Select size of bins by getting maximum and minimum and divide the substraction by 10
        size_bins = 10
        # Get the title by getting the name of the column
        title = x.name
        #Assign random colors to each graph
        color_kde = list(map(float, np.random.rand(3,)))
        color_bar = list(map(float, np.random.rand(3,)))

        # Plot the displot
        sns.distplot(x, bins=size_bins, kde_kws={"lw": 1.5, "alpha":0.8, "color":color_kde},
                       hist_kws={"linewidth": 1.5, "edgecolor": "grey",
                                "alpha": 0.4, "color":color_bar})
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14);
        plt.ylabel("Frequency", size=16, labelpad=15);
        # Customize title
        plt.title(title, size=18)
        # Customize grid and axes visibility
        plt.grid(False);
        plt.gca().spines["top"].set_visible(False);
        plt.gca().spines["right"].set_visible(False);
        plt.gca().spines["bottom"].set_visible(False);
        plt.gca().spines["left"].set_visible(False);   
    else:
        x = pd.DataFrame(x)
        # Plot       
        sns.catplot(x=x.columns[0], kind="count", palette="spring", data=x)
        # Customize title
        title = x.columns[0]
        plt.title(title, size=18)
        # Customize ticks and labels
        plt.xticks(size=14)
        plt.yticks(size=14);
        plt.xlabel("")
        plt.ylabel("Counts", size=16, labelpad=15);        
        # Customize grid and axes visibility
        plt.gca().spines["top"].set_visible(False);
        plt.gca().spines["right"].set_visible(False);
        plt.gca().spines["bottom"].set_visible(False);
        plt.gca().spines["left"].set_visible(False);


# We'll start by the **Spending Score**.

# In[16]:


spending = customers["Spending Score (1-100)"]


# In[17]:


statistics(spending)


# In[18]:


graph_histo(spending)


# Then, we'll check **Age**.

# In[19]:


age = customers["Age"]


# In[20]:


statistics(age)


# In[21]:


graph_histo(age)


# Finally, we'll explore **Annual Income** variable.

# In[22]:


income = customers["Annual Income (k$)"]


# In[23]:


statistics(income)


# In[24]:


graph_histo(income)


# In[25]:


gender = customers["Gender"]


# In[26]:


statistics(gender)


# In[27]:


graph_histo(gender)


# #### Correlation between parameteres

# Also, we will analyze the correlation between the numeric parameters. For that aim, we'll use the `pairplot` seaborn function. We want to see whether there is a difference between gender. So, we are going to set the `hue` parameter to get different colors for points belonging to female or customers.

# In[28]:


sns.pairplot(customers, x_vars = ["Age", "Annual Income (k$)", "Spending Score (1-100)"], 
               y_vars = ["Age", "Annual Income (k$)", "Spending Score (1-100)"], 
               hue = "Gender", 
               kind= "scatter",
               palette = "YlGnBu",
               height = 2,
               plot_kws={"s": 35, "alpha": 0.8});


# ### Dimensionality reduction

# Applying Principal Component Analysis (PCA) to discover which dimensions best maximize the variance of features involved.

# #### Principal Component Analysis (PCA)

# First, we'll transform the categorical variable into two binary variables.

# In[29]:


customers["Male"] = customers.Gender.apply(lambda x: 0 if x == "Male" else 1)


# In[30]:


customers["Female"] = customers.Gender.apply(lambda x: 0 if x == "Female" else 1)


# Customer ID is not a useful feature. Gender will split it into two binaries categories. It should not appear in the final dataset

# In[31]:


X = customers.iloc[:, 2:]


# In[32]:


X.head()


# In[33]:


# Apply PCA and fit the features selected
pca = PCA(n_components=2).fit(X)


# During the fitting process, the model learns some quantities from the data: the "components" and "explained variance".

# In[34]:


print(pca.components_)


# In[35]:


print(pca.explained_variance_)


# The components define the direction of the vector while the explained variance define the squared-length of the vector.
# 

# In[36]:


# Transform samples using the PCA fit
pca_2d = pca.transform(X)


# Now, representing using a type of scatter plot called, Biplot

# In[37]:


extra_graphs.biplot(pca_2d[:,0:2], np.transpose(pca.components_[0:2, :]), labels=X.columns)


# We can observe that Annual Income as well as Spending Score at the two most important components.

# ### K-means clustering 

# In order to cluster data, we need to determine how to tell if two data points are similar. 
# 
# So if the value is large, the points are very similar. If the value is small, the points are similar. 
# 

# ${\sqrt{\sum_{i=1}^n (x_i-y_i)^2}}$

# First, we need to fix the numbers of clusters to use. 

# There are several direct methods to perform this. Among them, we find the elbow and silhouette methods.

# We'll consider the total intra-cluster variation (or total within-cluster sum of square (WSS)). The goal is to minimize WSS.

# The Elbow method looks at how the total WSS varies with the number of clusters. 
# For that, we'll compute k-means for a range of different values of k. Then, we calculate the total WSS. We plot the curve WSS vs. number of clusters. 
# Finally, we locate the elbow or bend of the plot. This point is considered to be the appropriate number of clusters.

# In[38]:


wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss, c="#c51b7d")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.title('Elbow Method', size=14)
plt.xlabel('Number of clusters', size=12)
plt.ylabel('wcss', size=14)
plt.show() 


# In[65]:


# Kmeans algorithm
# n_clusters: Number of clusters. In our case 5
# init: k-means++. Smart initialization
# max_iter: Maximum number of iterations of the k-means algorithm for a single run
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. 
# random_state: Determines random number generation for centroid initialization.
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=10, n_init=10, random_state=0)

# Fit and predict 
y_means = kmeans.fit_predict(X)


# Now, let's check how our clusters look like:

# In[1]:


fig, ax = plt.subplots(figsize = (8, 6))

plt.scatter(pca_2d[:, 0], pca_2d[:, 1],
            c=y_means, 
            edgecolor="none", 
            cmap=plt.cm.get_cmap("Spectral_r", 5),
            alpha=0.5)
        
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)

plt.xticks(size=12)
plt.yticks(size=12)

plt.xlabel("Component 1", size = 14, labelpad=10)
plt.ylabel("Component 2", size = 14, labelpad=10)

plt.title('Domains grouped in 5 clusters', size=16)


plt.colorbar(ticks=[0, 1, 2, 3, 4]);

plt.show()


# In[102]:


centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ["Age", "Annual Income", "Spending", "Male", "Female"])


# In[103]:


centroids.index_name = "ClusterID"


# In[105]:


centroids["ClusterID"] = centroids.index
centroids = centroids.reset_index(drop=True)


# In[106]:


centroids


# The most important features appear to be Annual Income and Spending score. 
# We have people whose income is low but spend in the same range - segment 0. People whose earnings a high and spend a lot - segment 1. Customers whose income is middle range but also spend at the same level - segment 2. 
# Then we have customers whose income is very high but they have most spendings - segment 4. And last, people whose earnings are little but they spend a lot- segment 5.

# Imagine that tomorrow we have a new member. And we want to know which segment that person belongs. We can predict this.

# In[17]:


X_new = np.array([[43, 76, 56, 0, 1]]) 
 
new_customer = kmeans.predict(X_new)
print(f"The new customer belongs to segment {new_customer[0]}")


# In[ ]:




