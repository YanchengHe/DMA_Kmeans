import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import datetime

pd.set_option('display.max_columns', 25)


# ------ Define functions ------
def run_kmeans(n_clusters_f, init_f, df_f):
    ##### Complete this function
    # This function should at least take a dataframe as an argument. I have suggested additional arguments you may
    # want to provide, but these can be changed as you need to fit your solution.
    # The output of this function should be the input data frame will the model object KMeans and a data summary. The
    # function will need to add an additional column to the input dataframe called 'predict_cluster_kmeans'
    # that contains the cluster labels assigned by the algorithm.

    k_means_model_f = KMeans(n_clusters=n_clusters_f, init=init_f).fit_predict(df_f)
    k_means_model_f = pd.DataFrame(k_means_model_f)
    k_means_model_f['123'] = df_f.index
    k_means_model_f = k_means_model_f.set_index('123', drop=True, append=False, inplace=False, verify_integrity=False)
    k_means_model_f.columns = ['predict_cluster_kmeans']

    df_f = pd.concat([df_f, k_means_model_f], axis=1, join='outer')


    # summarize cluster attributes
    k_means_model_f_summary = df_f.groupby('predict_cluster_kmeans').agg(attribute_summary_method_dict)
    return k_means_model_f, k_means_model_f_summary

# ------ Import data ------
df_transactions = pd.read_csv('transactions_n100000.csv')

# ------ Engineer features -----
# --- convert from long to wide
df = df_transactions.pivot(index='ticket_id', columns='item_name', values='item_count').fillna(0)
df_transactions.reset_index(inplace=True)
df_transactions.drop(columns='index', inplace=True)

# --- add back date and location
df = df.merge(df_transactions[['ticket_id', 'location', 'order_timestamp']].drop_duplicates(), how='left', on='ticket_id')

# --- extract hour of day from datetime
df['hour'] = df.order_timestamp.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

# --- convert categorical store variables to dummies
encoded_data = OneHotEncoder(sparse=False).fit(np.array(df['location']).reshape(-1, 1))
##### use sklearn.preprocessing.OneHotEncoder() to create a class object called encoded_data (see documentation for OneHotEncoder online)
##### call the method used to fit data for a OneHotEncorder object. Note: you will have to reshape data from a column of the data frame. useful functions may be DataFrame methods .to_list(), .reshape(), and .shape()
col_map_store_binary = dict(zip(list(encoded_data.get_feature_names_out()), ['store_' + x.split('x0_')[1] for x in encoded_data.get_feature_names_out()]))

df_store_binary = pd.DataFrame(encoded_data.transform(X=np.array(df['location'].tolist()).reshape(df.shape[0], 1)))
df_store_binary.columns = encoded_data.get_feature_names_out()
df_store_binary.rename(columns=col_map_store_binary, inplace=True)

df = pd.concat([df, df_store_binary], axis=1)

# ------ RUN CLUSTERING -----
# --- set parameters
n_clusters = 3
init_point_selection_method = 'k-means++'

# --- select data
##### specify list of attributes on which to base clusters
cols_for_clustering = ['burger', 'fries', 'salad', 'shake', 'hour', 'store_1', 'store_2', 'store_3',
                       'store_4', 'store_5', 'store_6', 'store_7', 'store_8', 'store_9']
df_cluster = df.loc[:, cols_for_clustering]

# --- split to test and train
df_cluster_train, df_cluster_test, _, _, = train_test_split(df_cluster, [1]*df_cluster.shape[0], test_size=0.33)   # ignoring y values for unsupervised

# --- fit model
attribute_summary_method_dict = {'burger': np.mean, 'fries': np.mean, 'salad': np.mean, 'shake': np.mean, 'hour': np.mean, 'store_1': sum, 'store_4': sum, 'store_6': sum, 'store_3': sum, 'store_9': sum, 'store_2': sum, 'store_8': sum, 'store_5': sum, 'store_7': sum}
col_output_order = ['burger', 'fries', 'salad', 'shake', 'hour', 'store_1', 'store_2', 'store_3', 'store_4', 'store_5', 'store_6', 'store_7', 'store_8', 'store_9'] ##### specify order of output columns for easy of readability

# training data
train_model, train_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_train.reindex())
# testing data
test_model, test_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster_test.reindex())
# all data
model, model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster)

# --- run for various number of clusters
##### add the code to run the clustering algorithm for various numbers of clusters
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k, n_init=10)
    model.fit(df_cluster)
    inertias.append(model.inertia_)

# --- draw elbow plot
##### create an elbow plot for your numbers of clusters in previous step
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# --- output tagged data for examination ----
store_col_names = ['store_1', 'store_2', 'store_3', 'store_4', 'store_5', 'store_6', 'store_7', 'store_8', 'store_9']
df_cluster['store'] = None

for t_col in store_col_names:
    df_cluster.loc[df_cluster[t_col] == 1, 'store'] = t_col.split('_')[1]

df_cluster.to_csv('clustering_output.csv')

# assign cluster mode to location

test_model, test_model_summary = run_kmeans(n_clusters, init_point_selection_method, df_cluster)
df_cluster = pd.concat([df_cluster, test_model], axis=1, join='outer')
t_df = df_cluster.groupby('store')['predict_cluster_kmeans'].apply(lambda x: x.mode()).reset_index()[['store', 'predict_cluster_kmeans']]
t_df['store'] = t_df['store'].astype(str)
t_df['predict_cluster_kmeans'] = t_df['predict_cluster_kmeans'].astype(str)
df_transactions = df_transactions.astype(str)
df_transactions[['location', 'lat', 'long']].drop_duplicates().merge(t_df, how='left', left_on='location', right_on='store').to_csv('store_locations.csv')
#pd.concat([df_transactions, t_df], axis=1, join='outer').to_csv('store_locations.csv')

model = KMeans(n_clusters=3, n_init=10)
model.fit(df_cluster)
output = pd.DataFrame(model.cluster_centers_)
output.columns = df_cluster.columns
output.to_csv('Kmeans_output.csv')

# ---- Bonus code (not part of assignment) ------
# GMM
gmm_model = GaussianMixture(n_components=3, covariance_type='full')
gmm_predicted_cluster = gmm_model.fit_predict(df_cluster)
df_cluster['predict_cluster_gmm'] = gmm_predicted_cluster
gmm_model.weights_
gmm_model.means_
gmm_model.converged_
gmm_model_summary = df_cluster.groupby('predict_cluster_gmm').agg(attribute_summary_method_dict)
gmm_model_summary[col_output_order].to_csv('gmm_cluster_summary.csv')
