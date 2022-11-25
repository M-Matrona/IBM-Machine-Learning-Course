
import random
import numpy as np
import pandas as pd
import os 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd

os.chdir(r'C:\Data Science\IBM Machine Learning\Git\IBM-Machine-Learning-Course\Course_6_Recommender_Systems\Streamlit Recommender System App')


def combine_cluster_labels(user_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df

def load_course_genres():
    return pd.read_csv("course_genres.csv")

def load_profiles():
    return pd.read_csv('profile_df.csv')

def load_ratings():
    return pd.read_csv("ratings.csv")

def train_kmeans(no):
    
    user_profile_df=load_profiles()
    feature_names = list(user_profile_df.columns[1:])
    scaler = StandardScaler()
    user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])
    features = user_profile_df.loc[:, user_profile_df.columns != 'user']
    user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']
    kmeans= KMeans(n_clusters=no)
    kmeans.fit(features)
        
        
    cluster_df=combine_cluster_labels(user_ids, kmeans.labels_)
    
    
    return kmeans, cluster_df


def build_profile_vector(courses):
    a=1
    course_genres_df=load_course_genres()
    profile_df=load_profiles()
    
    profile=np.zeros(14) #empty profile series
    
    for course in courses:
        profile=profile + np.array(course_genres_df[course_genres_df['COURSE_ID']==course].iloc[0,2:])*3.0
        
    new_id=max(profile_df['user']) + 1
        
    #code for updating the profiles file
    
    cp=np.insert(profile, [0], new_id)    
    dft=pd.DataFrame(cp.reshape(1,-1),columns=profile_df.columns)
    
    #add to the profile dataframe if the last row is not equal to the profile vector
    if not (dft.iloc[-1,1:]==profile_df.iloc[-1,1:]).all():
        profile_df=profile_df.append(dft,ignore_index=True)
        profile_df.to_csv('profiles_df_temp.csv', index=False)
        a=0
    
    
    return profile,dft,profile_df


test_users_df=load_ratings()[['user','item']]

ml_courses=[i for i in set(test_users_df['item']) if 'ML' in i]

courses=ml_courses[0:8]

kmeans, cluster_df=train_kmeans(10)
   
#build the profile vector for the current user.
profile=build_profile_vector(courses)

profile_df=load_profiles()

#predict the label of the current user
label=float(kmeans.predict(profile.reshape(1,-1)))

#load the user/enrolled courses data
test_users_df=load_ratings()[['user','item']]

#label the rating data with clusters
test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')

#keep only the data of the cluster of interest
labelled_df=test_users_labelled[test_users_labelled['cluster']==label]

#add a count column for aggregation
labelled_df['count']=[1]*len(labelled_df)

#aggregate the number of counts
count_df=labelled_df.groupby(['item']).agg('count').sort_values(by='count',ascending=False)
count_df=count_df[['count']]

#list of courses and the number of times they appeared in cluster
courses=list(count_df.index)
scores=list(count_df['count'])

enrolled_courses=ml_courses
