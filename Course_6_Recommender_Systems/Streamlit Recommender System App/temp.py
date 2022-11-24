# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 07:32:19 2022

@author: mmatr
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd

# course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
# course_genres_df = pd.read_csv(course_genre_url)

os.chdir(r'C:\Users\mmatr\Desktop\Learning Data Science\IBM Machine Learning\Git\IBM-Machine-Learning-Course\Course_6_Recommender_Systems\Streamlit Recommender System App')


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")

def load_course_genres():
    return pd.read_csv("course_genres.csv")

def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")



def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict

def combine_cluster_labels(user_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id, res_dict


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

idx_id_dict, id_idx_dict = get_doc_dicts()
idx_id_dict, id_idx_dict=get_doc_dicts()
profile_df=load_profiles()

all_courses = set(idx_id_dict.values())

course_genres_df = load_course_genres()
test_users_df=load_ratings()[['user','item']]

kmeans, cluster_df=train_kmeans(10)

test_users_labelled = pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')

