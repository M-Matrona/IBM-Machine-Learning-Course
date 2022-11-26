import pandas as pd
import numpy as np

import streamlit as st

#kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

"""
idx_id_dict - dictionary of key:doc_index to value:doc_id
id_idx_dict - dictionary of key:doc_id to values:doc_index
sim_matrix - matrix where RC corresponds to the similiarity between two courses
             IBM was never clear about what this was.  Seems COSINE similiarity between 

"""

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

def load_profiles():
    return pd.read_csv('profile_df.csv')

# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def add_new_ratings(new_courses, params):
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
        user_df = pd.DataFrame(res_dict)
        
        if not (user_df.iloc[-1,1:]==ratings_df.iloc[-1,1:]).all():
            updated_ratings = pd.concat([ratings_df, user_df])
            updated_ratings.to_csv("ratings.csv", index=False)        
        
        profile=build_profile_vector(new_courses,new_id)
        
        params['profile']=profile
        params['new_user_id']=new_id
        params['user_df']=user_df
        
        return params

def build_profile_vector(courses,new_id):
    
    course_genres_df=load_course_genres()
    profile_df=load_profiles()
    
    profile=np.zeros(14) #empty profile series
    
    for course in courses:
        profile=profile + np.array(course_genres_df[course_genres_df['COURSE_ID']==course].iloc[0,2:])*3.0
        
    # """
    # turned off adding to csvs for same reason as above
    # """    
    
    cp=np.insert(profile, [0], new_id)    
    dft=pd.DataFrame(cp.reshape(1,-1),columns=profile_df.columns)
    
    if not (dft.iloc[-1,1:]==profile_df.iloc[-1,1:]).all():
        updated_profiles=pd.concat([profile_df, dft])
        updated_profiles.to_csv('profile_df.csv', index=False)
    
    
    return profile    
    
def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {} #res for recommendation result
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

def generate_recommendation_scores_user_profile(user_df, params):
    
    users = []
    courses = []
    scores = []
    
    score_threshold = 0.6
    
    
    idx_id_dict, id_idx_dict=get_doc_dicts()
    profile_df=load_profiles()
    
    test_user_ids=[params['new_user_id']]
    all_courses = set(idx_id_dict.values())
    
    course_genres_df = load_course_genres()
        
    profile=params['profile']
    
    unselected_course_ids = all_courses.difference(set(user_df['item']))
    
    for user_id in test_user_ids:
        
        # get user vector for the current user id

        test_user_vector=profile
        
        # get the unknown course ids for the current user id
        enrolled_courses = list(user_df['item'])
        unknown_courses = all_courses.difference(enrolled_courses)
        unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
        unknown_course_ids = unknown_course_df['COURSE_ID'].values
        
        # user np.dot() to get the recommendation scores for each course
        unknown_course_genres=unknown_course_df.iloc[:,2:].values
        recommendation_scores = np.dot(test_user_vector,unknown_course_genres.T)
        
        
        # Append the results into the users, courses, and scores list
        for i in range(0, len(unknown_course_ids)):
            score = recommendation_scores[i]
            # Only keep the courses with high recommendation score
            if score >= score_threshold:
                users.append(user_id)
                courses.append(unknown_course_ids[i])
                scores.append(score)
                
        res_df=build_results_df(users, courses, scores, params)        
        
    return res_df 
       
def top_courses(params, courses, res_df):
    
    if "top_courses" in params and params['top_courses'] <= len(courses):
            res_df=res_df.iloc[0:params['top_courses'],:]
    
    return res_df

def combine_cluster_labels(user_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df


# Model training
def train(model_name, params):
    # TODO: Add model training code here
    if model_name==models[0]: 
        pass
    elif model_name==models[2]:
        pass
    else:
        pass
        
def build_results_df(users, courses, scores, params):
    res_dict = {}
    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    
    res_df=top_courses(params, courses, res_df.sort_values(by=['SCORE'], ascending=False)) 
    
    return res_df

# Prediction
def predict(model_name, user_ids, params, user_df):
    if model_name==models[0]: 
        
        sim_threshold = 0.6
        
        if "sim_threshold" in params:
            sim_threshold = params["sim_threshold"] / 100.0
            
        idx_id_dict, id_idx_dict = get_doc_dicts()
        sim_matrix = load_course_sims().to_numpy()
        users = []
        courses = []
        scores = []
    
        for user_id in [params['new_user_id']]:
            # Course Similarity model
            if model_name == models[0]:
                ratings_df = load_ratings()
                user_ratings = user_df#ratings_df[ratings_df['user'] == user_id]
                enrolled_course_ids = user_ratings['item'].to_list()
                res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
                for key, score in res.items():
                    if score >= sim_threshold:
                        users.append(user_id)
                        courses.append(key)
                        scores.append(score)
            # TODO: Add prediction model code here
        
        res_df=build_results_df(users, courses, scores, params) 
        
        return res_df
    
    elif model_name==models[1]:
        return generate_recommendation_scores_user_profile(user_df, params)
    
    elif model_name==models[2]:
        #train the model on existing data.  Use it for labelling
        kmeans, cluster_df=train_kmeans(params)
        
        #grab the profile vector for the current user.        
        profile=params['profile']
        
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
        count_df.drop(labels=user_df['item'], errors='ignore',inplace=True)
           
        #list of courses and the number of times they appeared in cluster

        courses=list(count_df.index)
        scores=list(count_df['count'])

        users=[params['new_user_id']]*len(courses)
        
        res_df=build_results_df(users, courses, scores, params) 
             
        return res_df
    
def train_kmeans(params):
    
    user_profile_df=load_profiles()
    feature_names = list(user_profile_df.columns[1:])
    scaler = StandardScaler()
    user_profile_df[feature_names] = scaler.fit_transform(user_profile_df[feature_names])
    features = user_profile_df.loc[:, user_profile_df.columns != 'user']
    user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']
    kmeans= KMeans(n_clusters=params['cluster_no'])
    kmeans.fit(features)
             
    cluster_df=combine_cluster_labels(user_ids, kmeans.labels_)
    
    return kmeans, cluster_df

# 