import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

@st.cache_data
def load_data():
    # Load the data
    data_df = pd.read_excel('[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx', header=None)
    # Delete columns[0]
    data_df = data_df.drop(data_df.columns[0], axis=1)
    # Rename columns to match with jokes dataframe
    data_df.columns = data_df.columns = range(data_df.shape[1])

    # Convert data_df from wide format to long format
    data_df = data_df.stack().reset_index()
    data_df.columns = ['user_id', 'joke_id', 'rating']

    # Filter out missing ratings
    data_df = data_df[data_df['rating'] != 99.0]

    # Load the jokes
    jokes_df = pd.read_excel('Dataset4JokeSet.xlsx')
    jokes_df.columns = ['joke']
    jokes_df.index.name = 'joke_id'

    return data_df, jokes_df

def train_model(data_df):
    # Create a Reader object
    reader = Reader(rating_scale=(0, 5))

    # Load the data into a Dataset object
    data = Dataset.load_from_df(data_df[['user_id', 'joke_id', 'rating']], reader)

    # Build a full trainset from data
    trainset = data.build_full_trainset()

    # Train a SVD model
    algo = SVD()
    algo.fit(trainset)

    return algo

def recommend_jokes(algo, data_df, jokes_df, new_user_id, new_ratings):
    # Convert ratings from 0-5 scale to -10 to 10 scale
    new_ratings = {joke_id: info['rating']*4 - 10 for joke_id, info in new_ratings.items()}

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'joke_id': list(new_ratings.keys()),
    'rating': list(new_ratings.values())
    })

    data_df = pd.concat([data_df, new_ratings_df])

    # Generate recommendations for the new user
    iids = data_df['joke_id'].unique() # Get the list of all joke ids
    iids_new_user = data_df.loc[data_df['user_id'] == new_user_id, 'joke_id'] # Get the list of joke ids rated by the new user
    iids_to_pred = np.setdiff1d(iids, iids_new_user) # Get the list of joke ids the new user has not rated

    # Predict the ratings for all unrated jokes
    testset_new_user = [[new_user_id, iid, 0.] for iid in iids_to_pred]
    predictions = algo.test(testset_new_user)

    # Get the top 5 jokes with highest predicted ratings
    top_5_iids = [pred.iid for pred in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]]
    top_5_jokes = jokes_df.loc[jokes_df.index.isin(top_5_iids), 'joke']

    return top_5_jokes

def main():
# streamlit界面交互设置
columns = input('您的评分：')
print(满意度为：'columns')








    if __name__ == '__main__':
    main()