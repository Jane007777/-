import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

@st.cache
def load_data():
    data_df = pd.read_excel('[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx', header=None)
    data_df = data_df.drop(data_df.columns[0], axis=1)
    data_df.columns = range(data_df.shape[1])
    data_df = data_df.stack().reset_index()
    data_df.columns = ['user_id', 'joke_id', 'rating']
    data_df = data_df[data_df['rating'] != 99.0]
    jokes_df = pd.read_excel('Dataset4JokeSet.xlsx', header=None)
    jokes_df.columns = ['joke']
    jokes_df.index.name = 'joke_id'
    return data_df, jokes_df

def train_model(data_df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(data_df[['user_id', 'joke_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

def recommend_jokes(algo, data_df, jokes_df, new_user_id, new_ratings):
    new_ratings = {joke_id: info['rating']*4 - 10 for joke_id, info in new_ratings.items()}
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'joke_id': list(new_ratings.keys()),
    'rating': list(new_ratings.values())
    })
    data_df = pd.concat([data_df, new_ratings_df])
    iids = data_df['joke_id'].unique()
    iids_new_user = data_df.loc[data_df['user_id'] == new_user_id, 'joke_id']
    iids_to_pred = np.setdiff1d(iids, iids_new_user)
    testset_new_user = [[new_user_id, iid, 0.] for iid in iids_to_pred]
    predictions = algo.test(testset_new_user)
    top_5_iids = [pred.iid for pred in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]]
    top_5_jokes = jokes_df.loc[jokes_df.index.isin(top_5_iids), 'joke']
    return top_5_jokes

def main():
    data_df, jokes_df = load_data()
    new_user_id = data_df['user_id'].max() + 1
    if 'initial_ratings' not in st.session_state:
        st.session_state.initial_ratings = {}
        random_jokes = jokes_df.sample(n=3)
        for joke_id, joke in zip(random_jokes.index, random_jokes['joke']):
            st.session_state.initial_ratings[joke_id] = {'joke': joke, 'rating': 3}
    for joke_id, info in st.session_state.initial_ratings.items():
        st.write(info['joke'])
        info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'rec_{joke_id}')
    if st.button('Submit Ratings'):
        new_ratings_df = pd.DataFrame({
            'user_id': [new_user_id]*len(st.session_state.initial_ratings),
            'joke_id': list(st.session_state.initial_ratings.keys()),
            'rating': [info['rating'] for info in st.session_state.initial_ratings.values()]
        })
        data_df = pd.concat([data_df, new_ratings_df])
        algo = train_model(data_df)
        recommended_jokes = recommend_jokes(algo, data_df, jokes_df, new_user_id, st.session_state.initial_ratings)
        st.session_state.recommended_jokes = {}
        for joke_id, joke in zip(recommended_jokes.index, recommended_jokes):
            st.session_state.recommended_jokes[joke_id] = {'joke': joke, 'rating': 3}
    if 'recommended_jokes' in st.session_state:
        st.write('We recommend the following jokes based on your ratings:')
        for joke_id, info in st.session_state.recommended_jokes.items():
            st.write(info['joke'])
            info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'rec_{joke_id}')
        if st.button('Submit Recommended Ratings'):
            total_score = sum([info['rating'] for info in st.session_state.recommended_jokes.values()])
            percentage_of_total = (total_score / 25) * 100
            st.write(f'Your percentage of total possible score: {percentage_of_total}%')

if __name__ == '__main__':
    main()
