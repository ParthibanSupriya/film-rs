import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


def my_pltsavefig(fig_title, mydir='../figure/', fig_extension='.png'):
    filename = mydir + fig_title.replace(' ', '-') + fig_extension
    if not os.path.exists(filename):
        plt.savefig(filename)
        return True
    else:
        return False


def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(
            lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries',
            'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df


LOST_COLUMNS = [
    'actor_1_facebook_likes',    
    'actor_2_facebook_likes',    
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews'
    ]

TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
        'budget': 'budget',
        'genres': 'genres',
        'revenue': 'gross',
        'title': 'movie_title',
        'runtime': 'duration',
        'original_language': 'language',
        'keywords': 'plot_keywords',
        'vote_count': 'num_voted_users',
    }

IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}


# return missing value rather than an error upon indexing/key failure
def safe_access(container, index_values):
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan


def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])


def pipe_flattern_names(keywords):
    return '|'.join([x['name'] for x in keywords])


def convert_to_original_format(movies, credits):
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(
            lambda x: x.year)
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(
            lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(
            lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(
            lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(
            lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(
            lambda x: safe_access(x, [3, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(
            pipe_flattern_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(
            pipe_flattern_names)
    return tmdb_movies


# get all the unique key words in a given feature.
def all_keywords(df, col_name):
    unique_words = set()
    for liste_keywords in df[col_name].str.split('|').values:
        # iff NaN
        if isinstance(liste_keywords, float): continue
        unique_words = unique_words.union(liste_keywords)
    return unique_words


# get the frequency of all the unique key words.
def count_word(df, ref_col, liste):
    keyword_count = {}
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue
        for s in [s for s in liste_keywords if s in liste]:
            if pd.notnull(s): keyword_count[s] += 1

    keyword_occurences = []
    for k, v in keyword_count.items():
        keyword_occurences.append([k, v])
    keyword_occurences.sort(key = lambda x: x[1], reverse = True)
    # a little weird... there are entries here that have no keyword...
    keyword_occurences = [x for x in keyword_occurences if x[0]]
    return keyword_occurences, keyword_count


def random_color_func(word=None, font_size=None, position=None,
        orientation=None, font_path=None, random_state=None):
    # define the color of the words
    tone = 55.0
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return 'hsl({}, {}%, {}%)'.format(h, s, l)


def wordcloud_and_histogram(keyword_occurences, 
        show_histogram=False, col_name='PlotKeyword'):
    fig = plt.figure(1, figsize=(9, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    # no too many keywords to show
    trunc_occurences = keyword_occurences[0:50]
    words = {s[0]: s[1] for s in trunc_occurences}
    wordcloud = WordCloud(width=1000, height=300, background_color='black',
            max_words=1628, relative_scaling=1,
            color_func=random_color_func,
            normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation='bilinear')
    ax1.axis('off')

    if show_histogram:
        ax2 = fig.add_subplot(2, 1, 2)
        y_axis = [i[1] for i in trunc_occurences]
        x_axis = [k for k, i in enumerate(trunc_occurences)]
        x_label = [i[0] for i in trunc_occurences]
        plt.xticks(rotation=85, fontsize=13)
        plt.yticks(fontsize=15)
        plt.xticks(x_axis, x_label)
        plt.ylabel('No. of occurences', fontsize=18, labelpad=10)
        ax2.bar(x_axis, y_axis, align='center', color='g')
    t = '{} popularity'.format(col_name)
    plt.title(t, bbox={'facecolor': 'k', 'pad': 5}, color='w',
            fontsize=25)
    my_pltsavefig(t)
    plt.show()


# count of missing values of each feature.
def missing_values(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['filling_factor'] = (df.shape[0] - missing_df['missing_count'])\
            / df.shape[0] * 100
    missing_df = missing_df.sort_values('filling_factor').reset_index(drop=True)
    return missing_df
    

def group_by_decade_and_show(df):
    df['decade'] = df['title_year'].apply(lambda x: ((x - 1900) // 10) * 10)

    def get_stats(gr):
        return {'min': gr.min(),
                'max': gr.max(),
                'count': gr.count(),
                'mean': gr.mean()}
    test = df['title_year'].groupby(df['decade']).apply(get_stats).unstack()
    def label(s):
        val = (1900 + s, s)[s < 100]
        chaine = '' if s < 50 else "{}'s".format(int(val))
        return chaine
    sns.set_context('poster', font_scale=0.85)
    plt.rc('font', weight='bold')
    f, ax = plt.subplots(figsize=(11, 6))
    labels = [label(s) for s in test.index]
    sizes = test['count'].values
    explode = [0.2 if sizes[i] < 100 else 0.01 for i in range(11)]
    ax.pie(sizes, explode=explode, labels=labels,
            autopct=lambda x: '{:1.0f}%'.format(x) if x > 1 else '',
            shadow=False, startangle=0)
    ax.axis('equal')
    t = '% of films per decade'
    ax.set_title(t, bbox={'facecolor': 'k', 'pad': 5},
            color='w', fontsize=16)
    df.drop('decade', axis=1, inplace=True)
    my_pltsavefig(t)
    plt.show()
    return test


# The transformation from the old dataset to the new one.
def drop_old_columns(df):
    new_col_order = ['movie_title', 'title_year', 'genres', 'plot_keywords',
            'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
            'director_facebook_likes', 'actor_1_facebook_likes', 
            'actor_2_facebook_likes', 'actor_3_facebook_likes', 
            # missing: 'movie_facebook_likes', 
            'num_critic_for_reviews', 
            'num_user_for_reviews', 'num_voted_users', 'language', 'country',
            'imdb_score', 'movie_imdb_link', 'color', 'duration', 'gross']
    new_col_order = [col for col in new_col_order if col not in LOST_COLUMNS]
    new_col_order = [IMDB_COLUMNS_TO_REMAP[col] 
            if col in IMDB_COLUMNS_TO_REMAP else col
            for col in new_col_order]
    new_col_order = [TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES[col] 
            if col in TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES else col 
            for col in new_col_order]
    df_var_cleaned = df[new_col_order]
    return df_var_cleaned

