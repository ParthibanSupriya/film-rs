'''
ref: https://www.kaggle.com/sohier/film-recommendation-engine-converted-to-use-tmdb/data
'''
import os
import nltk
from exploration import load_tmdb_dataset, convert_to_original_format, \
        all_keywords, count_word, drop_old_columns, \
        missing_values, wordcloud_and_histogram, \
        group_by_decade_and_show
from cleaning import deduplicate, keywords_inventory, \
        replacement_of_keywords, clean_keywords_with_low_freq,\
        extract_keywords_from_title, variable_linreg_imputation,\
        fill_year, visualize_word_freq_diff, correlation,\
        visualize_filling_factor, pairplot
from engine import find_similarities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nltk.download('wordnet')
pd.set_option('display.max_colwidth', -1)


def main():
    #movies = load_tmdb_movies('../data/tmdb_5000_movies.csv')
    #credits = load_tmdb_credits('../data/tmdb_5000_credits.csv')
    movies, credits = load_tmdb_dataset()
    init_tmdb_movies = convert_to_original_format(movies, credits)

    liste = all_keywords(init_tmdb_movies, 'plot_keywords')
    ko, kc = count_word(init_tmdb_movies, 'plot_keywords', liste)
    print('\n\n\nThe top 10 keyword by frequency:')
    for k in ko[:10]: print(k)
    
    print('\n\n\nThen show the wordcloud and histogram of the top 50 keywords...')
    wordcloud_and_histogram(ko, show_histogram=True)
    missing_df = missing_values(init_tmdb_movies)
    print('\n\n\nfilling factor of each feature(column):')
    print(missing_df)
    
    group_by_decade_and_show(init_tmdb_movies)
    liste = all_keywords(init_tmdb_movies, 'genres')
    ko, kc = count_word(init_tmdb_movies, 'genres', liste)
    print('\n\n\nThe top 10 genre by frequency:')
    for k in ko[:10]: print(k)
    
    print('\n\n\nThen show the wordcloud and histogram of the top 50 genres...')
    wordcloud_and_histogram(ko, False, 'Genres')
    
    df_duplicate_cleaned = deduplicate(init_tmdb_movies)
    _, _1, keywords_select = keywords_inventory(df_duplicate_cleaned)
    df_keywords_cleaned = replacement_of_keywords(df_duplicate_cleaned, 
            keywords_select, column='plot_keywords', roots=True)
    df_keyword_occurence = clean_keywords_with_low_freq(df_keywords_cleaned)
    visualize_word_freq_diff(init_tmdb_movies, df_keyword_occurence)
    
    correlation(df_keyword_occurence)
    df_var_cleaned = drop_old_columns(df_keyword_occurence)
    missing_df = missing_values(df_var_cleaned)
    print('\n\n\nfilling factor of each feature(column) again:')
    print(missing_df)
    visualize_filling_factor(df_var_cleaned)
    
    print('Find out records that miss title year.')
    df_filling = df_var_cleaned.copy(deep=True)
    missing_year_info = df_filling[df_filling['title_year'].isnull()][
            ['movie_title', 'director_name', \
                    'actor_1_name', 'actor_2_name', 'actor_3_name']]
    print(missing_year_info[:10])
    fill_year(df_filling)
    extract_keywords_from_title(df_filling)
    print('Pairplot of feature \'gross\' and \'num_voted_users\'.')
    pairplot(df_filling, 'gross', 'num_voted_users')
    variable_linreg_imputation(df_filling, 'gross', 'num_voted_users')
    missing_df = missing_values(df_filling)
    print('\n\n\nFinally, filling factor of each feature(column):')
    print(missing_df)
    print('dataset size before dropping: ', df_filling.shape[0])
    df_filling = df_filling.replace([np.inf, -np.inf], np.nan)\
            .dropna(subset=df_filling.columns.values.tolist())
    print('dataset size after dropping: ', df_filling.shape[0])
    print('\n\n\n' + '='*40)
    print('Below is the test cases:')
    for i in range(0, 20, 3):
        find_similarities(df_filling, i, del_sequels=True, verbose=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()


