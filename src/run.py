import nltk
from exploration import load_tmdb_movies, \
        load_tmdb_credits, convert_to_original_format, \
        all_keywords, count_word, drop_old_columns, \
        missing_values
from cleaning import deduplicate, keywords_inventory, \
        replacement_of_keywords, clean_keywords_with_low_freq,\
        extract_keywords_from_title, variable_linreg_imputation,\
        fill_year
from engine import find_similarities


nltk.download('wordnet')

movies = load_tmdb_movies('../data/tmdb_5000_movies.csv')
credits = load_tmdb_credits('../data/tmdb_5000_credits.csv')
init_tmdb_movies = convert_to_original_format(movies, credits)
df_duplicate_cleaned = deduplicate(init_tmdb_movies)
_, _1, keywords_select = keywords_inventory(df_duplicate_cleaned)
df_keywords_cleaned = replacement_of_keywords(df_duplicate_cleaned, 
        keywords_select, column='plot_keywords', roots=True)
df_keyword_occurence = clean_keywords_with_low_freq(df_keywords_cleaned)
#visualize_word_freq_diff(init_tmdb_movies, df_keyword_occurence)
#correlation(df_keyword_occurence)
df_var_cleaned = drop_old_columns(df_keyword_occurence)
#visualize_filling_factor(df_var_cleaned)
df_filling = df_var_cleaned.copy(deep=True)
missing_year_info = df_filling[df_filling['title_year'].isnull()][
        ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']]
#print(missing_year_info[:10])
fill_year(df_filling)
extract_keywords_from_title(df_filling)
#pairplot(df_filling, 'gross', 'num_voted_users')
variable_linreg_imputation(df_filling, 'gross', 'num_voted_users')
print('\n\n\n' + '='*40)
for i in range(0, 20, 3):
    find_similarities(df_filling, i, del_sequels=True, verbose=True)
#plt.show()
