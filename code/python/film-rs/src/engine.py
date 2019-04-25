import math
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import pandas as pd


gaussian_filter = lambda x, y, sigma: math.exp(-(x - y)**2 / (2*sigma**2))


def entry_variables(df, id_entry):
    col_labels = []
    director_name = df['director_name'].iloc[[id_entry]]
    if not director_name.empty:
        for s in director_name.str.split('|'):
            col_labels.extend(s)
    for i in range(3):
        column = 'actor_NUM_name'.replace('NUM', str(i+1))
        actor_name = df[column].iloc[[id_entry]]
        if not actor_name.empty:
            for s in actor_name.str.split('|'):
                col_labels.extend(s)
    plot_keywords = df['plot_keywords'].iloc[[id_entry]]
    if not plot_keywords.empty:
        for s in plot_keywords.str.split('|'):
            col_labels.extend(s)
    return col_labels


def add_variables(df, REF_VAR):
    # There may be a bug here. What if there has been already a column
    # named the same name as one in REF_VAR.
    for s in REF_VAR: df[s] = pd.Series([0 for _ in range(len(df))])
    columns = ['genres', 'actor_1_name', 'actor_2_name',
            'actor_3_name', 'director_name', 'plot_keywords']
    for category in columns:
        for index, row in df.iterrows():
            if pd.isnull(row[category]): continue
            for s in row[category].split('|'):
                if s in REF_VAR: df.at[index, s] = 1
    return df


# recommend the most promising N(=31) films
def recommend(df, id_entry):
    N = 31
    df_copy = df.copy(deep = True)
    liste_genres = set()
    # Why do we need all the genres instead of the genre of the given id?
    for s in df['genres'].str.split('|').values:
        liste_genres = liste_genres.union(set(s))
    variables = entry_variables(df_copy, id_entry)
    variables += list(liste_genres)
    df_new = add_variables(df_copy, variables)

    X = df_new[variables].values
    nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto',
            metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    xtest = df_new.iloc[[id_entry]][variables].values
    xtest = xtest.reshape(1, -1)
    distances, indices = nbrs.kneighbors(xtest)
    return indices[0][:]


# liste_films: list of id of the selected films.
def extract_parameters(df, liste_films):
    parameter_films = ['_' for _ in range(len(liste_films))]
    i = 0
    max_users = -1
    for index in liste_films:
        parameter_films[i] = df.iloc[[index]][['movie_title',
            'title_year', 'vote_average', 'num_voted_users']].values.tolist()[0]
        parameter_films[i].append(index)
        max_users = max(max_users, parameter_films[i][3])
        i += 1
    title_main = parameter_films[0][0]
    annee_ref = parameter_films[0][1] # title_year
    parameter_films.sort(key = lambda x: criteria_selection(title_main,
        max_users, annee_ref, x[0], x[1], x[2], x[3]), reverse = True)
    return parameter_films


# compare the similarity of two strings.
def sequel(titre_1, titre_2):
    if fuzz.ratio(titre_1, titre_2) > 50 or fuzz.token_set_ratio(titre_1, titre_2) > 50:
        return True
    else:
        return False


def criteria_selection(title_main, max_users, 
        annee_ref, titre, annee, imdb_score, votes):
    if pd.notnull(annee_ref):
        facteur_1 = gaussian_filter(annee_ref, annee, 20)
    else:
        facteur_1 = 1
    
    sigma = max_users * 1.0
    if pd.notnull(votes):
        facteur_2 = gaussian_filter(votes, max_users, sigma)
    else:
        # why is it 0?
        facteur_2 = 0

    if sequel(title_main, titre):
        note = 0
    else:
        note = imdb_score**2 * facteur_1 * facteur_2
    return note


# select the top 5 from the N(=31) films.
def add_to_selection(film_selection, parameter_films):
    NUM_TO_RECOMMEND = 8
    film_list = film_selection[:]
    icount = len(film_list)
    for i in range(len(parameter_films)):
        already_in_list = False
        for s in film_selection:
            if s[0] == parameter_films[i][0] or\
                    sequel(s[0], parameter_films[i][0]):
                already_in_list = True
        if already_in_list: continue
        icount += 1
        if icount <= NUM_TO_RECOMMEND:
            film_list.append(parameter_films[i])
    return film_list


# If more than two films from a series are present, the older one is kept.
def remove_sequels(film_selection):
    removed_from_selection = []
    for i, film_1 in enumerate(film_selection):
        for j, film_2 in enumerate(film_selection):
            if j <= i: continue
            if sequel(film_1[0], film_2[0]):
                last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                removed_from_selection.append(last_film)
    film_list = [film for film in film_selection 
            if film[0] not in removed_from_selection]
    return film_list


def find_similarities(df, id_entry, del_sequels = True, verbose = False):
    if verbose:
        print(90*'_' + '\nQUERY: films similar to id={} -> \'{}\''.format(
            id_entry, str(df.iloc[[id_entry]]['movie_title'])))
    liste_films = recommend(df, id_entry)
    parameter_films = extract_parameters(df, liste_films)
    film_selection = []
    film_selection = add_to_selection(film_selection, parameter_films)
    if del_sequels: film_selection = remove_sequels(film_selection)
    film_selection = add_to_selection(film_selection, parameter_films)
    selection_titres = []
    for i, s in enumerate(film_selection):
        selection_titres.append([s[0].replace(u'\xa0', u''), s[4]])
        if verbose: print('n{:<2} -> {:<30}'.format(i+1, s[0]))
    return selection_titres

