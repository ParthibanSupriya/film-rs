import nltk
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import sys
import pandas as pd
import seaborn as sns
from exploration import all_keywords, count_word


# Actually this function does not change df.
def deduplicate(df):
    # actually no duplicate
    doubled_entries = df[df.id.duplicated()]
    #print(doubled_entries.shape)

    # check whether there is duplicate of movie title.
    df_temp = df
    list_var_duplicates = ['movie_title', 'title_year', 'director_name']
    liste_duplicates = df_temp['movie_title']\
            .map(df_temp['movie_title'].value_counts() > 1)
    print('Num of duplicate entries: {}'.format(
        len(df_temp[liste_duplicates][list_var_duplicates])))
    print(df_temp[liste_duplicates][list_var_duplicates].sort_values('movie_title'))
    df_duplicate_cleaned = df_temp
    return df_duplicate_cleaned


def keywords_inventory(df, column='plot_keywords'):
    PS = nltk.stem.PorterStemmer()
    keywords_roots = {}
    keywords_select = {}
    category_keys = []
    for s in df[column]:
        if pd.isnull(s): continue
        for t in s.split('|'):
            t = t.lower()
            racine = PS.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
            else:
                keywords_roots[racine] = {t}

    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = sys.maxsize
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
    print('Num of keywords in variable \'{}\': {}'.format(
        column, len(category_keys)))
    # get a feel of a sample of keywords that appear in close varieties
    icount = 0
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            icount += 1
            if icount < 15:
                print(icount, keywords_roots[s], len(keywords_roots[s]))
    return category_keys, keywords_roots, keywords_select

# replace keywords with their root-like word.
def replacement_of_keywords(df, dict_replacement, 
        column = 'plot_keywords', roots = False):
    PS = nltk.stem.PorterStemmer()
    df_new = df.copy(deep = True)
    for index, row in df_new.iterrows():
        chaine = row[column]
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'):
            clef = PS.stem(s) if roots else s
            if clef in dict_replacement.keys():
                nouvelle_liste.append(dict_replacement[clef])
            else:
                nouvelle_liste.append(s)
        df_new.at[index, 'plot_keywords'] = ('|'.join(nouvelle_liste))
    return df_new


# get the synonymes of the word 'mot_cle'
def get_synonymes(mot_cle):
    lemma = set()
    for ss in wordnet.synsets(mot_cle):
        for w in ss.lemma_names():
            index = ss.name().find('.') + 1
            # just get the 'nouns'.
            if ss.name()[index] == 'n':
                lemma.add(w.lower().replace('_', ' '))
    return lemma


def clean_keywords_with_low_freq(df):
    keywords = all_keywords(df, 'plot_keywords')
    ko, kc = count_word(df, 'plot_keywords', keywords)
    ko.sort(key = lambda x: x[1], reverse = False)
    key_count = {s[0]: s[1] for s in ko}
    '''
    # testing
    mot_cle = 'alien'
    lemma = get_synonymes(mot_cle)
    for s in lemma:
        print('"{:<30}" in keywords list -> {} {}'.format(
            s, s in keywords, kc[s] if s in keywords else 0))
    '''
    def test_keyword(mot, key_count, threshold):
        return key_count.get(mot, 0) >= threshold
    replacement_mot = {}
    icount = 0
    # the worst case is that keywords with frequency less than 5
    # are replaced by themselves
    for index, [mot, nb_apparitions] in enumerate(ko):
        # only filter keywords with frequency less than 5
        if nb_apparitions > 5: continue
        lemma = get_synonymes(mot)
        if len(lemma) == 0: continue
        liste_mots = [(s, key_count[s]) for s in lemma
                if test_keyword(s, key_count, key_count[mot])]
        liste_mots.sort(key = lambda x: (x[1], x[0]), reverse = True)
        if len(liste_mots) <= 1: continue # no replacement
        if mot == liste_mots[0][0]: continue # replacement by itself
        icount += 1
        if icount < 8:
            print('{:<12} -> {:<12} (init: {})'.format(
                mot, liste_mots[0][0], liste_mots))
        replacement_mot[mot] = liste_mots[0][0]

    print(90*'_' + '\nThe replacement concerns {}% of the keywords'.format(
        round(len(replacement_mot) / len(keywords) * 100, 2)))

    print('KEYWORDS THAT APPEAR BOTH IN KEYS AND VALUES: \n' + 45*'-')
    icount = 0
    for s in replacement_mot.values():
        if s in replacement_mot.keys():
            icount += 1
            if icount < 10: print('{:<20} -> {:<20}'.format(s, replacement_mot[s]))

    for k, v in replacement_mot.items():
        if v in replacement_mot.keys():
            replacement_mot[k] = replacement_mot[v]

    df_keywords_synonyms = replacement_of_keywords(df, replacement_mot, 
            roots = False)
    keywords, _2, _3 = \
            keywords_inventory(df_keywords_synonyms, column='plot_keywords')
    new_keyword_occurences, _4 = count_word(df_keywords_synonyms, 
            'plot_keywords', keywords)
    #print(len(new_keyword_occurences))
    #print(new_keyword_occurences[:5])
    df_keywords_occurence = \
            replacement_of_low_frequency_keywords(df_keywords_synonyms, 
                    new_keyword_occurences)
    #keywords_inventory(df_keywords_occurence, column='plot_keywords')
    return df_keywords_occurence


def replacement_of_low_frequency_keywords(df, keyword_occurences):
    df_new = df.copy(deep = True)
    key_count = {}
    for s in keyword_occurences:
        key_count[s[0]] = s[1]
    for index, row in df_new.iterrows():
        chaine = row['plot_keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'):
            # a little tricky. Notice that this statement is not always true.
            if key_count.get(s, 4) > 3: nouvelle_liste.append(s)
        df_new.at[index, 'plot_keywords'] = '|'.join(nouvelle_liste)
    return df_new


def visualize_word_freq_diff(df_before, df_after, column='plot_keywords'):
    def get_keyword_occurence(df, column, reverse = True):
        keywords, _, __ = \
                keywords_inventory(df, column)
        ko, _ = count_word(df, column, keywords)
        ko.sort(key = lambda x: x[1], reverse = reverse)
        return ko
    ko_before = get_keyword_occurence(df_before, column)
    ko_after = get_keyword_occurence(df_after, column)
    #font = {'family': 'fantasy', 'weight': 'normal', 'size': 15}
    # mpl.rc('font', **font)
    y_axis_before = [i[1] for i in ko_before]
    x_axis_before = [k for k in range(len(ko_before))]
    y_axis_after = [i[1] for i in ko_after]
    x_axis_after = [k for k in range(len(ko_after))]
    f, ax = plt.subplots(figsize = (9, 5))
    ax.plot(x_axis_before, y_axis_before, 'r-', label='before cleaning')
    ax.plot(x_axis_after, y_axis_after, 'b-', label='after cleaning')
    legend = ax.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('medium')
    plt.ylim((0, 25))
    plt.axhline(y = 3.5, linewidth = 2, color = 'k')
    plt.xlabel('keywords index', family = 'fantasy', fontsize = 15)
    plt.ylabel('Num of occurences', family = 'fantasy', fontsize = 15)
    plt.text(3500, 4.5, 'threshold for keyword delation', fontsize = 13)


def correlation(df):
    f, ax = plt.subplots(figsize = (12, 9))
    corrmat = df.dropna(how='any').corr()
    k = 17
    cols = corrmat.nlargest(k, 'num_voted_users')['num_voted_users'].index
    cm = np.corrcoef(df[cols].dropna(how='any').values.T)
    sns.set(font_scale = 1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
            fmt='.2f', annot_kws={'size': 10}, linewidth=0.1,
            cmap='coolwarm', yticklabels=cols.values,
            xticklabels=cols.values)
    f.text(0.5, 0.93, 'Correlation coefficients', ha='center',
            fontsize=18, family='fantasy')


def visualize_filling_factor(df):
    missing_df = missing_values(df)
    y_axis = missing_df['filling_factor']
    x_label = missing_df['column_name']
    x_axis = missing_df.index
    fig = plt.figure(figsize = (11, 4))
    plt.xticks(rotation=80, fontsize=14)
    plt.yticks(fontsize=13)

    N_thresh = 5
    plt.axvline(x=N_thresh - 0.5, linewidth = 2, color = 'r')
    plt.text(N_thresh - 4.8, 30, 'filling factor \n < {}%'.format(
        round(y_axis[N_thresh], 1)), fontsize = 15, family = 'fantasy', 
        bbox = {'boxstyle': 'round', 'ec': (1.0, 0.5, 0.5), 'fc': (0.8, 0.5, 0.5)})
    N_thresh = 13
    plt.axvline(x=N_thresh-0.5, linewidth=2, color='g')
    plt.text(N_thresh, 30, 'filling factor \n = {}%'.format(
        round(y_axis[N_thresh], 1)), fontsize = 15, family = 'fantasy', 
        bbox = {'boxstyle': 'round', 'ec': (1.0, 0.5, 0.5), 'fc': (0.5, 0.8, 0.5)})

    plt.xticks(x_axis, x_label, family='fantasy', fontsize=14)
    plt.ylabel('Filling factor (%)', family='fantasy', fontsize=16)
    plt.bar(x_axis, y_axis)

# It seems this function could not work as expected.
# Because the row with title_year as null has all attributes as null.
def fill_year(df):
    col = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
    usual_year = [0 for _ in range(4)]
    var = [0 for _ in range(4)]
    for i in range(4):
        usual_year[i] = df.groupby(col[i])['title_year'].mean()
    actor_year = {}
    for i in range(4):
        for s in usual_year[i].index:
            if s in actor_year.keys():
                if pd.notnull(usual_year[i][s]) and pd.notnull(actor_year[s]):
                    actor_year[s] = (actor_year[s] + usual_year[i][s]) / 2
                elif pd.isnull(actor_year[s]):
                    actor_year[s] = usual_year[i][s]
            else:
                actor_year[s] = usual_year[i][s]
    missing_year_info = df[df['title_year'].isnull()]
    icount_replaced = 0
    for index, row in missing_year_info.iterrows():
        value = [np.NaN for _ in range(4)]
        icount = 0
        sum_year = 0
        for i in range(4):
            var[i] = df.loc[index][col[i]]
            if pd.notnull(var[i]):
                value[i] = actor_year[var[i]]
            if pd.notnull(value[i]):
                icount += 1
                sum_year += actor_year[var[i]]
        if icount != 0:
            sum_year = sum_year / icount
        if int(sum_year) > 0:
            icount_replaced += 1
            df.at[index, 'title_year'] = int(sum_year)
            if icount_replaced < 10:
                print('title:{:<45} -> year:{:<20}'.format(
                    df.loc[index]['movie_title'], int(sum_year)))


# It seems this function did not work
# because all fields of the column 'plot_keywords' are filled.
def extract_keywords_from_title(df):
    icount = 0
    keywords = all_keywords(df, 'plot_keywords')
    for index, row in df[df['plot_keywords'].isnull()].iterrows():
        icount += 1
        liste_mot = row['movie_title'].strip().split()
        new_keywords = []
        for s in liste_mot:
            lemma = get_synonymes(s)
            for t in list(lemma):
                if t in keywords:
                    new_keywords.append(t)
        if new_keywords and icount < 15:
            print('{:<50} -> {:<30}'.format(row['movie_title'], str(new_keywords)))
        if new_keywords:
            df.at[index, 'plot_keywords'] = '|'.join(new_keywords)


def pairplot(df, col1, col2):
    sns.set(font_scale=1.25)
    sns.pairplot(df.dropna(how='any')[[col1, col2]], diag_kind='kde', height=2.5)


def variable_linreg_imputation(df, col_to_predict, ref_col):
    regr = linear_model.LinearRegression()
    test = df[[col_to_predict, ref_col]].dropna(how='any', axis=0)
    X = np.array(test[ref_col])
    Y = np.array(test[col_to_predict])
    # why bother?
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)
    regr.fit(X, Y)

    test = df[df[col_to_predict].isnull() & df[ref_col].notnull()]
    for index, row in test.iterrows():
        value = float(regr.predict(row[ref_col]))
        df.at[index, col_to_predict] = value

