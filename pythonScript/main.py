import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler, MaxAbsScaler
from elasticsearch import Elasticsearch, helpers
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA
import seaborn as sns; sns.set_theme()


# returns a ES connection object
def connect_to_elastic():
    client = Elasticsearch(host="localhost", port=9200, verify_certs=True)
    if client.ping():
        return client
    else:
        raise ValueError("Connection failed")


# stores data to elastic
def put_data_to_elastic(filename, index_name):
    client = connect_to_elastic()
    try:
        with open(filename, encoding='utf8') as csv_file:
            # convert csv to dictionary
            reader = csv.DictReader(csv_file)
            # Index documents
            helpers.bulk(client, reader, index=index_name)
    except FileNotFoundError:
        print('File "' + str(filename) + '" doesn\'t exists..')


# matches data with simple OR similarity
def get_data_from_elastic_simple(index_name, keyword) -> 'match query as pd':
    client = connect_to_elastic()
    # refresh elastic
    client.indices.refresh(index=index_name)
    # elasticsearch query to match keyword on field title
    match_query = {
        "match": {
            "book_title": keyword
        }
    }
    # execute query and return matches
    res = client.search(index=index_name, query=match_query, size=10000)

    books = []
    score = []
    for hit in res['hits']['hits']:
        books.append(hit['_source'])
        score.append(hit['_score'])

    df = pd.DataFrame(books)
    df['score'] = score

    return df


def get_data_from_elastic_custom(keyword, user_id, activate_nn=False) -> 'custom match query as pd':
    # inner function to return book avg
    def get_isbn_rating_avg(book_code):
        # match isbn and boost if user has read it
        query = {
            "bool": {
                "must": [
                    {
                        "match": {
                            "isbn": book_code
                        }
                    }
                ],
                "should": [
                    {
                        "match": {
                            "uid": user_id
                        }
                    }
                ]
            }
        }

        # execute query
        r = client.search(index='books-ratings', query=query, size=10000)

        s = 0
        cnt = 0
        # get user's grade
        user_rating = int(r['hits']['hits'][0]['_source']['rating'])
        for h in r['hits']['hits']:
            # grade > 0
            some_user_rate = int(h['_source']['rating'])
            if some_user_rate:
                s += some_user_rate
                cnt += 1

        # checking for division by zero
        if cnt != 0:
            return (s / cnt), user_rating
        else:
            return s, user_rating

    client = connect_to_elastic()

    # get all the books matching keyword
    books = get_data_from_elastic_simple('books', keyword)

    avgs = []
    ratings = []
    # separate lists for user's book ratings and avg's
    for isbn in books['isbn']:
        mean, rate = get_isbn_rating_avg(isbn)
        avgs += [mean]
        ratings += [rate]

    books['rating'] = ratings

    if activate_nn:
        # books with no rating by user
        fil = (books['rating'] == 0)
        # check if there are unrated books
        if fil.value_counts()[0]:
            books.loc[fil, 'rating'] = teach_data(user_id, books[fil])

    score = []
    # formulating and calculating the scores
    for i in range(len(books)):
        score += [(0.65 * books.loc[i, 'score']) + (0.25 * books.loc[i, 'rating']) + (0.1 * avgs[i])]

    # replace with the new custom scores
    books['score'] = score

    # sort by 'score'
    return books.sort_values(by=['score'], ascending=False)


def get_data_to_teach_user(user_id):
    client = connect_to_elastic()
    res = client.search(index='books-ratings', query={"match": {"uid": user_id}}, size=10000)

    array = []
    # get all the books(isbn) for a user into a list
    for hit in res['hits']['hits']:
        # keep rated by user books
        user_rating = int(hit['_source']['rating'])
        if user_rating:
            isbn_rating = (hit['_source']['isbn'], user_rating)
            array.append(isbn_rating)

    book_titles = []
    summaries = []
    ratings = []
    book_isbn = []
    # checking if all books exists in index 'books'
    for t in array:
        res = client.search(index='books', query={"match": {"isbn": t[0]}})
        # if specific book exists in index 'books'
        if res['hits']['total']['value']:
            # get the rating
            ratings.append(int(t[1]))

            for hit in res['hits']['hits']:
                book_titles += [hit['_source']['book_title']]
                summaries += [hit['_source']['summary']]
                book_isbn += [hit['_source']['isbn']]

    # convert data to pandas df
    d = {'isbn': book_isbn, 'book_title': book_titles, 'summary': summaries, 'rating': ratings}
    return pd.DataFrame(d)


def teach_data(user_id, zero_books):
    # candidates for ML: 131837, 135458, 124078, 254, random: 1 -> 145451

    user_nz_data = get_data_to_teach_user(user_id)

    # return zeros if there are not enough data
    if len(user_nz_data) <= 1:
        raise Exception("User's train data are not enough")
    # Information Retrieval
    user_nz_data['title_summary'] = user_nz_data['book_title'] + ' ' + user_nz_data['summary']
    # split 80-20 the (known) data for training and testing
    train, test = train_test_split(user_nz_data.loc[:, ['title_summary', 'rating']], test_size=0.3, random_state=42)
    print(len(train), len(test))
    # these are the data for training the model
    train_x = [t_s for t_s in train['title_summary']]
    train_y = [int(rate) for rate in train['rating']]

    # these are the data for testing/predicting the model
    test_x = [t_s for t_s in test['title_summary']]
    test_y = [int(rate) for rate in test['rating']]

    # finding the dictionary
    # and converting words to matrices
    vectorizer = CountVectorizer()
    # Dimensions of document-term-matrix are n x m
    # where n the total summaries
    # and m the total words from corpus
    train_x_vectors = vectorizer.fit_transform(train_x)

    test_x_vectors = vectorizer.transform(test_x)

    # scale data
    scaler = MaxAbsScaler()
    train_x_vectors = scaler.fit_transform(train_x_vectors)
    test_x_vectors = scaler.transform(test_x_vectors)

    # RandomForestClassifier
    # Default Classifier
    clf_rfc = RandomForestClassifier()
    clf_rfc.fit(train_x_vectors, train_y)

    # LogisticRegression classifier
    clf_log = LogisticRegression()
    clf_log.fit(train_x_vectors, train_y)

    # DecisionTreeClassifier
    clf_dec = DecisionTreeClassifier()
    clf_dec.fit(train_x_vectors, train_y)

    # linear SVC
    clf_svm = svm.SVC(kernel='linear', C=4)
    clf_svm.fit(train_x_vectors, train_y)

    print("Mean Accuracies: ")
    print("1. RandomForest: ", clf_rfc.score(test_x_vectors, test_y))
    print("2. LogisticRegression: ", clf_log.score(test_x_vectors, test_y))
    print("3. DecisionTree: ", clf_dec.score(test_x_vectors, test_y))
    print("4. SVM: ", clf_svm.score(test_x_vectors, test_y))
    pick = int(input("Classifier: "))

    zero_books_x = [t_s for t_s in (zero_books.loc[:, 'book_title'] + ' ' + zero_books.loc[:, 'summary'])]
    zero_books_x_vectors = vectorizer.transform(zero_books_x)

    # Pick the most suitable model (out of 4)
    if pick == 1:
        return clf_rfc.predict(zero_books_x_vectors)
    elif pick == 2:
        return clf_log.predict(zero_books_x_vectors)
    elif pick == 3:
        return clf_dec.predict(zero_books_x_vectors)
    elif pick == 4:
        return clf_svm.predict(zero_books_x_vectors)


def fetch_to_cluster(size):
    client = connect_to_elastic()

    match_query = {"match_all": {}}

    # execute query and return matches
    res = client.search(index='books', query=match_query, size=size)

    summary = []
    isbns = []
    for hit in res['hits']['hits']:
        summary.append(hit['_source']['summary'])
        isbns.append(hit['_source']['isbn'])

    # aggregation for each isbn up to 100 users
    aggr = {
        "aggs": {
            "top_hits": {
                "size": 100,
                "_source": ['uid', 'rating']
            }
        }
    }

    users = []
    # for each book
    for isbn in isbns:
        match_isbn = {"match": {"isbn": isbn}}
        res = client.search(index='books-ratings', query=match_isbn, aggregations=aggr, size=0)
        temp_list = []
        # get users and ratings
        for hit in res['aggregations']['aggs']['hits']['hits']:
            match_user = {"match": {"uid": hit['_source']['uid']}}
            userRes = client.search(index='books-users', query=match_user, size=1)
            # if user exists
            if userRes['hits']['hits']:
                temp = userRes['hits']['hits'][0]['_source']
                temp_list.append((temp['location'], temp['age'], hit['_source']['rating']))
        users.append(temp_list)

    return pd.DataFrame({"summary": [s for s in summary], "users": [user for user in users]},
                        columns=['summary', 'users'])


def find_opt_k(x):
    k_rng = range(1, 15)
    sse = []
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(x)
        sse.append(km.inertia_)

    choice = input('Plot Elbow Graph? (Y/N) --> ')
    if choice.upper() == 'Y':
        plt.xlabel('K')
        plt.ylabel('SSE')
        plt.title("Elbow Method For Optimal K")
        plt.plot(k_rng, sse)
        plt.show()

    kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")

    return kl.elbow


def clusterData(metric, userId=None):
    def clear_them(df, i):
        countries = []
        ratings = []
        # return data from cluster i as df
        tempdf = df[df['cluster'] == i].loc[:, ['users']]
        usa_countries = ['pender', 'washington,', 'florida,', 'missouri,', 'republic', 'california,', 'carolina,', 'massachusetts,', 'nebr,', 'tennessee,', 'states', 'pennsylvania,', 'texas,', 'ohio,', 'york,']
        for index, row in tempdf.iterrows():
            # if not empty
            if row['users']:
                # for each user in each book
                for t in row['users']:
                    # country = ",".join(t[0].split(", ")[-2:])
                    country = t[0].split()[-1]
                    # if country registered and rate not 0
                    if len(country) > 2 and int(t[2]):
                        if country in usa_countries:
                            country = 'usa'
                        countries.append(country)
                        ratings.append(int(t[2]))

        # group data
        group_it = pd.DataFrame({"country": [c for c in countries], "rating": [r for r in ratings]},
                                columns=['country', 'rating']) \
            .value_counts(['country', 'rating']).reset_index()
        group_it.rename({group_it.columns[-1]: 'times'}, axis=1, inplace=True)

        return group_it

    # fetch summaries
    df = fetch_to_cluster(int(input('Books to fetch: ')))

    # convert summaries to vectors and normalize them
    vec = CountVectorizer()
    X = vec.fit_transform(df['summary']).toarray()

    # dim reduction with SVD
    xSvd = PCA(2).fit_transform(X)

    # fit them to kmeans
    opt_k = find_opt_k(xSvd)

    if metric == 'cosine_similarity':
        # data normalization
        xSvd = normalize(xSvd)
        # calculate magnitudes and divide by it
        length = np.sqrt((xSvd ** 2).sum(axis=1))[:, None]
        xSvd = xSvd / length

        # produce k-means
        kmeans = KMeans(n_clusters=opt_k).fit(xSvd)

        # calculate centroids
        len_ = np.sqrt(np.square(kmeans.cluster_centers_).sum(axis=1)[:, None])
        centroids = kmeans.cluster_centers_ / len_
    elif metric == 'euclidean_distance':
        # produce k-means
        kmeans = KMeans(n_clusters=opt_k).fit(xSvd)
        centroids = kmeans.cluster_centers_
    else:
        raise SyntaxError('Choose one of the following: "euclidean_distance", "cosine_distance"')

    df['cluster'] = kmeans.predict(xSvd)

    # Getting unique clusters
    u_clusters = np.unique(df['cluster'])

    # plotting the results:
    for i in u_clusters:
        plt.scatter(xSvd[df['cluster'] == i, 0], xSvd[df['cluster'] == i, 1], label=i)

    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='*', label='centroid')
    # convert metric input to graph title
    words = metric.replace('_', ' ').split()
    title = words[0].capitalize() + " " + words[1].capitalize()
    plt.title(title)
    plt.legend()
    plt.show()

    # form data for heat map for each cluster
    for i in range(0, opt_k):
        plot_df = clear_them(df, i)
        plot_df = plot_df.pivot("country", "rating", "times")
        sns.heatmap(plot_df, linewidths=.3, yticklabels=True)
        plt.yticks()
        plt.title('Cluster ' + str(i))
        plt.show()


def main():
    # pd.set_option("max_columns", 3)
    # pd.set_option("max_rows", 100)
    while True:
        print("1. Load document to elastic")
        print("2. Retrieve data from elastic")
        print("3. Cluster Data")
        print("4. Exit")
        choice = int(input("Choice: "))
        if choice == 1:
            file_name = input("Give filename")
            index_name = input("Give index name")
            put_data_to_elastic(file_name, index_name)

        elif choice == 2:
            keyword = input("Give keyword:")

            print("1. Search with default metric")
            print("2. Search with custom metric")
            choice = int(input("Choice: "))
            if choice == 1:
                books = get_data_from_elastic_simple('books', keyword)
                print('DEFAULT MATCH QUERY METRIC'.center(180, '='))
                print(books.loc[:, ['book_title', 'book_author', 'score']])
                print(180 * "=")
            elif choice == 2:
                user = int(input("Give user id: "))
                ch = input("Combine neural network? (Y/N) --> ").upper()
                if ch == 'Y':
                    state = True
                else:
                    state = False
                books = get_data_from_elastic_custom(keyword, user, activate_nn=state)
                print('CUSTOM MATCH QUERY METRIC'.center(180, '='))
                print(books.loc[:, ['book_title', 'book_author', 'score']])
                print(180 * "=")
        elif choice == 3:
            print('1. Euclidean Distance')
            print('2. Cosine Similarity')
            ch = int(input('Choice: '))
            dist = 'cosine_similarity'
            if ch == 1:
                dist = 'euclidean_distance'
            clusterData(dist)
        else:
            break


main()