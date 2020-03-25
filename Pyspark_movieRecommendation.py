# Collaborative Filtering based on item-based algorithm (cosine similarity)


import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

# load the movie names and genre from data

def load_MovieNames():
    
    movieNames = {}
    with open('movies.dat') as f:
        for line in f:
            fields = line.split('::')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
            
    return movieNames

def make_pair((user, ratings)): # userRatings:(userID, ((movieID, rating), (movieID, rating)))
    
    (movie_1, rating_1) = ratings[0] # ((movieID, rating), (movieID, rating))
    (movie_2, rating_2) = ratings[1] # ((movieID, rating), (movieID, rating))
    
    return ((movie_1, movie_2), (rating_1, rating_2))

def filter_dup((userID, ratings)):
    
    (movie_1, rating_1) = ratings[0]
    (movie_2, rating_2) = ratings[1]
    
    return movie_1 < movie_2

# Calculate the cosine similarity
def cos_sim(rating_pair):
    
    num = 0
    sum_xx = sum_yy = sum_xy = 0
    for rating_X, rating_Y in rating_pair:
        sum_xx += rating_X * rating_X
        sum_yy += rating_Y * rating_Y
        sum_xy += rating_X * rating_Y
        num += 1

    up = sum_xy
    down = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (down):
        score = (up / (float(down)))

    return (score, num)


conf = SparkConf()
sc = SparkContext(conf = conf)


name = loadMovieNames()

data = sc.textFile('ml-1m/ratings.dat')

# Map ratings to key / value pairs: (user ID,(movie ID, rating))
ratings = data.map(lambda line: line.split('::')).map(lambda line: (int(line[0]), (int(line[1]), float(line[2]))))



# set the partition
ratings_part = ratings.partitionBy(100)

# Self-join to find every combination.
joined_ratings = ratings_part.join(ratings_part)

# RDD now consists of (userID,((movieID, rating), (movieID, rating)))

# Filter out duplicate pairs
joined_ratings = joined_ratings.filter(filter_dup)

# Now key by (movie1, movie2) pairs.
movie_pairs = joined_ratings.map(make_pair).partitionBy(100)

# ((movie1, movie2), (rating1, rating2))
# Now collect all ratings for each movie pair and compute similarity
movie_ratings = movie_pairs.groupByKey()

# ((rating1, rating2), (rating1, rating2)) ...
# compute similarities.
movie_sim = movie_ratings.mapValues(cos_sim).persist()


movie_sim.sortByKey()
movie_sim.saveAsTextFile("movie-sims")


if (len(sys.argv) > 1):

    score_threshold = 0.95 # set the threshold
    times = 1000 # the number of users giving the rating 

    movieID = int(sys.argv[1])

    results = moviePairSimilarities.filter(lambda((pair,sim)):         
                                                   (pair[0] == movieID or pair[1] == movieID)         
                                                   and sim[0] > score_threshold and sim[1] > times)

    # Sort by quality score.
    results = results.map(lambda((pair,sim)): (sim, pair)).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + name[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result 
        similar_movie = pair[0]
        if (similar_movie == movieID):
            similar_movie = pair[1]
        print(f'''movie:{name[similar_movie]} score:{str(sim[0])} reviews:{str(sim[1])}''')

