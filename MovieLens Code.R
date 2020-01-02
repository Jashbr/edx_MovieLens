
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

##set.seed(1, sample.kind="Rounding")
set.seed(1)
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,] #train set
temp <- movielens[test_index,] #test set

# Creating a filtered test set, making sure not to include movieId and userId not in edx set 
##Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, removed)

#MovieLens Data
head(movielens,10)

#To check unique values
movielens %>% summarize(n_users = n_distinct(userId),n_movies = n_distinct(movieId))

#Structure of train set
str(edx)

#Structure of test set
str(validation)

#Top 10 popularly rated movies
edx %>%
  count(movieId) %>%
  arrange(desc(n)) %>%  
  slice(1:10)

#Movie rated oftentimes
edx %>%
  filter(movieId == 296) %>%
  head()

#Distribution of Movies
edx %>%
  count(movieId) %>% 
  ggplot(aes(n)) +
  geom_histogram(fill = "grey30",bins = 20, color = "cadetblue") + scale_x_log10() + 
  labs(y = "Number of movies",x = "Number of ratings") +
  ggtitle("Distribution of Rated movies")

# Movies rated by few number of users
edx %>%
  group_by(movieId) %>%
  summarize(n_ratings=n()) %>%
  group_by(n_ratings) %>%
  summarize(n_mov = n()) %>%
  filter(n_ratings < 6) %>%
  ggplot(aes(x=n_ratings,y=n_mov,fill=n_ratings)) + 
  geom_col() +
  labs(y = "Number of movies",x = "Number of ratings") +
  theme_minimal()

#Distribution of Users rating movies
edx %>%
  count(userId) %>% 
  ggplot(aes(n)) +
  geom_histogram(fill = "grey30",bins = 20, color = "cadetblue") + scale_x_log10() + 
  labs(y = "Number of users",x = "Number of ratings") +
  ggtitle("Distribution of Users rating movies")

#Mean movie ratings given by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(fill = "grey30",bins = 30, color = "cadetblue") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()

#Distribution of Whole star and Half star ratings
edx %>%
  ggplot(aes(x=rating)) +
  geom_histogram(fill = "grey30",bins = 20, color = "cadetblue") + 
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  labs(y = "Number of ratings",x = "Ratings") +
  ggtitle("Distribution of Whole star & Half star rating")

#Modeling Approaches
#Average of all ratings
mu <- mean(edx$rating)                  
mu

#Predicted Rating Validation
validation %>% filter(movieId == 231) %>% group_by(movieId) %>% summarize(m_i = mean(rating))

#Model1 - RMSE
rmse_1 <- RMSE(validation$rating, mu) 
rmse_1
rmse_results <- tibble(method = "Model1 : Average Movie Rating", RMSE = rmse_1) 

#Movie Effect
#Rating Prediction
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

movie_avgs %>% ggplot(aes(b_i)) +
  geom_histogram(fill = "grey30",bins = 20, color = "cadetblue") + 
  ylab("Number of movies") +
  ggtitle("Number of movies with computed b_i")

predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>% 
  mutate(pred = mu + b_i) %>% 
  pull(pred)

#Model2 - RMSE
rmse_2 <- RMSE(predicted_ratings, validation$rating)
rmse_2
rmse_results <- bind_rows(rmse_results,tibble(method="Model2 : Movie Effect", RMSE = rmse_2 ))

#Introducing User Effects
user_avgs<- edx %>%
  left_join(movie_avgs, by='movieId') %>% group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- validation%>% 
  left_join(movie_avgs, by='movieId') %>% left_join(user_avgs, by='userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

#Model3 - RMSE
rmse_3 <- RMSE(predicted_ratings, validation$rating)
rmse_3
rmse_results <- bind_rows(rmse_results,tibble(method="Model3 : Movie and User Effect",RMSE = rmse_3))

#Regularization
#Need for Regularization
validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i),mu_plus_bi = mu + b_i) %>%
  arrange(desc(abs(residual))) %>%  
  slice(1:10) %>% 
  select(movieId,rating,mu_plus_bi,b_i) %>%
  distinct()
edx %>% count(movieId) %>% 
  left_join(movie_avgs, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  slice(1:5) %>% 
  select(movieId,n)

#Choosing Tuning Parameter lambda
#Regularized Movie Effect
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  predicted_ratings <-
    validation %>%
    left_join(b_i, by = "movieId") %>% mutate(pred = mu + b_i ) %>% pull(pred)
  return(RMSE(predicted_ratings, validation$rating)) })
qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)] 
lambda

#Model4 - RMSE
rmse_results <- bind_rows(rmse_results,tibble(method="Model4 : Regularized Movie effect model",RMSE = min(rmses)))

#Regularized Movie and User Effect
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>% group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-
    validation %>%
    left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId") %>% mutate(pred = mu + b_i + b_u) %>% pull(pred)
  return(RMSE(predicted_ratings, validation$rating)) })
qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)] 
lambda

#Model5 - RMSE
rmse_results <- bind_rows(rmse_results,tibble(method="Model5 : Regularized Movie and User effect model",RMSE = min(rmses)))

#Methods - RMSE
rmse_results %>% knitr::kable()
