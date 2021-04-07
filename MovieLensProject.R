##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes as we will load packages and library

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# We will now download the MovieLens data from internet and store in variable "d1" and then create two data frames as "ratings" with 4 columns and "movies" with three columns  

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile() 
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Since i am using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId), title = as.character(title),genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>%  semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# The above chunk of code gives a partition of the dataset for training and testing our dataset. It also removes the unnecessary files from the working directory

##########################################################################################################################
# Exploratory analysis
##########################################################################################################################

# Quick preview of the edx dataset shows 6 columns
head(edx)

# timestamp needs to be converted if used, and release year will need to be split from the title if useful for prediction
# genres is a single pipe-delimited string containing the various genre categories a movie might be categorized under, and this will need to be split out if it affects rating outcome

# Examining the distribution of "rating" in the training "edx" data set.
table(edx$rating)
###From above output, we can find the rating range as: 0.5, 1, 1.5, 2, 2.5, 3, 3.5,  4, 4.5, 5

# We can also see the proportion of each ratings
prop.table(table(edx$rating))
###From above output, we can see that maximum ratings are "3" & "4" and same is also visible from bar plot 
barplot(table(edx$rating))

#Finding frequencies of users, movies & genres in the edx dataset
edx %>% summarize(users = n_distinct(userId), movies = n_distinct(movieId), genres = n_distinct(genres))

# Distribution of Movie Ratings
> edx %>% group_by(movieId) %>% summarize(n = n()) %>% ggplot(aes(n)) + geom_histogram(fill = "grey", color = "red", bins = 10) + scale_x_log10() + ggtitle("Movies Ratings Distribution")

#Distribution of Users
edx %>% group_by(userId) %>% summarize(n = n()) %>% ggplot(aes(n)) + geom_histogram(fill = "grey", color = "red", bins = 10) +
  scale_x_log10() + ggtitle("Distribution of Users Ratings")
### From above output, we can see that movie rating distribution follows a almost normal distribution 


#Distribution of ratings by year
edx %>% 
  mutate(release = str_sub(title, start = -5, end = -2)) %>%
  filter(as.numeric(release) > 1958) %>%
  group_by(movieId) %>% 
  summarize(n = n(), release = first(release)) %>%
  ggplot(aes(x = release, y = n)) +
  geom_boxplot(fill = "burlywood", alpha = 0.2) +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  xlab("Release Year") + ylab("Count") +
  ggtitle("Distribution of number of ratings by release year")
### From above output, we can see that between 1993 to 1995 has seen a much higher number of ratings than any other year. It also seems that movies released in the 1990-2000 periods have more ratings on average. However, the rate of ratings tend to decrease for newer movies. For these reasons we will split year out as it may affect our model 

# Modify the year as a column in the both datasets
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))


# In the following code, we extract the different genres values to examine its effect on the average rating.

genres <- edx %>%
  group_by(movieId) %>%
  summarize(r = mean(rating), title = title[1], genre = genres[1]) %>%
  separate_rows(genre, sep = "\\|") %>%
  group_by(genre) %>%
  summarize(r = mean(r)) %>%
  filter(!genre == "(no genres listed)")

genres %>%
  ggplot(aes(x=r, y=reorder(genre, r), fill=r)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_cartesian(xlim = c(0, 5)) +
  scale_fill_distiller(palette = "OrRd") +
  labs(x="Average Rating", y="Genres") +
  geom_vline(xintercept = mean(edx$rating), color = "blue4") +
  ggtitle("Distribution of Ratings by Genres")
### From above output, it seems that movies tagged with the genre "Film-Noir" tend to have higher ratings, while movies in the "Horror" and "Sci-Fi" genres tend to have lower ratings. The average rating is also plotted as a blue line as a reference.

# Method:
# The most suitable approach to this project to get a RMSE of less than 0.86490 would be the Optimal Least Squares Method.
# The essential part in this exercise is to arrive at the optimal tuning factor 'Lambda' at which the RMSE is minimal and also below 0.86490. 
#The following codes is used to determine the optimal tuning factor: (it will take few minutes to run)

#  Root Mean Square Error Loss Function
  RMSE <- function(true_ratings, predicted_ratings)
  {
       sqrt(mean((true_ratings - predicted_ratings)^2))
  }
  
  lambdas <- seq(0, 5, 0.25)

  rmses <- sapply(lambdas,function(l)
    {
   #Calculate the mean of ratings from the edx training set
         mu <- mean(edx$rating)
         
  #Adjust mean by movie effect and penalize low number on ratings
         pen_m <- edx %>% 
         group_by(movieId) %>%
         summarize(pen_m = sum(rating - mu)/(n()+l))
         
  #Adjust mean by user and movie effect and penalize low number of ratings
         pen_m_u <- edx %>% 
         left_join(pen_m, by="movieId") %>%
         group_by(userId) %>%
         summarize(pen_m_u = sum(rating - pen_m - mu)/(n()+l))
  
   #Adjust mean by user, movie,year effect and penalize low number of ratings
        pen_y <- edx %>%
        left_join(pen_m, by='movieId') %>%
        left_join(pen_m_u, by='userId') %>%
        group_by(year) %>%
        summarize(pen_y = sum(rating - mu - pen_m - pen_m_u)/(n()+l), n_y = n())  
         
  #Predict ratings in the training set to derive optimal penalty value 'lambda'
         predicted_ratings <- edx %>% 
         left_join(pen_m, by = "movieId") %>%
         left_join(pen_m_u, by = "userId") %>%
         left_join(pen_y, by ="year")%>%
         mutate(pred = mu + pen_m + pen_m_u + pen_y) %>%
         .$pred
         
         return(RMSE(predicted_ratings, edx$rating))
  })
 
  plot(lambdas, rmses)
  lambda <- lambdas[which.min(rmses)]  #This gives optimal lambda
  
#Now, the prediction on the validation set would be done based on the Lambda obtained.

#Derive the mean from the training set
  mu <- mean(edx$rating)
    
#Calculate movie effect with optimal lambda
  movie_effect <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = sum(rating - mu)/(n()+lambda), n_i = n())
  
#Calculate user effect with optimal lambda
 user_effect <- edx %>% 
 left_join(movie_effect, by='movieId') %>%
 group_by(userId) %>%
 summarize(b_u = sum(rating - mu - b_m)/(n()+lambda), n_u = n())
        
#Calculate year effect with optimal lambda
 year_reg <- edx %>%
 left_join(movie_effect, by='movieId') %>%
 left_join(user_effect, by='userId') %>%
 group_by(year) %>%
 summarize(b_y = sum(rating - mu - b_m - b_u)/(n()+lambda), n_y = n())
          
#Predict ratings on validation set
 predicted_ratings <- validation %>% 
 left_join(movie_effect, by='movieId') %>%
 left_join(user_effect, by='userId') %>%
 left_join(year_reg, by = 'year') %>%
 mutate(pred = mu + b_m + b_u + b_y) %>% 
 .$pred
 
model_rmse <- RMSE(validation$rating,predicted_ratings)

rmse_results <- data_frame(method="Regularized Movie, User, Year Effect Model", RMSE = model_rmse)
    
# Conclusion:
#The aim of the project was to predict movie ratings from a dataset of existing movie ratings. The optimal least square method turns out to be one of the better methods to the algorithm. As can be observed, the RMSE of 0.8566952 at a Lambda of 0.5.  
