##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes as we will load packages and library

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
#if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org") 
#Some machines may not have Latex installation and hence need above package

library(tidyverse)
library(caret)
library(data.table)
#library(tinytex)

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

##########################################################
# Exploratory analysis
##########################################################

# Quick preview of the edx dataset shows 6 columns. "userId","movieID","rating","timestamp","title","genres" in the subset. Each row represent a single rating of a user for a single movie.
head(edx)

# Two Important Observations :
#1. Timestamp will need to be converted if used, and release year will need to be split from the title if used for prediction
#2. Genres is a single pipe-delimited string containing the various genre categories a movie might be categorized under, and this will need to be split out if it affects rating outcome

# A Summary of the subset confirms that there are no missing values.
summary(edx)

# Examining the distribution of "rating" in the training "edx" data set.
table(edx$rating)
###From above output, we can find the rating range as: 0.5, 1, 1.5, 2, 2.5, 3, 3.5,  4, 4.5, 5

# We can also see the proportion of each ratings
prop.table(table(edx$rating))
###From above output, we can see that maximum ratings are "3" & "4" and same is also visible from bar plot 
barplot(table(edx$rating))

#Finding frequencies of users, movies & genres in the edx dataset
edx %>% 
  summarize(users = n_distinct(userId), movies = n_distinct(movieId), genres = n_distinct(genres))

# Distribution of Movie Ratings
edx %>% 
  group_by(movieId) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(fill = "turquoise4", color = "white", bins = 10) + 
  scale_x_log10() + 
  ggtitle("Movies Ratings Distribution")

#Distribution of Users
edx %>% 
  group_by(userId) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(n)) + 
  geom_histogram(fill = "turquoise2", color = "white", bins = 10) + 
  scale_x_log10() + 
  ggtitle("Distribution of Users Ratings")
### From above output, we can see that movie rating distribution follows a almost normal distribution 


#Distribution of ratings by year
edx %>% 
  mutate(release = str_sub(title, start = -5, end = -2)) %>%
  filter(as.numeric(release) > 1958) %>%
  group_by(movieId) %>% 
  summarize(n = n(), release = first(release)) %>%
  ggplot(aes(x = release, y = n)) +
  geom_boxplot(fill = "turquoise", alpha = 0.2) +
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
  scale_fill_distiller(palette = "Blues") +
  labs(x="Average Rating", y="Genres") +
  geom_vline(xintercept = mean(edx$rating), color = "red") +
  ggtitle("Distribution of Ratings by Genres")
### From above output, it seems that movies tagged with the genre "Film-Noir" tend to have higher ratings, while movies in the "Horror" and "Sci-Fi" genres tend to have lower ratings. The average rating is also plotted as a red line as a reference.
# We will now try and model to account for biases based on - average rating, movie, users, year and genres

# Modeling Method:
# The most suitable approach to this project is to get a RMSE of less than 0.86490. 
# The essential part in this exercise is to arrive at the optimal tuning factor 'Lambda' at which the RMSE is minimal and also below 0.86490. 
# The following codes is used to determine the optimal tuning factor: (it will take few minutes to run)

# As instructed we will split the edx data set also into two sets - training and test sets, to experiment with multiple parameters

edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.15, list = FALSE)
train_edx <- edx[-edx_index,]
test_edx <- edx[edx_index,]

test_edx <- test_edx %>% 
  semi_join(train_edx, by = "movieId") %>%
  semi_join(train_edx, by = "userId")

rm(edx_index)

#  Root Mean Square Error Loss Function
  RMSE <- function(true_ratings, predicted_ratings)
  {
       sqrt(mean((true_ratings - predicted_ratings)^2))
  }
  
# Model 1: Rating Average
  mu <- mean(train_edx$rating)
  
  rat_avg_rmses <- RMSE(mu,test_edx$rating)
  
  rmse_results <- tibble(method = "Average rating", RMSE = min(rat_avg_rmses))
  rmse_results %>% knitr::kable()
#Since the Rating Average RMSE is 1.0611, which is very high compared to our target, for all other models we will try obatianing RMSE using regularization approach

# Model 2: Movie Effect with Regularization
  lambdas <- seq(0, 5, 0.25)
  
  movie_rat_rmses <- sapply(lambdas,function(l)
  {
      pen_m <- train_edx %>% 
      group_by(movieId) %>%
      summarize(pen_m = sum(rating - mu)/(n()+l))
    
  #Predict ratings in the TEST set 
      predicted_ratings <- test_edx %>% 
      left_join(pen_m, by = "movieId") %>%
      mutate(pred = mu + pen_m ) %>%
      .$pred
    
  return(RMSE(predicted_ratings,test_edx$rating))
  })
  rmse_results <- bind_rows(rmse_results,
                  tibble(method="Movie Effects Model",RMSE = min(movie_rat_rmses)))
  rmse_results %>% knitr::kable()

#Model 3: User + Movie Effect with Regularization
  user_movie_rat_rmses <- sapply(lambdas,function(l)
  {
    #Adjust mean by movie effect 
      pen_m <- train_edx %>% 
      group_by(movieId) %>%
      summarize(pen_m = sum(rating - mu)/(n()+l))
    
    #Adjust mean by user and movie effect 
       pen_m_u <- train_edx %>% 
       left_join(pen_m, by="movieId") %>%
       group_by(userId) %>%
       summarize(pen_m_u = sum(rating - mu - pen_m)/(n()+l))

    #Predict ratings in the TEST set 
      predicted_ratings <- test_edx %>% 
      left_join(pen_m, by = "movieId") %>%
      left_join(pen_m_u, by = "userId") %>%
      mutate(pred = mu + pen_m + pen_m_u) %>%
      .$pred
    
      return (RMSE(predicted_ratings,test_edx$rating))
  })
      rmse_results <- bind_rows(rmse_results,
                      tibble(method="User + Movie Effects Model",RMSE = min(user_movie_rat_rmses)))
      rmse_results %>% knitr::kable()  

      
# Model 4: Movie + User + Genre Effects Model
    genres_user_movie_rat_rmses <- sapply(lambdas,function(l)
    {
      #Adjust mean by movie effect 
      pen_m <- train_edx %>% 
      group_by(movieId) %>%
      summarize(pen_m = sum(rating - mu)/(n()+l))
    
    #Adjust mean by user and movie effect 
      pen_m_u <- train_edx %>% 
      left_join(pen_m, by="movieId") %>%
      group_by(userId) %>%
      summarize(pen_m_u = sum(rating - mu - pen_m)/(n()+l))
    
    #Adjust mean by user, movie,genres effect 
      pen_g <- train_edx %>%
      left_join(pen_m, by="movieId") %>%
      left_join(pen_m_u, by="userId") %>%
      group_by(genres) %>%
      summarize(pen_g = sum(rating - mu - pen_m - pen_m_u)/(n()+l))  
    
    #Predict ratings in the TEST set
      predicted_ratings <- test_edx %>% 
      left_join(pen_m, by = "movieId") %>%
      left_join(pen_m_u, by = "userId") %>%
      left_join(pen_g, by = "genres")%>%
      mutate(pred = mu + pen_m + pen_m_u + pen_g) %>%
      .$pred
        
      return(RMSE(predicted_ratings,test_edx$rating))
    })  
      rmse_results <- bind_rows(rmse_results,
                    tibble(method="Genres + User + Movie Effects Model",RMSE = min(genres_user_movie_rat_rmses)))
      rmse_results %>% knitr::kable()    

      
# Model 5: Movie + User + Genre + Year Effects Model  
   
   rg_yr_gn_us_mov_rat_rmses <- sapply(lambdas,function(l)
    {
      #Adjust mean by movie effect and penalize low number on ratings
         pen_m <- train_edx %>% 
         group_by(movieId) %>%
         summarize(pen_m = sum(rating - mu)/(n()+l))
         
     #Adjust mean by user and movie effect and penalize low number of ratings
         pen_m_u <- train_edx %>% 
         left_join(pen_m, by="movieId") %>%
         group_by(userId) %>%
         summarize(pen_m_u = sum(rating - mu - pen_m)/(n()+l))
  
    #Adjust mean by user, movie,genres effect and penalize low number of ratings
         pen_g <- train_edx %>%
         left_join(pen_m, by="movieId") %>%
         left_join(pen_m_u, by="userId") %>%
         group_by(genres) %>%
         summarize(pen_g = sum(rating - mu - pen_m - pen_m_u)/(n()+l))  
        
    #Adjust mean by user, movie,year & genres effect and penalize low number of ratings    
         pen_y <- train_edx %>%
         left_join(pen_m, by="movieId") %>%
         left_join(pen_m_u, by="userId") %>%
         left_join(pen_g, by="genres") %>%
         group_by(year) %>%
         summarize(pen_y = sum(rating - mu - pen_m - pen_m_u - pen_g)/(n()+l))
        
             
    #Predict ratings in the training set to derive optimal penalty value 'lambda'
         predicted_ratings <- test_edx %>% 
         left_join(pen_m, by = "movieId") %>%
         left_join(pen_m_u, by = "userId") %>%
         left_join(pen_g, by = "genres")%>%
         left_join(pen_y, by = "year") %>%
         mutate(pred = mu + pen_m + pen_m_u + pen_g + pen_y) %>%
         .$pred
         
         return(RMSE(predicted_ratings,test_edx$rating))
  })
 
  rmse_results <- bind_rows(rmse_results,
                  tibble(method="Year + Genres + User + Movie Effect Model",RMSE = min(rg_yr_gn_us_mov_rat_rmses)))
  rmse_results %>% knitr::kable()
  
  plot(lambdas, rg_yr_gn_us_mov_rat_rmses)
  lambda <- lambdas[which.min(rg_yr_gn_us_mov_rat_rmses)]  #This gives optimal lambda
  
#Now, the prediction on the validation set would be done based on the Lambda obtained.

#Calculate movie effect with optimal lambda
    movie_effect <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_m = sum(rating - mu)/(n()+lambda))
  
#Calculate user effect with optimal lambda
    user_effect <- edx %>% 
    left_join(movie_effect, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_m)/(n()+lambda))
        
#Calculate genres effect with optimal lambda
    genres_effect <- edx %>%
    left_join(movie_effect, by="movieId") %>%
    left_join(user_effect, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+lambda))
          
#Calculate year effect with optimal lambda
    year_effect <- edx %>%
    left_join(movie_effect, by="movieId") %>%
    left_join(user_effect, by="userId") %>%
    left_join(genres_effect, by="genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - mu - b_m - b_u - b_g)/(n()+lambda))    


#Predict ratings on validation set
    predicted_ratings <- validation %>% 
    left_join(movie_effect, by="movieId") %>%
    left_join(user_effect, by="userId") %>%
    left_join(genres_effect, by="genres") %>%
    left_join(year_effect, by ="year") %>%
    mutate(pred = mu + b_m + b_u + b_g + b_y) %>% 
   .$pred
 
final_model_rmse <- RMSE(validation$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                tibble(method="Year + Genres + Movie + User Effect Model after validation",
                RMSE = min(final_model_rmse)))
rmse_results  %>% knitr::kable()

# Results:
# The aim of the project was to predict movie ratings from a dataset of existing movie ratings. The optimal least square method turns out to be one of the better methods to the algorithm.
# We first split the data between edx and validation data sets, and then we split edx further into train and test data. We ran five models and calculated RMSE at every stage and progressively seen it immproved to optimized number of 0.86429.  The most effective prediction model was "Movie + User + Genre + Year Effects Model" where biases by movie, user, genre and year were used on the training set and then regularized.

# Conclusion:
#The entire MovieLens project helps understand how sometimes knowingly or unknowingly biases can creep in our most simplest of day to day actions. As a budding data scientist it should be our foremost responsibility to lookout for these biases and keep removing it from larger ML models