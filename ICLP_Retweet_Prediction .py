#imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#reading data from excel file
file = pd.read_excel("tweets.xlsx")

#looking at data to get a understanding of the data
file.head()

#getting total number of rows and columns
file.shape

#getting stats foe each column
file.describe()

#to check empty cells in each feature variable
file.count()

# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 3, sharey=True)
file.plot(kind='scatter', x='UserFollowersCount', y='TweetRetweetCount',ax=axs[0],figsize=(16,8))
file.plot(kind='scatter', x='TweetFavoritesCount', y='TweetRetweetCount', ax=axs[1])
file.plot(kind='scatter', x='UserFriendsCount', y='TweetRetweetCount', ax=axs[2])

fig, axs = plt.subplots(1, 3, sharey=True)
file.plot(kind='scatter', x='UserListedCount', y='TweetRetweetCount',ax=axs[0],figsize=(16,8))
file.plot(kind='scatter', x='UserTweetCount', y='TweetRetweetCount', ax=axs[1])
file.plot(kind='scatter', x='MacroIterationNumber', y='TweetRetweetCount', ax=axs[2])

#creating a new feature of count of hashtags within a tweet
Hashtags_count = []
file.TweetHashtags.fillna("a",inplace=True)
for i in range(len(file.TweetHashtags)):
    if file.TweetHashtags[i]== "a":
        Hashtags_count.append(0)
    else:
        Hashtags_count.append((len([j for j in file.TweetHashtags[i].split(",")])))

#adding new feature to the dataframe
file['Hashtags_count']=Hashtags_count

# lets take X be the feature matrix and Y be the target variable
Y= file.TweetRetweetCount
X= file.ix[:, file.columns != 'TweetRetweetCount']

# create a Python list of feature names
feature_cols =['UserFollowersCount','TweetFavoritesCount','UserFriendsCount','UserListedCount','UserTweetCount','MacroIterationNumber','Hashtags_count']

# use the list to select a subset of the original DataFrame
X1 = X[feature_cols]

#splitting the file into train and test (75% rows in train and 25% in test)
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, random_state=1)

#checking train and test files shape
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


#Linear regression on training data and checking on test data
# instantiate
linreg = LinearRegression()

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, Y_train)

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)

# pair the feature names with the coefficients
list(zip(feature_cols, linreg.coef_))

# make predictions on the testing set
Y_pred = linreg.predict(X_test)

#calculating RMSE
print(np.sqrt(mean_squared_error(Y_test, Y_pred)))

print("Actual vs Predicted using Linear Regression")
pd.DataFrame({'Predicted Retweet':Y_pred,'Actual Retweets':Y_test})

#Linear Regression using Cross Validation
lm = LinearRegression()
scores = cross_val_score(lm, X1, Y, cv=13, scoring='neg_mean_squared_error')
print(scores)

# fix the sign of MSE scores
mse_scores = -scores
print(mse_scores)

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)

# calculate the average RMSE
print(rmse_scores.mean())


#Random Forest on Training Data and Checking on Test Data
rf = RandomForestRegressor()
rf.fit(X_train,Y_train)

Y_pred = rf.predict(X_test)

#calculating RMSE
print(np.sqrt(mean_squared_error(Y_test, Y_pred)))

print("Actual vs Predicted using Random Forest")
pd.DataFrame({'Predicted Retweet':Y_pred,'Actual Retweets':Y_test})

