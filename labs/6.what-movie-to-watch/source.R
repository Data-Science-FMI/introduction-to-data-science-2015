# install.packages("recommenderlab")

library(ggplot2)
library(recommenderlab)

data(MovieLense)
MovieLense
as(MovieLense[1:15, 1:5], "matrix")

image(sample(MovieLense, 500), main = "Raw ratings")

qplot(getRatings(MovieLense), binwidth = 1, 
      main = "Histogram of ratings", xlab = "Rating")
summary(getRatings(MovieLense))

qplot(getRatings(normalize(MovieLense, method = "Z-score")),
      main = "Histogram of normalized ratings", xlab = "Rating") 
summary(getRatings(normalize(MovieLense, method = "Z-score")))

qplot(rowCounts(MovieLense), binwidth = 10, 
      main = "Movies Rated on average by user", 
      xlab = "# of users", 
      ylab = "# of movies rated")

qplot(colMeans(MovieLense), binwidth = .1, 
      main = "Mean rating for each movie", 
      xlab = "Rating", 
      ylab = "# of movies")

recommenderRegistry$get_entries(dataType = "realRatingMatrix")


scheme <- evaluationScheme(MovieLense, method = "split", train = .9,
                           k = 1, given = 10, goodRating = 4)


model.popular <- Recommender(getData(scheme, "train"), method = "POPULAR")
model.ibcf <- Recommender(getData(scheme, "train"), method = "IBCF")
model.ubcf <- Recommender(getData(scheme, "train"), method = "UBCF")

predict.popular <- predict(model.popular, getData(scheme, "known"), type = "ratings")
predict.ibcf <- predict(model.ibcf, getData(scheme, "known"), type = "ratings")
predict.ubcf <- predict(model.ubcf, getData(scheme, "known"), type = "ratings")

predict.err <- rbind(calcPredictionAccuracy(predict.popular, getData(scheme, "unknown")), 
                     calcPredictionAccuracy(predict.ubcf, getData(scheme, "unknown")), 
                     calcPredictionAccuracy(predict.ibcf, getData(scheme, "unknown")))

rownames(predict.err) <- c("POPULAR", "UBCF", "IBCF")

predict.err


# Predict good movies for user 10 and 11
barplot(table(MovieLense[10:10]@data@x))
barplot(table(MovieLense[11:11]@data@x))
recom <- predict(model.ubcf, MovieLense[10:11], n=100)
best <- bestN(recom, n=3)
as(best, 'list')

algorithms <- list(
  "random items" = list(name="RANDOM", param=list(normalize = "Z-score")),
  "popular items" = list(name="POPULAR", param=list(normalize = "Z-score")),
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score",
                                                 method="Cosine",
                                                 nn=50, minRating=3)),
  "item-based CF" = list(name="IBCF", param=list(normalize = "Z-score"
  ))
  
)

results <- evaluate(scheme, algorithms, n=c(1, 3, 5, 10, 15, 20))

runs <- c("RANDOM", "POPULAR", "UBCF", "IBCF")

for(i in 1:4) {
  print(runs[i])
  print(getConfusionMatrix(results@.Data[[i]])[[1]])
} 

plot(results, annotate = 1:4, legend="topleft")

plot(results, "prec/rec", annotate=2:4)