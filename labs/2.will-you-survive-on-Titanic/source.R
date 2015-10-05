library(ggplot2)

train.df <- read.csv("data/train.csv", stringsAsFactors=FALSE, na.strings=c("NA", ""))

## Understanding

dim(train.df)
head(train.df)
names(train.df)

require(Amelia)
missmap(train.df, main="Missing Training Data Map", col=c("#FF4081", "#3F51B5"), legend=FALSE)

## Plotting

train.df$Survived <- factor(train.df$Survived, levels=c(1,0))
levels(train.df$Survived) <- c("Survived", "Persished")
train.df$Pclass <- as.factor(train.df$Pclass)
levels(train.df$Pclass) <- c("1st Class", "2nd Class", "3rd Class")
train.df$Sex <- factor(train.df$Sex, levels=c("female", "male"))
levels(train.df$Sex) <- c("Female", "Male")

barplot(table(train.df$Survived))
barplot(table(train.df$Survived, train.df$Sex), xlab = "Gender", ylab = "Number of People",
        main = "Passenger Survival by Gender",
        legend=rownames(table(train.df$Survived, train.df$Sex)), beside=TRUE)
qplot(Fare, data=train.df, colour=Survived, geom="density",
      main="Passenger Survival by Fare Price")

mosaicplot(train.df$Pclass ~ train.df$Survived, main="Passenger Survival by Class",
           color=c("#4CAF50", "#F44336"), shade=FALSE,  xlab="", ylab="",
           off=c(0), cex.axis=1.4)

mosaicplot(train.df$Sex ~ train.df$Survived, main="Passenger Survival by Gender",
           color=c("#4CAF50", "#F44336"), xlab="", ylab="",
           off=c(0), cex.axis=1.4)

hist(train.df$Age, xlab="age", main="Passenger Age", breaks = 25, col="#2196F3")

## First model - all perish

prop.table(table(train.df$Survived))

test.df <- read.csv("data/test.csv", stringsAsFactors=FALSE)
test.row.count <- nrow(test.df)
test.df$Survived <- rep(0, test.row.count)

prediction.df <- data.frame(PassengerId = test.df$PassengerId, Survived = test.df$Survived)
write.csv(prediction.df, file = "submissions/all-perish.csv", row.names = FALSE)

## Second model - females survive

prop.table(table(train.df$Sex, train.df$Survived),1)
test.df$Survived <- 0
test.df$Survived[test.df$Sex == 'female'] <- 1

prediction.df <- data.frame(PassengerId = test.df$PassengerId, Survived = test.df$Survived)
write.csv(prediction.df, file = "submissions/females-survive.csv", row.names = FALSE)

## Adding a feature - child

train.df <- read.csv("data/train.csv", stringsAsFactors=FALSE)
train.df$Child <- 0
train.df$Child[train.df$Age < 18] <- 1

aggregate(Survived ~ Child + Sex, data=train.df, FUN=function(x) {sum(x)/length(x)})

train.glm <- glm(Survived ~ Pclass + Sex + Age + Child, family = binomial, data = train.df)
summary(train.glm)

test.df <- read.csv("data/test.csv", stringsAsFactors=FALSE)

test.df$Child <- 0
test.df$Child[test.df$Age < 18] <- 1

p.hats <- predict.glm(train.glm, newdata = test.df, type = "response", na.action = na.pass)

survival <- vector()
for(i in 1:length(p.hats)) {
  if(is.na(p.hats[i])) {
    survival[i] <- 0
  } else if(p.hats[i] > 0.5) {
    survival[i] <- 1
  } else {
    survival[i] <- 0
  }
}

prediction.df <- data.frame(PassengerId = test.df$PassengerId, Survived = survival)
write.csv(prediction.df, file = "submissions/log-regression.csv", row.names = FALSE)