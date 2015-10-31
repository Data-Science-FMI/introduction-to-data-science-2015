install.packages("tm")
install.packages("dplyr")
install.packages("wordcloud")
install.packages("caret")
install.packages("klaR")
install.packages("doMC")

library(klaR)
library(dplyr)
library(tm)
library(wordcloud)
library(caret)
library(doMC)

registerDoMC()

sms.raw <- read.csv("data/sms_spam.csv", stringsAsFactors = FALSE)
sms.raw$type <- factor(sms.raw$type)

table(sms.raw$type)

sms.corpus <- Corpus(VectorSource(sms.raw$text))
sms.corpus.clean <- sms.corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
sms.dtm <- DocumentTermMatrix(sms.corpus.clean)

wordcloud(sms.corpus.clean, min.freq = 40, random.order = FALSE)

spam <- subset(sms.raw, type == "spam")
ham <- subset(sms.raw, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

train.index <- createDataPartition(sms.raw$type, p=0.75, list=FALSE)

sms.raw.train <- sms.raw[train.index,]
sms.raw.test <- sms.raw[-train.index,]
sms.corpus.clean.train <- sms.corpus.clean[train.index]
sms.corpus.clean.test <- sms.corpus.clean[-train.index]
sms.dtm.train <- sms.dtm[train.index,]
sms.dtm.test <- sms.dtm[-train.index,]

prop.table(table(sms.raw.train$type))
prop.table(table(sms.raw.test$type))


sms.dict <- findFreqTerms(sms.dtm.train, lowfreq=5)
sms.train <- DocumentTermMatrix(sms.corpus.clean.train, list(dictionary=sms.dict))
sms.test <- DocumentTermMatrix(sms.corpus.clean.test, list(dictionary=sms.dict))

# modified sligtly fron the code in the book
convert.counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("no", "yes"))
}
sms.train <- sms.train %>% apply(MARGIN=2, FUN=convert.counts)
sms.test <- sms.test %>% apply(MARGIN=2, FUN=convert.counts)


ctrl <- trainControl(method="cv", 10)
sms.nb.model <- train(sms.train, sms.raw.train$type, method="nb",
                    trControl=ctrl)


sms.nb.predictions <- predict(sms.nb.model, sms.test)
cm1 <- confusionMatrix(sms.nb.predictions, sms.raw.test$type, positive="spam")
cm1


sms.nb.model2 <- train(sms.train, sms.raw.train$type, method="nb", 
                    tuneGrid=data.frame(.fL=1, .usekernel=FALSE),
                    trControl=ctrl)
sms.nb.predictions2 <- predict(sms.nb.model2, sms.test)
cm2 <- confusionMatrix(sms.nb.predictions2, sms.raw.test$type, positive="spam")
cm2