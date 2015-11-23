#install.packages("slam")
#install.packages("SnowballC")
#install.packages("topicmodels")
#install.packages("textcat")
library(tm)
library(dplyr)
library(doMC)
library(stringr)
library(RColorBrewer)
library(wordcloud)
library(ggplot2)
library(SnowballC)
library(slam)
library(topicmodels)
library(textcat)

registerDoMC()

df <- read.csv("data/tweets.csv", header = F, stringsAsFactors = F, encoding = "UTF-8")
colnames(df) <- c("polarity", "tweet_id", "created_at", "query", "user", "text")
head(df)

random.tweets <- df[sample(nrow(df), 20000, replace=FALSE),]

random.tweets$language <- textcat(iconv(enc2utf8(random.tweets$text), sub = "byte"))
random.tweets$language[is.na(random.tweets$language)] <- "english"

english.tweets <- random.tweets[random.tweets$language == "english", ]
english.tweets$text = iconv(enc2utf8(english.tweets$text), sub = "byte")

head(english.tweets)

ggplot(data=english.tweets, aes(x=factor(polarity, labels = c("negative", "positive")))) + 
    geom_bar(binwidth=1, fill="#2196F3") + 
    labs(x="emotion", y="number of tweets") +
    ggtitle("General distribution of tweet sentiment") +
    theme(plot.title = element_text(size=18, face="bold"))

## Preprocessing

createCorpus <- function(text) {
  corpus <- Corpus(VectorSource(text))
  corpus.clean <- corpus %>%
    tm_map(content_transformer(tolower)) %>%
    tm_map(content_transformer(function(x) gsub("http[^[:space:]]*", "", x))) %>%
    tm_map(removeNumbers) %>%
    tm_map(removeWords, c(stopwords(kind = "en"), stopwords(kind = "smart"), "via", "available", "rt", "cc", "just")) %>%
    tm_map(removePunctuation) %>%
    tm_map(stripWhitespace)
  return(corpus.clean)
}

polarities = levels(factor(english.tweets$polarity))
polarities.count = length(polarities)
documents = rep("", polarities.count)

for (i in 1:polarities.count)
{
  text = english.tweets[english.tweets$polarity == polarities[i],]$text
  documents[i] = paste(text, collapse=" ")
}

documents = removeWords(documents, stopwords("english"))
doc.corpus = Corpus(VectorSource(documents))
doc.tdm = as.matrix(TermDocumentMatrix(doc.corpus))
colnames(doc.tdm) <- c("negative", "positive")

corpus.clean <- createCorpus(english.tweets$text)

corpus.stemmed <- tm_map(corpus.clean, content_transformer(stemDocument))
 
## Check tweets after processing

for (i in c(1:15)) {
  cat(paste0("[", i, "] "))
  writeLines(strwrap(as.character(corpus.clean[[i]]), 80)) 
}

comparison.cloud(doc.tdm, colors = c("#F44336","#4CAF50"), max.words = 200,
                 scale = c(3,.5), random.order = FALSE, title.size = 1.5)

## Analyze word frequencies

corpus.tdm <- TermDocumentMatrix(corpus.stemmed, control = list(minWordLength = 1))
inspect(corpus.tdm)
findFreqTerms(corpus.tdm, lowfreq = 1000)
corpus.freq <- row_sums(corpus.tdm, na.rm = T)
df.freq <- data.frame(term = names(corpus.freq), freq = corpus.freq)

ggplot(subset(df.freq, freq > 3000), aes(term, freq)) + 
  geom_bar(stat = "identity", fill="#2196F3") +
  labs(x="term", y="frequency") +
  ggtitle("Frequent terms in tweets") +
  theme(plot.title = element_text(size=18, face="bold"))
  
wordcloud(
  corpus.stemmed, max.words = 40, random.order = FALSE,
  colors = brewer.pal(6, "Dark2")
)

top.terms <- data.frame(term = stemCompletion(names(sort(row_sums(corpus.tdm), decreasing=T)[1:150]), corpus.clean),
                        freq = sort(row_sums(corpus.tdm), decreasing=T)[1:150], stringsAsFactors=F)

ggplot(top.terms[1:15,], aes(term, freq)) + 
  geom_bar(stat = "identity", fill="#2196F3") +
  labs(x="word", y="frequency") +
  ggtitle("Frequent words in tweets") +
  theme(plot.title = element_text(size=18, face="bold"))

## Find word associations

showAssociations <- function(corpus, association, dictionary, corlimit = 0.1, sep = ", ") {
  asocs <- findAssocs(corpus, association, corlimit)[[association]]
  cat(stemCompletion(names(sort(asocs)), dictionary = dictionary), sep = sep)
}

showAssociations(corpus.tdm, "smile", corpus.clean, corlimit = 0.12)
showAssociations(corpus.tdm, "task", corpus.clean, corlimit = 0.10)
showAssociations(corpus.tdm, "airport", corpus.clean)

## Analyze hashtags

all.hashtags = unlist(str_extract_all(english.tweets$text, "#\\w+"))
wordcloud(all.hashtags, max.words = 40, random.order = FALSE,
  colors = brewer.pal(6, "Dark2"), min.freq = 2)

## Hierarchical clustering

dense.tdm <- removeSparseTerms(corpus.tdm, sparse = 0.98)
dense.tdm.matrix <- as.matrix(dense.tdm)
dist.matrix <- dist(scale(dense.tdm.matrix))
fit <- hclust(dist.matrix, method = "ward.D")
plot(fit)

k <- 6

rect.hclust(fit, k = k)

dense.tdm.matrix.T <- t(dense.tdm.matrix)
kmeans.res <- kmeans(dense.tdm.matrix.T, k)
round(kmeans.res$centers, digits = 3)

for (i in 1:k) {
  cat(paste("cluster ", i, ": ", sep = ""))
  s <- sort(kmeans.res$centers[i, ], decreasing = T)
  cat(names(s)[1:5], "\n")
}

## LDA topic modeling

corpus.dtm <- as.DocumentTermMatrix(dense.tdm)

corpus.dtm.new <- corpus.dtm[row_sums(corpus.dtm) > 0, ]
lda <- LDA(corpus.dtm.new, k = 5)
(topics <- terms(lda, 3))
