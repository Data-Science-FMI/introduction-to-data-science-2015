# install.packages("car")
# install.packages("ggplot2")
# install.packages("Amelia")
# install.packages("stargazer")
# install.packages("corrplot")

library(car)
library(ggplot2)
library(MASS)
library(corrplot)
library(stargazer)

data = Prestige
head(data)
str(data)
dim(data)

data <- setNames(cbind(rownames(data), data, row.names = NULL), 
         c("role", "education", "income", "women", "prestige", "census", "type"))

head(data)

require(Amelia)
missmap(data, main="Missing Training Data Map", col=c("#FF4081", "#3F51B5"), legend=FALSE)

newdata <- data[,c(2:3)]
summary(newdata)

qplot(education, data = newdata, geom="histogram", binwidth=1) +
  labs(title = "Historgram of Average Years of Education") +
  labs(x ="Average Years of Education") +
  labs(y = "Frequency") +
  scale_y_continuous(breaks = c(1:20), minor_breaks = NULL) +
  scale_x_continuous(breaks = c(6:16), minor_breaks = NULL) +
  geom_vline(xintercept = mean(newdata$education), show_guide=TRUE, color="red", labels="Average") +
  geom_vline(xintercept = median(newdata$education), show_guide=TRUE, color="blue", labels="Median")

barplot(table(data$type))

qplot(income, data = newdata, geom="histogram", fill=I("#4CAF50"), binwidth=1000) +
  labs(title = "Historgram of Average Income") +
  labs(x ="Average Income") +
  labs(y = "Frequency") +
  scale_y_continuous(breaks = c(1:20), minor_breaks = NULL) +
  scale_x_continuous(breaks = c(0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000), minor_breaks = NULL) +
  geom_vline(xintercept = mean(newdata$income), show_guide=TRUE, color="#D32F2F", labels="Average") +
  geom_vline(xintercept = median(newdata$income), show_guide=TRUE, color="#1976D2", labels="Median")

qplot(education, income, data = newdata, main = "Relationship between Income and Education") +
  scale_y_continuous(breaks = c(1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 25000), minor_breaks = NULL) +
  scale_x_continuous(breaks = c(6:16), minor_breaks = NULL)

# Simple linear regression model

education.c = scale(newdata$education, center=TRUE, scale=FALSE)
education.c = as.data.frame(education.c)
names(education.c)[1] = "education.c"
newdata = cbind(newdata, education.c)
mod1 = lm(income ~ education.c, data = newdata)
summary(mod1)

qplot(education.c, income, data = newdata, main = "Relationship between Income and Education") +
  stat_smooth(method="lm", col="#D32F2F", se=FALSE) +
  scale_y_continuous(breaks = c(1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 25000), minor_breaks = NULL)

# Multiple linear regression

newdata = data[,c(2:6)]
summary(newdata)
plot(newdata, pch=16, col="#2196F3", main="Matrix Scatterplot of Income, Education, Women and Prestige")

education.c = scale(newdata$education, center=TRUE, scale=FALSE)
prestige.c = scale(newdata$prestige, center=TRUE, scale=FALSE)
women.c = scale(newdata$women, center=TRUE, scale=FALSE)

new.c.vars = cbind(education.c, prestige.c, women.c)
newdata = cbind(newdata, new.c.vars)
names(newdata)[6:8] = c("education.c", "prestige.c", "women.c" )
summary(newdata)

model2 = lm(income ~ education.c + prestige.c + women.c, data=newdata)
summary(model2)
residualPlot(model2)

newdatacor = cor(newdata[1:4])
corrplot(newdatacor, method = "number")

model3 = lm(log(income) ~ prestige.c + I(prestige.c^2) + women.c , data=newdata)
summary(model3)
residualPlot(model3)
