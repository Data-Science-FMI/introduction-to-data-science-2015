install.packages("randomForest")
install.packages("readr")
install.packages("ggbiplot")
install.packages("devtools")
install.packages("caret")
install.packages("stats")

library(readr)
library(randomForest)
library(ggplot2)
library(ggbiplot)
library(caret)

library(devtools)
install_github("vqv/ggbiplot")

train <- read.csv("data/train.csv")
test <- read.csv("data/test.csv")

colors<-c('white','black')
cus_col<-colorRampPalette(colors=colors)

## Plot the average image of each digit
par(mfrow=c(4,3),pty='s',mar=c(1,1,1,1),xaxt='n',yaxt='n')
all_img<-array(dim=c(10,28*28))
for(di in 0:9)
{
  print(di)
  all_img[di+1,]<-apply(train[train[,1]==di,-1],2,sum)
  all_img[di+1,]<-all_img[di+1,]/max(all_img[di+1,])*255
  
  z<-array(all_img[di+1,],dim=c(28,28))
  z<-z[,28:1] ##right side up
  image(1:28,1:28,z,main=di,col=cus_col(256))
}

train_1000 <- train[sample(nrow(train), size = 1000),]
ggplot(data = train_1000, aes(x = pixel152, y = pixel153, color = factor(label))) + geom_point()


pc <- prcomp(train_1000[, -1], cor=TRUE, scores=TRUE)
ggbiplot(pc, obs.scale = 1, var.scale = 1, groups = factor(train_1000$label),
         ellipse = TRUE, circle = F, var.axes = F) + 
  scale_color_discrete(name = '') + 
  theme(legend.direction = 'horizontal', legend.position = 'top')

#####
# Function
#####
marginToDigit <- function(digitMatrix){
  val <- c(rep(0,28*4))
  valName <- c(rep(0,28*4))
  idx <- 1
  
  # Left
  for(i in 1:nrow(digitMatrix)){
    j <- 1
    while(j <= ncol(digitMatrix) && digitMatrix[i,j]==0){
      j <- j+1
    }  
    val[idx] <- j
    valName[idx] <- paste("marginLeft",i, sep="")
    idx <- idx+1
  }
  
  #Right
  for(i in 1:nrow(digitMatrix)){
    j <- ncol(digitMatrix)
    while(j >= 1 && digitMatrix[i,j]==0){
      j <- j-1
    }  
    j <- ncol(digitMatrix)+1-j
    val[idx] <- j
    valName[idx] <- paste("marginRight",i, sep="")
    idx <- idx+1
  }
  
  # Top
  for(j in 1:ncol(digitMatrix)){
    i <- 1
    while(i <= nrow(digitMatrix) && digitMatrix[i,j]==0){
      i <- i+1
    }  
    val[idx] <- i
    valName[idx] <- paste("marginTop",j, sep="")
    idx <- idx+1
  }
  
  # Bottom
  for(j in 1:ncol(digitMatrix)){
    i <- ncol(digitMatrix)
    while(i >= 1 && digitMatrix[i,j]==0){
      i <- i-1
    }  
    i <- ncol(digitMatrix)+1-i
    val[idx] <- i
    valName[idx] <- paste("marginBottom",j, sep="")
    idx <- idx+1
  }
  
  marginValue <- val
  names(marginValue) <- valName
  return(marginValue)
}

marginToMarginDiff <- function(row){
  marginDiff.val <- c(diff(row[1:28]),
                      diff(row[29:56]), 
                      diff(row[57:84]), 
                      diff(row[85:112]))
  marginDiff.name <- c(paste("leftMarginDiff",1:27, sep=""),
                       paste("rightMarginDiff",1:27, sep=""),
                       paste("topMarginDiff",1:27, sep=""),
                       paste("bottomMarginDiff",1:27, sep=""))
  marginDiff <- marginDiff.val
  names(marginDiff) <- marginDiff.name
  return (marginDiff)
}

lineCount <- function(digitMatrix){
  horizontalCount <- apply(digitMatrix, 1, function(x){
    tempDiff <- diff(c(0,x))
    return( sum(tempDiff > 0) ) 
  })
  verticalCount <- apply(digitMatrix, 2, function(x){
    tempDiff <- diff(c(0,x))
    return( sum(tempDiff > 0) )
  })
  lineCountName <- c(paste("horizontalLineCount",1:28,sep=""),
                     paste("verticalLineCount",1:28,sep=""))
  lineCountVal <- c(horizontalCount, verticalCount)
  names(lineCountVal) <- lineCountName
  return (lineCountVal)
}

getMostImportantFeature <- function(rfObject, numOfFeatures){
  imp <- importance(rfObject)
  imp <- data.frame(imp)
  imp <- imp[order(-imp$MeanDecreaseGini),,drop=F]
  return(row.names(imp)[1:numOfFeatures])
}

#####
# Main Script
#####


cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

# train
cat("Building train features\n")
allRowMean <- apply(train[,-1],1,mean)
train.rowMean <- data.frame(train$label, allRowMean)
names(train.rowMean) <- c("label", "pixel.mean")
rm(allRowMean)

allPixelInBinary <- ifelse(train[,-1]<train.rowMean$pixel.mean,"0","1")
train.binaryPixel <- data.frame(label=train$label,allPixelInBinary)
for(i in 2:ncol(train.binaryPixel)){
  levels(train.binaryPixel[,i]) <- c("0","1")
}
rm(allPixelInBinary)

marginVal <- apply(train.binaryPixel[,-1], 1, function(row){
  mat <- matrix(as.numeric(row), 28, 28, byrow=T)
  return(marginToDigit(mat))
}) 
train.margin <- cbind(label=train$label, apply(marginVal, 1, function(x) return(x)))
rm(marginVal)

marginDiff <- apply(train.margin[,-1], 1, marginToMarginDiff)
train.marginDiff <- cbind(label=train$label, apply(marginDiff, 1, function(x) return(x)))
rm(marginDiff)

train.highPixelCount <- data.frame(label=train$label, highPixelCount=apply(train.binaryPixel, 1, function(x) return(sum(as.numeric(x)))))

lineCountVal <- apply(train.binaryPixel[,-1], 1, function(row){
  mat <- matrix(as.numeric(row), 28, 28, byrow=T)
  return(lineCount(mat))
})
train.lineCount <- cbind(label=train$label, apply(lineCountVal, 1, function(x) return(x)))
rm(lineCountVal)

# test
cat("Building test features\n")
allRowMean <- apply(test, 1, mean)
test.rowMean <- data.frame(pixel.mean = allRowMean)
rm(allRowMean)

allPixelInBinary <- ifelse(test < test.rowMean$pixel.mean,"0","1")
test.binaryPixel <- data.frame(allPixelInBinary)
for(i in 1:ncol(test.binaryPixel)){
  levels(test.binaryPixel[,i]) <- c("0","1")
}
rm(allPixelInBinary)

marginVal <- apply(test.binaryPixel, 1, function(row){
  mat <- matrix(as.numeric(row), 28, 28, byrow=T)
  return(marginToDigit(mat))
}) 
test.margin <- apply(marginVal, 1, function(x) return(x))
rm(marginVal)

marginDiff <- apply(test.margin, 1, marginToMarginDiff)
test.marginDiff <- apply(marginDiff, 1, function(x) return(x))
rm(marginDiff)

test.highPixelCount <- data.frame(highPixelCount = apply(test.binaryPixel, 1, function(x) return(sum(as.numeric(x)))))

lineCountVal <- apply(test.binaryPixel, 1, function(row){
  mat <- matrix(as.numeric(row), 28, 28, byrow=T)
  return(lineCount(mat))
})
test.lineCount <- apply(lineCountVal, 1, function(x) return(x))
rm(lineCountVal)

# combine feture for trainning and testing

train.combine <- data.frame(train.binaryPixel, 
                            pixel.mean=train.rowMean$pixel.mean, 
                            highPixelCount = train.highPixelCount$highPixelCount,
                            train.margin[,-1], 
                            train.marginDiff[,-1],
                            train.lineCount[,-1])

test.combine <- data.frame(test.binaryPixel, 
                           pixel.mean=test.rowMean$pixel.mean, 
                           highPixelCount = test.highPixelCount$highPixelCount,
                           test.margin, 
                           test.marginDiff,
                           test.lineCount)

# training

numTrain <- 10000
numTrees <- 25
nodeSize <- 3

trainning.labels <- as.factor(train.combine[,1])
trainning.data <- train.combine[,-1]
rf <- randomForest(trainning.data, 
                   trainning.labels, 
                   xtest=test.combine, 
                   ntree=numTrees, 
                   do.trace=T,
                   nodesize=nodeSize)

predictions <- data.frame(ImageId=1:nrow(test), Label=levels(trainning.labels)[rf$test$predicted])

head(predictions)

write_csv(predictions, "submission.csv")
