# Code from the tutorial here:
# https://www.kaggle.com/c/facial-keypoints-detection/details/getting-started-with-r

#---------------- read data ---------------------

d.train <- read.csv("data/training.csv", stringsAsFactors=F)
# Compactly Display the Structure of an Arbitrary R Object
str(d.train)
# ?str
head(d.train)

#---------------- trim data ---------------------

# Unfortunately the rightmost column is quite long, 
# so the output is not very readable.
# Save the column as another variable, and remove it from d.train.
im.train <- d.train$Image

#  Assigning NULL to a column removes it from the dataframe.
d.train$Image <- NULL

head(d.train)

#----------------- trim data 2: Image -----------------

# For each image (i.e. in each row) it contains a long string of numbers,
# where each number represents the intensity of a pixel in the image.
im.train[1] # First value in the column

# To analyze these further, we convert these strings to integers 
# by splitting them and converting the result to integer.

# strsplit: splits the string, 
# unlist: simplifies its output to a vector of strings 
# as.integer: converts it to a vector of integers.

as.integer(unlist(strsplit(im.train[1], " ")))

# Iterate through each row in im.train 
# and apply the string to integers conversion above.
# We can therefore use a multi core approach using a new library.

library(foreach)
im.train <- foreach(im=im.train, .combine=rbind) %do% {
  as.integer(unlist(strsplit(im, " ")))
}
str(im.train)

# Repeat the process for test.csv.
d.test  <- read.csv("data/test.csv", stringsAsFactors=F)
im.test <- foreach(im = d.test$Image, .combine=rbind) %do% {
  as.integer(unlist(strsplit(im, " ")))
}
d.test$Image <- NULL

#------------------- save data ---------------------

# Save the data as a R data file at this point, 
# so no need to repeat this process again

save(d.train, im.train, d.test, im.test, file='data.Rd')

# load data
# load('data.Rd')

#------------------- visualize --------------------

# To visualize each image, we thus need to first convert 
# these 9216 integers into a 96x96 matrix

im <- matrix(data=rev(im.train[1,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))

# check1
points(96-d.train$nose_tip_x[1],         96-d.train$nose_tip_y[1],         col="red")
points(96-d.train$left_eye_center_x[1],  96-d.train$left_eye_center_y[1],  col="blue")
points(96-d.train$right_eye_center_x[1], 96-d.train$right_eye_center_y[1], col="green")

# check2
for(i in 1:nrow(d.train)) {
  points(96-d.train$nose_tip_x[i], 96-d.train$nose_tip_y[i], col="red")
}

# check3
idx <- which.max(d.train$nose_tip_x) # Easter egg for min value.
im  <- matrix(data=rev(im.train[idx,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))
points(96-d.train$nose_tip_x[idx], 96-d.train$nose_tip_y[idx], col="red")

#--------------- simple benchmark ------------------

# One of the simplest things to try is to compute 
# the mean of the coordinates of each keypoint in the training set
# and use that as a prediction for all images. 
# This is a very simplistic algorithm, as it completely ignores the images, 
# but we can use it a starting point to build a first submission.

colMeans(d.train, na.rm=T) # colMeans (na.rm=T tells colMeans to ignore missing values)

# To build a submission file we need to apply 
# these computed coordinates to the test instances.

p           <- matrix(data=colMeans(d.train, na.rm=T), nrow=nrow(d.test), ncol=ncol(d.train), byrow=T)
colnames(p) <- names(d.train)
predictions <- data.frame(ImageId = 1:nrow(d.test), p)
head(predictions)

#---------------- shape up the submission ----------------

library(reshape2)
submission <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")
head(submission)

example.submission <- read.csv("data/IdLookupTable.csv")
sub.col.names = c("RowId", "Location")
example.submission$Location <- NULL
submission <- merge(example.submission, submission, all.x=T, sort=F)
submission <- submission[, sub.col.names]
write.csv(submission, file="submission_means.csv", quote=F, row.names=F)


#---------------- using image patches ------------------

# The idea is to extract a patch around this keypoint in each image, and average the result.
# This average_patch can then be used as a mask to search for the keypoint in test images.

patch_size <- 10
search_size <- 2

coordinate.names <- gsub("_x", "", names(d.train)[grep("_x", names(d.train))])


# for each one, compute the average patch
mean.patches <- foreach(coord = coordinate.names) %dopar% {
  cat(sprintf("computing mean patch for %s\n", coord))
  coord_x <- paste(coord, "x", sep="_")
  coord_y <- paste(coord, "y", sep="_")
  
  # compute average patch
  patches <- foreach (i = 1:nrow(d.train), .combine=rbind) %do% {
    im  <- matrix(data = im.train[i,], nrow=96, ncol=96)
    x   <- d.train[i, coord_x]
    y   <- d.train[i, coord_y]
    x1  <- (x-patch_size)
    x2  <- (x+patch_size)
    y1  <- (y-patch_size)
    y2  <- (y+patch_size)
    if ( (!is.na(x)) && (!is.na(y)) && (x1>=1) && (x2<=96) && (y1>=1) && (y2<=96) )
    {
      as.vector(im[x1:x2, y1:y2])
    }
    else
    {
      NULL
    }
  }
  matrix(data = colMeans(patches), nrow=2*patch_size+1, ncol=2*patch_size+1)
}

# for each coordinate and for each test image, find the position that best correlates with the average patch
p <- foreach(coord_i = 1:length(coordinate.names), .combine=cbind) %dopar% {
  # the coordinates we want to predict
  coord   <- coordinate.names[coord_i]
  coord_x <- paste(coord, "x", sep="_")
  coord_y <- paste(coord, "y", sep="_")
  
  # the average of them in the training set (our starting point)
  mean_x  <- mean(d.train[, coord_x], na.rm=T)
  mean_y  <- mean(d.train[, coord_y], na.rm=T)
  
  # search space: 'search_size' pixels centered on the average coordinates 
  x1 <- as.integer(mean_x)-search_size
  x2 <- as.integer(mean_x)+search_size
  y1 <- as.integer(mean_y)-search_size
  y2 <- as.integer(mean_y)+search_size
  
  # ensure we only consider patches completely inside the image
  x1 <- ifelse(x1-patch_size<1,  patch_size+1,  x1)
  y1 <- ifelse(y1-patch_size<1,  patch_size+1,  y1)
  x2 <- ifelse(x2+patch_size>96, 96-patch_size, x2)
  y2 <- ifelse(y2+patch_size>96, 96-patch_size, y2)
  
  # build a list of all positions to be tested
  params <- expand.grid(x = x1:x2, y = y1:y2)
  
  # for each image...
  r <- foreach(i = 1:nrow(d.test), .combine=rbind) %do% {
    if ((coord_i==1)&&((i %% 100)==0)) { cat(sprintf("%d/%d\n", i, nrow(d.test))) }
    im <- matrix(data = im.test[i,], nrow=96, ncol=96)
    
    # ... compute a score for each position ...
    r  <- foreach(j = 1:nrow(params), .combine=rbind) %do% {
      x     <- params$x[j]
      y     <- params$y[j]
      p     <- im[(x-patch_size):(x+patch_size), (y-patch_size):(y+patch_size)]
      score <- cor(as.vector(p), as.vector(mean.patches[[coord_i]]))
      score <- ifelse(is.na(score), 0, score)
      data.frame(x, y, score)
    }
    
    # ... and return the best
    best <- r[which.max(r$score), c("x", "y")]
  }
  names(r) <- c(coord_x, coord_y)
  r
}

# prepare file for submission
predictions        <- data.frame(ImageId = 1:nrow(d.test), p)
submission         <- melt(predictions, id.vars="ImageId", variable.name="FeatureName", value.name="Location")

example.submission <- read.csv("data/IdLookupTable.csv")
sub.col.names = c("RowId", "Location")
example.submission$Location <- NULL

submission <- merge(example.submission, submission, all.x=T, sort=F)
submission <- submission[, sub.col.names]

write.csv(submission, file="submission_search.csv", quote=F, row.names=F)