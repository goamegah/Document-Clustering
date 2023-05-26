library('clustrd')
# fonction de reducedKmeans
reducedKmeans<- function(path,k) {
data <- read.csv(path)
outRKM = cluspca(data, k, 2, method = "RKM", rotation = "varimax", scale = FALSE, nstart = 10)
#ploter avec 2 dimensions  
plot(outRKM, cludesc = TRUE)
}

# --------- Dataset classic 3 -----------#

# Representation Word2vec

path="C:/Users/ThinkPad/Desktop/Projet data2/classic3_word2vec.csv"
reducedKmeans(path,3)


# Representation Glove


path="C:/Users/ThinkPad/Desktop/Projet data2/classic3_glove.csv"
reducedKmeans(path,3)



# --------- Dataset bbc -----------#


# Representation Word2vec

path="C:/Users/ThinkPad/Desktop/Projet data2/bbc_word2vec.csv"

reducedKmeans(path,5)

# Representation Glove
path="C:/Users/ThinkPad/Desktop/Projet data2/bbc_glove.csv"

reducedKmeans(path,5)


# --------- Dataset classic4 -----------#


# Representation Word2vec

path="C:/Users/ThinkPad/Desktop/Projet data2/classic4_word2vec.csv"

reducedKmeans(path,4)

# Representation Glove
path="C:/Users/ThinkPad/Desktop/Projet data2/classic4_glove.csv"

reducedKmeans(path,4)


