ds <- Titanic
dsNeu <- as.data.frame.matrix(Titanic) 
summary(ds)
library(reshape2)
dsNeu2 <- melt(Titanic)
Titanic
View(Titanic)
