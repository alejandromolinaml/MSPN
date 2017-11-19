trim <- function (x) gsub("^\\s+|\\s+$", "", x)

remove.dots <- function (x) gsub("\\.", "", x)

process.row <- function(row){
  fname <- row[1]
  ftype <- remove.dots(trim(row[2]))
  if (length(row) == 3)
  {
    fdomain <- split(remove.dots(row[3]), ',')
    print(class(fdomain))
  }
  else
  {
    fdomain <- ''
  }
  row[1] <- fname
  row[2] <- ftype
  row[3] <- fdomain
  row
}

collect.names <- function(row){
  trim(row[1])
}

collect.families <- function(row){
  trim(remove.dots(row[2]))
}

collect.domains <- function(row){
  if (length(row) == 3)
  {
    trim(remove.dots(row[3]))
  }
  else
  {
    ''
  }
}

load.feature.info <- function(data.path){
  df <- read.delim(data.path, sep=':',col.names=c("feature.name","feature.type","feature.domain"), header=F)
  names <- apply(df, 1, collect.names)
  families <- apply(df, 1, collect.families)
  domains <- apply(df, 1, collect.domains)
  # print(domains)
  domains <- lapply(domains, function(x) strsplit(x, ',')[[1]])
  # print(domains)
  list(names, families, domains)
}

data.transformer <- function(data.frame, feature.names, feature.families, feature.domains){
  # str(data.frame)
  for (i in 1:length(feature.names))
  {
    # print(data.frame[,feature.names[i]])
    #print(cat('processing', i, feature.names[i], feature.families[i]))
    # print(str(data.frame[,feature.names[i]]))
    
    if (feature.families[i] == 'continuous')
    {
      data.frame[,feature.names[i]] <- as.numeric(data.frame[,feature.names[i]])
    }
    else if (feature.families[i] == 'discrete')
    {
      # print(cat(feature.names[i], 'discrete'))
      
      data.frame[,feature.names[i]] <- factor(data.frame[,feature.names[i]], levels=feature.domains[[i]], ordered=T)
      # print('discrete')
      # print(str(data.frame[,feature.names[i]]))
    }
    else if (feature.families[i] == 'categorical')
    {
      # f<- factor(data.frame[,feature.names[i]], levels=feature.domains[i], ordered=F)
      # data.frame[,feature.names[i]] <- as.character(data.frame[,feature.names[i]])
      # print(data.frame[,feature.names[i]])
      # f<- factor(data.frame[,feature.names[i]])
      data.frame[,feature.names[i]] <- factor(data.frame[,feature.names[i]], levels=feature.domains[[i]], ordered=F)
      # print('categorical')
      # print(data.frame[,feature.names[i]])
    }
  }
  data.frame
}

load.dataset.splits <- function(data.path){
  
  info <- load.feature.info(paste(data.path, '.features', sep=''))
  train <- read.table(paste(data.path, '.train.data', sep=''), sep=",", header=F, col.names=info[[1]], fill=FALSE, strip.white=T, check.names=FALSE)
  valid <- read.table(paste(data.path, '.valid.data', sep=''), sep=",", header=F, col.names=info[[1]], fill=FALSE, strip.white=T, check.names=FALSE)
  test <- read.table(paste(data.path, '.test.data', sep=''), sep=",", header=F, col.names=info[[1]], fill=FALSE, strip.white=T, check.names=FALSE)
  
  train <- data.transformer(train, info[[1]], info[[2]], info[[3]])
  print(str(train))
  valid <- data.transformer(valid, info[[1]], info[[2]], info[[3]])
  test <- data.transformer(test, info[[1]], info[[2]], info[[3]])
  
  return (list(train=train, valid=valid, test=test))
}
