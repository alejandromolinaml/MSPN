#
# loading the adult dataset
data <- read.table('../TF_SPN-old/TFSPN/src/mlutils/datasets/adult.data', sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education",  "education_num","marital", "occupation", "relationship", "race","sex","capital_gain", "capital_loss", "hr_per_week","country", "income"),fill=FALSE,strip.white=T)

cat('Adult dataset loaded, containing', dim(data), 'entries\n')
cat('Column types', str(data), '\n')

## setting types to either factors, numeric or ordered factors
data <- transform(data, age=as.numeric(age), fnlwgt=as.numeric(fnlwgt), education_num=as.numeric(education_num), capital_gain=as.numeric(capital_gain), capital_loss=as.numeric(capital_loss), hr_per_week=as.numeric(hr_per_week))
data <- transform(data, type_employer=as.character(type_employer), education=as.character(education), marital=as.character(marital), occupation=as.character(occupation), relationship=as.character(relationship), race=as.character(race), sex=as.character(sex), country=as.character(country), income=as.character(income))
cat('Column types', str(data), '\n')


## learning a hybrid BN with MoTBFs
require(MoTBFs)

## first learn a DAG structure with PC (it discretizes!)
# dag <- LearningHC(data, numIntervals=3)
dag <- LearningHC(data)

## then learn the parameters as Mixtures of truncated Polinomials
intervals <- 3
max_params <- 5
potential <- "MOP"
P1 <- MoTBFs_Learning(graph=dag, data=data, POTENTIAL_TYPE=potential, maxParam=max_params, numIntervals=intervals)
printBN(P1)

ll <- logLikelihood.MoTBFBN(P1, data)
print(cat('MoP log likelihood', ll))

## then learn the parameters as Mixtures of truncated Exponentials
intervals <- 3
max_params <- 5
potential <- "MTE"
P2 <- MoTBFs_Learning(graph=dag, data=data, POTENTIAL_TYPE=potential, maxParam=max_params, numIntervals=intervals)
printBN(P2)

ll <- logLikelihood.MoTBFBN(P2, data)
print(cat('MTE log likelihood', ll))



