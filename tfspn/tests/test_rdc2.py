
from pdn.independenceptest import getIndependentGroups

import numpy

from tfspn.rdc import rdc


def test_pyrdc_vs_Rrdc_bernoulli_data_monodim():

    numpy.random.seed(42)

    gen = numpy.random.poisson(5, 1000)
    gen2 = numpy.random.poisson(5, 1000)
    
    genSorted = numpy.sort(gen)
    
    genmixture = numpy.hstack((gen, numpy.random.poisson(25, len(gen))))
    genmixture2 = numpy.hstack((gen2, numpy.random.poisson(25, len(gen2))))
    
    genmixtureSorted = numpy.sort(genmixture)
    
    negdependency = numpy.transpose(numpy.hstack((numpy.vstack(([0] * len(gen), gen)), numpy.vstack((gen, [0] * len(gen))))))
    
    genmixture3 = numpy.hstack((numpy.random.poisson(2, 1000), numpy.random.poisson(13, 2000)))
    negdependencymixture = numpy.transpose(numpy.hstack((numpy.vstack(([0] * len(genmixture3), genmixture3)), numpy.vstack((genmixture3, [0] * len(genmixture3))))))
    negdependencymixture = negdependencymixture[~numpy.all(negdependencymixture == 0, axis=1)]
    
    independent = numpy.transpose(numpy.vstack((gen, gen2)))
    independentmixture = numpy.transpose(numpy.vstack((genmixture, genmixture2)))
    
    opposite = numpy.transpose(numpy.vstack((genSorted, genSorted[::-1])))
    oppositemixture = numpy.transpose(numpy.vstack((genmixtureSorted, genmixtureSorted[::-1])))
    
    cooccur = numpy.transpose(numpy.vstack((gen, gen)))
    cooccurmixture = numpy.transpose(numpy.vstack((genmixture, genmixture)))

    print()
    print("cooccur", rdc(cooccur[:,0], cooccur[:,1]))

    print("negdependency", rdc(negdependency[:,0], negdependency[:,1]))
    print("negdependencymixture", rdc(negdependencymixture[:,0], negdependencymixture[:,1]) )
    
    print("independent", rdc(independent[:,0], independent[:,1]) )
    print("independentmixture", rdc(independentmixture[:,0], independentmixture[:,1]) )
    
    print("opposite", rdc(opposite[:,0], opposite[:,1]) )
    print("oppositemixture", rdc(oppositemixture[:,0], oppositemixture[:,1]) )
    
    print("cooccur", rdc(cooccur[:,0], cooccur[:,1]) )
    print("cooccurmixture", rdc(cooccurmixture[:,0], cooccurmixture[:,1]) )

    print(getIndependentGroups(independentmixture, 0.05, "poisson"))


test_pyrdc_vs_Rrdc_bernoulli_data_monodim()