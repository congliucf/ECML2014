ECML2014
========

This repository contains Matlab implementation of paper

`C. Li, M. Georgiopoulos and G. C. Anagnostopuolos, "Conic Multi-Task Classification", ECMLPKDD 2014.`

# Prerequisites

1. Please install and setup LIBSVM: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
2. Please install and setup CVX: http://cvxr.com/


# Suggested use:

1. Training:
  
    `[svmModel, theta] = train(trainSample, C, rLambda, a, p);`

2. Test:

    `[cpr, cprEachTask] = test(trainSample, testSample, svmModel, theta);`


Please refer to the train() and test() function for detailed explanation of each input/output.
