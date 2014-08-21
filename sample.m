% A sample script to show how to use the train() and test() function.

load robot

C = 81;
p = 2;
rLambda = 16;
a = 1.5;

[svmModel, theta] = train(trainSample, C, rLambda, a, p);
[cpr, cprEachTask] = test(trainSample, testSample, svmModel, theta);
