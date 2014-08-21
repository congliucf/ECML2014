function [cpr, cprEachTask] = test(trainSample, testSample, svmModel, theta)
% A function that performs the prediction phase of the multi-task multiple
% kernel learning model, which should be trained by the train() function.
%
% Input:
%    trainSample - Shoule be exactly the same as the input of the train()
%                  function.
%     testSample - A 1-by-T sized struct, where T is the number of tasks. 
%                  In the t-th struct, it should contain the test data  
%                  and the true labels of the t-th task, i.e., 
%                  testSample(t) is comprised with testSample(t).data and 
%                  testSample(t).label, for all t = 1, ..., T.
%                  
%                  testSample(t).data should be an Mt-by-D sized matrix, 
%                  where Mt is the number of test samples of the t-th 
%                  task, and D is the number of features for each test
%                  sample. 
%                  
%                  testSample(t).label should be an Mt-by-1 sized vector, 
%                  with each element to be either 1 or -1, representing the
%                  true label of the corresponding test sample of the t-th 
%                  task.
%       svmModel - The output of the train() function.
%          theta - The output of the train() function.
%
% Output:
%            cpr - The overall averaged correct prediction rate for all 
%                  tasks: 
%                  (# correct predictions from all tasks) / (# number of 
%                  samples from all tasks)
%    cprEachTask - A 1-by-T sized vector. The t-th element contains correct
%                  prediction rate of the t-th task.


numTasks = size(trainSample, 2);

for t = 1:numTasks
    testData(t).kernelMatrix = genKernelMatrices(testSample(t).data, trainSample(t).data);
    testData(t).label = testSample(t).label;
end

numKernels = size(testData(1).kernelMatrix, 3);

predictedLabel = [];
realLabel = [];
for t = 1:numTasks
    dataMatrix = zeros(size(testData(t).kernelMatrix(:, :, 1)));
    for m = 1:numKernels
        dataMatrix = dataMatrix + theta(m) .* testData(t).kernelMatrix(:, :, m);
    end
    dataMatrix = [(1:size(dataMatrix, 1))', dataMatrix];
    
    [predict, accuracy, decValues] = svmpredict(testData(t).label, dataMatrix, svmModel(t));

    predictedLabel = [predictedLabel; predict];
    realLabel = [realLabel; testData(t).label];
    cprEachTask(t) = accuracy(1);
end

cpr = sum(predictedLabel == realLabel) / size(predictedLabel, 1);

end

