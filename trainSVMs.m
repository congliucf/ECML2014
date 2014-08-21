function [ model ] = trainSVMs(trainData, theta, C)

numTasks = size(trainData, 2);
numKernels = size(trainData(1).kernelMatrix, 3);

%% Train SVM based on the MKL setting
for t = 1:numTasks
    dataMatrix = zeros(size(trainData(t).kernelMatrix(:, :, 1)));
    for m = 1:numKernels
        dataMatrix = dataMatrix + theta(m) .* (trainData(t).kernelMatrix(:, :, m));
    end
    dataMatrix = [(1:size(trainData(t).label, 1))', dataMatrix];
    parameter = ['-t 4 -c ', num2str(C)];
    model(t) = svmtrain(trainData(t).label, dataMatrix, parameter);
end
    
end

