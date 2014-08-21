function [ value ] = calFunctionValue( model, trainData, theta )
% A function that calculates the SVM function values for each task with 
% multiple kernels, given the corresponding SVM models, training data and 
% the kernel combination coefficients.

numTasks = size(trainData, 2);
numKernels = size(trainData(1).kernelMatrix, 3);

value = zeros(numTasks, 1);

for t = 1:numTasks
    kernel = zeros(size(trainData(t).kernelMatrix(:, :, 1)));
    for m = 1:numKernels
        kernel = kernel + theta(m) .* trainData(t).kernelMatrix(:, :, m);
    end

    e = ones(size(trainData(t).kernelMatrix(:, :, 1), 1), 1);
    alphaall = zeros(size(e));
    alphaall(model(t).SVs) = model(t).sv_coef;

    value(t) = e' * abs(alphaall) - 0.5 * alphaall' * kernel * alphaall;
end

end