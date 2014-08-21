function [svmModel, theta] = train(trainSample, C, rLambda, a, p)
% A function that trains a multi-task multiple kernel learning model based
% on the paper
%   
%    C. Li, M. Georgiopoulos and G. C. Anagnostopuolos, "Conic 
%    Multi-Task Classification", ECMLPKDD 2014.
%
% Input:
%    trainSample - A 1-by-T sized struct, where T is the number of tasks. 
%                  In the t-th struct, it should contain the training data  
%                  and labels of the t-th task, i.e., trainSample(t) 
%                  is comprised with trainSample(t).data and 
%                  trainSample(t).label, for all t = 1, ..., T.
%                  
%                  trainSample(t).data should be an Nt-by-D sized matrix, 
%                  where Nt is the number of training samples of the t-th 
%                  task, and D is the number of features for each training
%                  sample. 
%                  
%                  trainSample(t).label should be an Nt-by-1 sized vector, 
%                  with each element to be either 1 or -1, representing the
%                  label of the corresponding training sample of the t-th 
%                  task.
%
%    For the meaning of the following four inputs, please refer to the
%    paper.
%              C - The SVM parameter.
%        rLambda - The upper bound of lambda.
%              a - The upper bound of the sum of 1/lambda_t
%              p - Specifies the L_p-norm constraint of theta
%
% Output:
%       svmModel - A 1-by-T sized struct, where the t-th struct contains
%                  the output of LIBSVM, trained on the data of the t-th
%                  task.
%          theta - The kernel combination coefficients.

numTasks = size(trainSample, 2);

for t = 1:numTasks
    trainData(t).kernelMatrix = genKernelMatrices(trainSample(t).data, trainSample(t).data);
    trainData(t).label = trainSample(t).label;
end

numKernels = size(trainData(1).kernelMatrix, 3);

% Initialize lambda and theta.
lambda = rand(numTasks, 1);
theta = rand(numKernels, 1);

% Start training via block coordinate descent.
iter = 0;
change = 1;
while change > 0.001
    iter = iter+1;
    
    thetaPrevious = theta;
    lambdaPrevious = lambda;
    
    svmModel = trainSVMs(trainData, theta, C);
    
    wNorm = calculateWNorm(trainData, svmModel, theta, lambda);
    theta = updateTheta(wNorm, p);
    
    functionValue = calFunctionValue(svmModel, trainData, theta);
    lambda = updateLambda(functionValue, rLambda, a);
    
    change = (sum(abs(lambda - lambdaPrevious)) + sum(abs(theta - thetaPrevious))) / (numTasks + numKernels);
    disp(['iter = ', num2str(iter), ' change of parameters = ', num2str(change)]);
end

end

%% Update lambda by using CVX.
function lambda = updateLambda(functionValue, rLambda, a)

numTasks = size(functionValue, 1);

cvx_begin
    variable lambdaVar(numTasks)
    minimize(lambdaVar' * functionValue)
    subject to
        lambdaVar <= rLambda
        lambdaVar >= 1
        sum(pow_p(lambdaVar, -1)) <= a
cvx_end

lambda = lambdaVar;

end


%% Update theta via closed-form solution.
function theta = updateTheta(G, p)

g = sum(G, 1)';
theta = (g ./ norm(g, p/(p+1))) .^ (1/(p+1));

end


%% Calculate the norms of w_t^m
function wNorm = calculateWNorm(trainData, model, theta, lambda)

numTasks = size(trainData, 2);
numKernels = size(trainData(1).kernelMatrix, 3);

wNorm = zeros(numTasks, numKernels);

for t = 1:numTasks
    e = ones(size(trainData(t).kernelMatrix(:, :, 1), 1), 1);
    alphaall = zeros(size(e));
    alphaall(model(t).SVs) = model(t).sv_coef; 

    for m = 1:numKernels
        wNorm(t, m) =  max([alphaall' * (trainData(t).kernelMatrix(:, :, m)) * alphaall, 0]);
        wNorm(t, m) = wNorm(t, m) * lambda(t) * (theta(m)^2);
    end
end

end