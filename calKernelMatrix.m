function kernelMatrix = calKernelMatrix(ker, data1, data2, varargin)
% This function is used to compute the Kernel matrix based on the training
% samples. The resulting kernel matrix is normalized.
%
% Input:
%      ker - string, name of the Kernels, this function supports three
%            types of Kernels: 'linear', 'poly', 'Gaussian'.
%    data1 - A n-by-i sized matrix, where n is the number of samples and i
%            is the number of features of each sample.
%    data2 - A m-by-j sized matrix, where m is the number of samples and j
%            is the number of features of each sample.
% Parameters:
%        b - Bias of the polynomial Kernel, default = 0.
%        d - Degree of the polynomial Kernel, default = 2.
%        s - Sigma of the Gaussian Kernel, default = 1.
%
% Output:
%        K - A n-by-m sized kernel matrix.


% Set default parameters.
pars.b = 0;
pars.d = 2;
pars.s = 1;

% Pass the parameters in the function if they exist.
for i =1:2:length(varargin)
    if i < length(varargin)
        eval(['pars.', varargin{i}, '=', num2str(varargin{i+1}), ';']);
    end
end

% Compute the kernel matrix.
switch ker
    case 'linear'
        kernelMatrix = data1 * data2';
        
        % normalize the kernel matrix
        temp1 = sqrt(diag(data1 * data1'));
        norm1 = repmat(temp1, 1, size(data2, 1));
        
        temp2 = sqrt(diag(data2 * data2'));
        norm2 = repmat(temp2', size(data1, 1), 1);
        
        kernelMatrix = kernelMatrix ./ (norm1 .* norm2);
        clear norm1 norm2 temp1 temp2;
    case 'poly'
        kernelMatrix = (data1 * data2' + pars.b).^pars.d;
        
        % normalize the kernel matrix
        temp1 = sqrt(diag((data1 * data1' + pars.b).^pars.d));
        norm1 = repmat(temp1, 1, size(data2, 1));

        temp2 = sqrt(diag((data2 * data2' + pars.b).^pars.d));
        norm2 = repmat(temp2', size(data1, 1), 1);

        kernelMatrix = kernelMatrix ./ (norm1 .* norm2);
        clear norm1 norm2 temp1 temp2;
    case 'Gaussian'
        exponent = bsxfun(@plus, sum(data1 .* data1, 2), (-2) * data1 * data2');
        exponent = bsxfun(@plus, sum(data2 .* data2, 2)', exponent);
        kernelMatrix = exp(-exponent / pars.s);
        clear exponent;
    otherwise
        error(['Sorry, we do not support this kernel: ', ker]);
end