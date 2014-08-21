function kernelMatrix = genKernelMatrices(data1, data2)
% A function that calculates multiple kernel matrices based on data1 and
% data2.
%
% Input:
%           data1 - A n-by-i sized matrix, where n is the number of 
%                   samples, and i is the number of features of each 
%                   sample.
%           data2 - A m-by-j sized matrix, where m is the number of 
%                   samples, and j is the number of features of each 
%                   sample.
%
% Output:
%    kernelMatrix - A n-by-m-by-k sized multidimensional array, where k is
%                   the number of kernels. By default, 9 Gaussian kernels,
%                   1 polynomial kernel and 1 linear kernel are used, sums 
%                   to 11 kernels.

sigmaForGaussian = [-7 -5 -3 -1 0 1 3 5 7];
degreeForPoly = [2];

for sigmaIndex = 1:size(sigmaForGaussian, 2)
    kernelMatrix(:, :, sigmaIndex) = calKernelMatrix('Gaussian', data1, data2, 's', 2^(sigmaForGaussian(sigmaIndex)));
end

for degreeIndex = 1:size(degreeForPoly, 2)
    kernelMatrix(:, :, sigmaIndex + degreeIndex) = calKernelMatrix('poly', data1, data2, 'd', degreeForPoly(degreeIndex));
end

kernelMatrix(:, :, sigmaIndex + degreeIndex + 1) = calKernelMatrix('linear', data1, data2);

end

