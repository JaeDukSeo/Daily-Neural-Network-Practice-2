Function [cost, grad] = sparseCodingWeightCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
 %sparseCodingWeightCost - given the features in                          featureMatrix , 
 % computes the cost and gradient with respect to
 % the weights, given In weightMatrix
 % parameters
 % weightMatrix - the weight matrix. weightMatrix(:, c) is the cth basis
 %                    vector.
 % featureMatrix - the feature matrix. featureMatrix(:, c) is the features
 %                    forThe cth example
 % visibleSize - number of pixels in the patches
 % numFeatures - number of features
 % patches - patches
 % gamma - weight decay parameter (on weightMatrix)
 % lambda - L1 sparsity weight (on featureMatrix)
 % epsilon - L1 sparsity epsilon
 % groupMatrix - grouping matrix. groupMatrix(r, :) indicates the
 % features included in the rth group. groupMatrix(r, c)
 %                    is  1  if the cth feature is  in the rth group and0 
%                    otherwise.

    if exist( ' groupMatrix ' , ' var ' )
        Assert(size(groupMatrix, 2 ) == numFeatures, ' groupMatrix has bad dimension ' );
    else 
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2 );

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
     %    Write code to compute the cost and gradient with respect to the
     %weights given in weightMatrix.     
     % -------------------- YOUR CODE HERE ---- ----------------    
 %% finds the cost function of the target
    Delta = weightMatrix*featureMatrix- patches;
    fResidue = sum(sum(delta.^ 2 ))./numExamples;% reconstruction error
    fWeight = gamma*sum(weightMatrix(:).^ 2 );% prevents the value of the base element from being too large
    sparsityMatrix = sqrt(groupMatrix*(featureMatrix.^ 2 )+ epsilon);
    fSparsity = lambda*sum(sparsityMatrix(:));% penalty value for feature coefficient
    cost = fResidue + fWeight  + fSparsity;% cost function target
 % + cost = fResidue fWeight;
    
    %% finds the partial derivative function of the target cost function
    Grad = ( 2 * weightMatrix * featureMatrix  * (featureMatrix') * - 2 * patches *featureMatrix'  ) ./ numExamples + 2 * Gamma * weightMatrix;
    grad = Grad(:);
    
    
    
    
    
    
End