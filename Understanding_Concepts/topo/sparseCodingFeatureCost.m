Function [cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
 %sparseCodingFeatureCost - given the weights in weightMatrix,
 %                           computes the cost and gradient with respect to
 % the features, given In featureMatrix
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
     % features given in featureMatrix.     
     % You may wish to write the non- topographic version, ignoring
     %    the grouping matrix groupMatrix first, and extend the 
     % non- topographic version To the topographic version later.
     % -------------------- YOUR CODE HERE --------------------
    
      %% finds the cost function of the target
    Delta = weightMatrix*featureMatrix- patches;
    fResidue = sum(sum(delta.^ 2 ))./numExamples;% reconstruction error
    fWeight = gamma*sum(weightMatrix(:).^ 2 );% prevents the value of the base element from being too large
    sparsityMatrix = sqrt(groupMatrix*(featureMatrix.^ 2 )+ epsilon);
    fSparsity = lambda*sum(sparsityMatrix(:)); % penalty value for feature coefficient
    cost = fResidue + fSparsity + fWeight;% case A is a constant, can not
 % cost = fResidue ++ fSparsity;

    % finds the partial derivative function of the target cost function
    gradResidue = ( 2*featureMatrix * weightMatrix * weightMatrix' - 2 *weightMatrix * patches )./ numExamples;
  
    % non-topology:
    isTopo = 1 ; %isTopo = 1, expressed as topology
     if ~ isTopo
        gradSparsity = lambda*( featureMatrix./ sparsityMatrix);
        
    % Topology when
     else 
        gradSparsity = lambda*groupMatrix *(groupMatrix*(featureMatrix.^2)+epsilon).^(-0.5).*featureMatrix;
        %Be careful that the last item is the inner product multiplication 
    end
    grad = gradResidue + gradSparsity;
    grad = grad(:);
    
    
    
    
    
End