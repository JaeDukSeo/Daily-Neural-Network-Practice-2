function [cost, grad] = orthonormalICACost(theta, visibleSize, numFeatures, patches, epsilon)
%orthonormalICACost - compute the cost and gradients for orthonormal ICA
%                     (i.e. compute the cost ||Wx||_1 and its gradient)

    weightMatrix = reshape(theta, numFeatures, visibleSize);
    
    cost = 0;
    grad = zeros(numFeatures, visibleSize);
    
    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   weights given in weightMatrix.     
    % -------------------- YOUR CODE HERE --------------------   
    
%     cost = sum(sum(abs(W*patches)));

    Y = weightMatrix*patches;% d*n
    cost = sum(sum(sqrt(Y.^2 + epsilon)));
    
    
%     N = size(patches,2);
%     for j = 1:numFeatures
%         for i = 1:N
%             grad(j,:) = grad(j,:) + weightMatrix(j,:)*patches(:,i)*patches(:,i)'/sqrt((weightMatrix(j,:)*patches(:,i))^2+epsilon);
%         end
%     end
% tic
    grad = (Y./sqrt(Y.^2+epsilon))*patches';
%     toc
%     grad = weightMatrix*patches* diag(1./sqrt(sum(Y.*Y)+epsilon))*patches';
%     grad = (1./sqrt(weightMatrix*patches.^2 + epsilon)) .* weightMatrix*patches*patches';
    grad = grad(:);
end