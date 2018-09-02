function [cost, grad] = RICACost(theta, visibleSize, numFeatures, patches, epsilon)
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
    m = size(patches,2);
    
    
    %========= RICA:
%     COST FUNCTION:
    aux1 = sqrt(((weightMatrix*patches).^2) + epsilon);
    aux2 =  (weightMatrix'*weightMatrix*patches - patches).^2;
    cost = sum(aux2(:))/m + sum(aux1(:))/m;    
    
%     GRADIENT:
    aux3 = weightMatrix'*weightMatrix*patches - patches;
    part1 = ((weightMatrix*aux3*(patches'))+ ((weightMatrix*patches)*(aux3')));
    part2 = ((weightMatrix*patches)./aux1)*patches';
%     grad = ((1./aux1).*(weightMatrix*patches))*patches';
    grad = ((2/m).*part1) + (part2./m);
%     grad = (((weightMatrix*patches)./aux1)*patches');
    grad = grad(:);
%     
end
