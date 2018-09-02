%% CS294A/CS294W Independent Component Analysis (ICA) Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  ICA exercise. In this exercise, you will need to modify
%  orthonormalICACost.m and a small part of this file, ICAExercise.m.

%%======================================================================
clear all; clc;
temp = load('olivettifaces.mat');
temp_face = temp.faces;
temp_reshape = reshape(temp_face,[64,64,400]);
display_network(temp_face(:, 1:100),64);
%%
temp_small = imresize(temp_reshape,0.5);
temp_small_flat = reshape(temp_small,[32*32,400]);
display_network(temp_small_flat(:, 1:100),32);
%% STEP 0: Initialization
%  Here we initialize some parameters used for the exercise.

numPatches = 20000;
numFeatures = 16*16;
imageChannels = 3;
patchDim = 8;
visibleSize = 32*32;

outputDir = '.';
epsilon = 1e-6; % L1-regularisation epsilon |Wx| ~ sqrt((Wx).^2 + epsilon)
%%======================================================================
%% STEP 1: Sample patches
patches = temp_small_flat;
display_network(patches(:, 1:100),32);
%%======================================================================
%% STEP 2: ZCA whiten patches
%  In this step, we ZCA whiten the sampled patches. This is necessary for
%  orthonormal ICA to work.

patches = patches / 255;
meanPatch = mean(patches, 2);
patches = bsxfun(@minus, patches, meanPatch);

sigma = patches * patches';
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s))) * u';
patches_white = ZCAWhite * patches;
%%
display_network(patches_white(:, 1:100),32);
%%======================================================================
%% STEP 4: Optimization for orthonormal ICA
%  Optimize for the orthonormal ICA objective, enforcing the orthonormality
%  constraint. Code has been provided to do the gradient descent with a
%  backtracking line search using the orthonormalICACost function 
%  (for more information about backtracking line search, you can read the 
%  appendix of the exercise).
%
%  However, you will need to write code to enforce the orthonormality 
%  constraint by projecting weightMatrix back into the space of matrices 
%  satisfying WW^T  = I.
%
%  Once you are done, you can run the code. 10000 iterations of gradient
%  descent will take around 2 hours, and only a few bases will be
%  completely learned within 10000 iterations. This highlights one of the
%  weaknesses of orthonormal ICA - it is difficult to optimize for the
%  objective function while enforcing the orthonormality constraint - 
%  convergence using gradient descent and projection is very slow.

weightMatrix = rand(numFeatures, visibleSize);

[cost, grad] = orthonormalICACost(weightMatrix(:), visibleSize, numFeatures, patches_white, epsilon);

fprintf('%11s%16s%10s\n','Iteration','Cost','t');

startTime = tic();

% Initialize some parameters for the backtracking line search
alpha = 0.5;
t = 1;
lastCost = 1e40;
display_network(weightMatrix(:,1:100),32);
%%
display_network(weightMatrix(1:100,:)',16);
%%
% Do 10000 iterations of gradient descent
for iteration = 1:1000
                       
    grad = reshape(grad, size(weightMatrix));
    grad = grad/norm(grad,'fro'); %%% 
    newCost = Inf;        
    linearDelta = sum(sum(grad .* grad));
    
    % Perform the backtracking line search
    while 1
        considerWeightMatrix = weightMatrix - t * grad;
        % -------------------- YOUR CODE HERE --------------------
        % Instructions:
        %   Write code to project considerWeightMatrix back into the space
        %   of matrices satisfying WW^T = I.
        %   
        %   Once that is done, verify that your projection is correct by 
        %   using the checking code below. After you have verified your
        %   code, comment out the checking code before running the
        %   optimization.
        
        % Project considerWeightMatrix such that it satisfies WW^T = I
        [U S V] = svd(considerWeightMatrix*considerWeightMatrix');
        S = 1./sqrt(diag(S));
        considerWeightMatrix = U*diag(S)*U'*considerWeightMatrix;
        
%         error('Fill in the code for the projection here');        
        
        % Verify that the projection is correct
        temp = considerWeightMatrix * considerWeightMatrix';
        temp = temp - eye(numFeatures);
        assert(sum(temp(:).^2) < 1e-23, 'considerWeightMatrix does not satisfy WW^T = I. Check your projection again');
%         error('Projection seems okay. Comment out verification code before running optimization.');
        
        % -------------------- YOUR CODE HERE --------------------                                        

        [newCost, newGrad] = orthonormalICACost(considerWeightMatrix(:), visibleSize, numFeatures, patches_white, epsilon);
%         disp(['diff = ' num2str(newCost  - (lastCost - alpha * t * linearDelta))]);
        if newCost > lastCost - alpha * t * linearDelta
            t = 0.9 * t;
        else
%             disp('break');
            break;
        end
    end
   
    if abs(lastCost - newCost) <0.000001
        break;
    end
    lastCost = newCost;
    weightMatrix = considerWeightMatrix;
    
    fprintf('  %9d  %14.6f  %8.7g\n', iteration, newCost, t);
    
    t = 1.1 * t;
    
    cost = newCost;
    grad = newGrad;
           
    figure(1);
    display_network(weightMatrix(1:100,:)',16);           
end
