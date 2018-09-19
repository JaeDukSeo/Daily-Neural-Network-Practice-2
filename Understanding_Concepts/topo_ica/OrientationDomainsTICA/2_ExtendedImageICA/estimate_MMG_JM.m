function estimate_MMG( whiteningMatrix, dewhiteningMatrix, fname, p )
% estimate - algorithms for topographic, subspace, and standard ICA
%
% SYNTAX:
% estimate( whiteningMatrix, dewhiteningMatrix, fname, dims );
%
% NOTE: X passed as global!
%
% X                  preprocessed (whitened) sample vectors in columns
% whiteningMatrix    transformation from observation space to X-space
% dewhiteningMatrix  inverse transformation
% fname              name of file to write
%
% PARAMETERS COMMON TO ALL ALGORITHMS
%
% p.seed             random number generator seed
% p.write            iteration interval for writing results to disk
%
% ICA (with FastICA algorithm, tanh nonlinearity)
%
% p.model            'ica'
% p.algorithm        'fixed-point'
% p.components       number of ICA components to estimate
% 
% ISA (gradient descent with adaptive stepsize)
%
% p.model            'isa'
% p.algorithm        'gradient'
% p.groupsize        dimensionality of subspaces
% p.groups           number of independent subspaces to estimate
% p.stepsize         starting stepsize
% p.epsi             small positive constant
%
% TOPOGRAPHIC ICA (gradient descent with adaptive stepsize)
%
% p.model            'tica'
% p.algorithm        'gradient'
% p.xdim             columns in map
% p.ydim             rows in map
% p.maptype          'standard' or 'torus'
% p.neighborhood     'ones3by3' (only one implemented so far)
% p.stepsize         starting stepsize
% p.epsi             small positive constant
% p.neighborhoodN    MMG param
% p.nonLinarity      MMG param
% p.startPoint       MMG param
% p.star             MMG param
% p.Vstepsize        MMG param
% COMMENTS-----------------------------------------------------------------
% MMG & JM included more options:
%  - Size of the neighborhood is a variable
%  - Nonlinearity used can be: 
%  - Starting point, could be:
%       * Randon
%       * NonRandon,  if this is the case, the Starting point must be an
%         input, given by p.start
%-------------------------------------------------------------------
% PRELIMINARIES
%-------------------------------------------------------------------

% Print options
fprintf('You have selected the following options...\n');
p
% If we have the TICA model, generate the neighborhood
if strcmp(p.model,'tica')

  % Neighborhood matrix: NB(i,j) = strength of unit j in neighb. of 
  % unit i. In addition, we will create a matrix NBNZ, which gives 
  % the positions of the non-zero entries in NB, to lower the 
  % computational expenses.
  fprintf('Generating neighborhood matrix...\n');
  [NBNZ,NB] = GenerateNB_MMG( p );
  
end

% Take the data from the global variable
global X;
N = size(X,2);

%-------------------------------------------------------------------
% SETTING UP THE STARTING POINT...
%-------------------------------------------------------------------

% Initialize the random number generator.
rand('seed',p.seed);

% Take random initial vectors...
if strcmp(p.model,'ica')
  B = randn(size(X,1),p.components);      
elseif strcmp(p.model,'isa')
  B = randn(size(X,1),p.groupsize*p.groups);    
elseif strcmp(p.model,'tica')
    % MMG modification 
      if strcmp(p.startPoint,'Rand') 
          B = randn(size(X,1),p.xdim*p.ydim); 
      else
          B = p.start;
      end
end

% ...and decorrelate (=orthogonalize in whitened space)
B = B*real((B'*B)^(-0.5));
n = size(B,2);

%-------------------------------------------------------------------
% START THE ITERATION...
%-------------------------------------------------------------------

% Print the time when started (and save along with parameters).
c=clock;
if c(5)<10, timestarted = ['Started at: ' int2str(c(4)) ':0' int2str(c(5))];
else timestarted = ['Started at: ' int2str(c(4)) ':' int2str(c(5))];
end
fprintf([timestarted '\n']);
p.timestarted = timestarted;

% Initialize iteration counter
iter=0;

% Use starting stepsize for gradient methods
if strcmp(p.algorithm,'gradient')
  stepsize = p.stepsize; 
  obj = [];
  objiter = [];
end

% Loop forever, writing result periodically
while 1  
  
  % Increment iteration counter
  iter = iter+1;  
  fprintf('(%d)',iter);
  
  %-------------------------------------------------------------
  % FastICA step MMG deleted on purpose
  %-------------------------------------------------------------  
    
  %-------------------------------------------------------------
  % ISA gradient MMG deleted on purpose
  %-------------------------------------------------------------     
  
  %-------------------------------------------------------------
  % TICA gradient
  %-------------------------------------------------------------  
  
  if strcmp(p.model,'tica') & strcmp(p.algorithm,'gradient')

    % Calculate linear filter responses and their squares
    n2=randperm(size(X,2));% MMG modification
    n22=floor(1*size(X,2));  % If you choose this factor 1 less than 1 you take different minisets in each iteration (faster)
    U = B'*X(:,n2(1:n22)); Usq = U.^2;
    
    % Calculate local energies
    E= NB*Usq; % MMG modification: the diff are due to numerical issues (it is possible due to flat neighborhood)
%     tic
%     for ind=1:n
%       E(ind,:) = NB(ind,NBNZ{ind}) * Usq(NBNZ{ind},:);
%     end   
   
    
    % MMG modification: Nonlinearity
   %  Original
   if strcmp(p.nonLinarity, 'ori')
       g = -((p.epsi + E).^(-0.5));
   elseif strcmp(p.nonLinarity, 'expo1') 
       e = 0.2;
       K = (p.epsi^e)/(p.epsi^0.5);
       g = -K*((p.epsi + E).^(-e));
   elseif strcmp(p.nonLinarity, 'expo2') 
       e = 0.4;
       K = (p.epsi^e)/(p.epsi^0.5);
       g = -K*((p.epsi + E).^(-e));
   elseif strcmp(p.nonLinarity, 'expo3') 
       e = 0.6;
       K = (p.epsi^e)/(p.epsi^0.5);
       g = -K*((p.epsi + E).^(-e));
     
  elseif  strcmp(p.nonLinarity, 'atan')  
      K =15;
      a =2;
      g = K*2*(atan(a*E)-pi/2)/pi;    
   end
    % Calculate convolution with neighborhood
%     for ind=1:n
%       F(ind,:) = NB(ind,NBNZ{ind}) * g(NBNZ{ind},:);
%     end
    % MMG modification:  the diff are due to numerical issues
    F=NB*g;
    
    % This is the total gradient
    dB = X(:,n2(1:n22))*(U.*F)'/N;
    
  end

  %-------------------------------------------------------------
  % ADAPT STEPSIZE FOR GRADIENT ALGORITHMS
  %-------------------------------------------------------------  

  if strcmp(p.algorithm,'gradient')

    % Perform this adaptation only every 5 steps
    if rem(iter,10)==0 | iter==1
      tic
      % Take different length steps
      Bc{1} = B + 0.5*stepsize*dB;
      Bc{2} = B + 1.0*stepsize*dB;
      Bc{3} = B + 2.0*stepsize*dB;
       
      % Calculate objective values in each case
      for i=1:3
          % Orthogonalize each one
          Bc{i} = Bc{i}*real((Bc{i}'*Bc{i})^(-0.5)); 
          % ISA objective MMG deleted on purpose
          % TICA objective          
          Usq = (Bc{i}'*X(:,n2(1:n22))).^2;
          % for ind=1:n
          %    E(ind,:)= NB(ind,NBNZ{ind}) * Usq(NBNZ{ind},:);
          % end
          E= NB*Usq;
          objective(i) = mean(mean(sqrt(p.epsi+E)));
          
      end
      
      % Compare objective values, pick step giving minimum
      if objective(1)<objective(2) & objective(1)<objective(3)
          % Take shorter step
          stepsize = stepsize/2;
          fprintf('Stepsize now: %.4f\n',stepsize);
          obj = [obj objective(1)];
      elseif objective(1)>objective(3) & objective(2)>objective(3)
          % Take longer step
          stepsize = stepsize*2;
          fprintf('Stepsize now: %.4f\n',stepsize);
          obj = [obj objective(3)];
      else
          % Don't change step
          obj = [obj objective(2)];
      end
      
      objiter = [objiter iter];
      fprintf('\nObjective value: %.6f\n',obj(end));
      p.Vstepsize=[p.Vstepsize stepsize];
      
    end
toc
    B = B + stepsize*dB;
    
  end
  
  
  %-------------------------------------------------------------
  % Ortogonalize (equal to decorrelation since we are 
  % in whitened space)
  %-------------------------------------------------------------
  
  B = B*real((B'*B)^(-0.5));
  
  %-------------------------------------------------------------
  % Write the results to disk
  %-------------------------------------------------------------
  
  if rem(iter,p.write)==0 | iter==1
    
    A = dewhiteningMatrix * B;
    W = B' * whiteningMatrix;
      
    fprintf(['Writing file: ' fname '...']);
    if strcmp(p.algorithm,'gradient')
      eval(['save ' fname ' W A p iter obj objiter']);
    else
      eval(['save ' fname ' W A p iter']);
    end
    fprintf(' Done!\n');      
    
  end
  
end

% We never get here...
return;




%-----------------------------------------------------------------
% GenerateNB - generates the neighborhood matrix for TICA
%-----------------------------------------------------------------
% MMG version of the GenerateNB
function [NBNZ,NB] = GenerateNB_MMG_JM( p )

% This will hold the neighborhood function entries
NB = zeros(p.xdim*p.ydim*[1 1]);

% This is currently the only implemented neighborhood
% if strcmp(p.neighborhood,'ones3by3')==0
%   error('No such neighborhood allowed!');
% end
N=p.neighborhoodN; % MMG 
% Step through nodes one at a time to build the matrix
ind = 0;
for y=1:p.ydim
  for x=1:p.xdim
    
    ind = ind+1;

    % Rectangular neighbors  % MMG version
    [xn,yn] = meshgrid( (x-N):(x+N), (y-N):(y+N) );
    xn = reshape(xn,[1 (2*N+1)^2]);
    yn = reshape(yn,[1 (2*N+1)^2]);
      
    if strcmp(p.maptype,'torus')
      
      % Cycle round      
      i = find(yn<1); yn(i)=yn(i)+p.ydim;
      i = find(yn>p.ydim); yn(i)=yn(i)-p.ydim;
      i = find(xn<1); xn(i)=xn(i)+p.xdim;
      i = find(xn>p.xdim); xn(i)=xn(i)-p.xdim;
      
    elseif strcmp(p.maptype,'standard')
      
      % Take only valid nodes
      i = find(yn>=1 & yn<=p.ydim & xn>=1 & xn<=p.xdim);
      xn = xn(i);
      yn = yn(i);
      
    else
      error('No such map type!');
    end
    
    % Set neighborhood
    NB( ind, (yn-1)*p.xdim + xn )=1;
    
  end
end

% For each unit, calculate the non-zero columns!
for i=1:p.xdim*p.ydim
  NBNZ{i} = find(NB(i,:));
end

return;
