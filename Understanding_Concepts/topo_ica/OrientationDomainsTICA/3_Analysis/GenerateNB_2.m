%-----------------------------------------------------------------
% GenerateNB - generates the neighborhood matrix for TICA
%-----------------------------------------------------------------
function [NBNZ,NB] = GenerateNB_2( p )

% This will hold the neighborhood function entries
NB = zeros(p.xdim*p.ydim*[1 1]);

% This is currently the only implemented neighborhood
% if strcmp(p.neighborhood,'ones3by3')==0
%   error('No such neighborhood allowed!');
% end
N=p.neighborhoodN;
% Step through nodes one at a time to build the matrix
ind = 0;
for y=1:p.ydim
  for x=1:p.xdim
    
    ind = ind+1;

    % Rectangular neighbors
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
