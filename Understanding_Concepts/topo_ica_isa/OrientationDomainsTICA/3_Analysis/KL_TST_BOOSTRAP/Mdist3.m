function [D ]=Mdist3(S1,xmin ,xmax,N,tipe)
% Mesure of distance used, options:
%   'KL':            Kullback–Leibler divergence
%   'bhattacharyya': Bhattacharyya distance
%   'empty_cells':   Number of empty cells          
 

NS1=length(S1);
bins=cell(1);
bins{1}=linspace(xmin,xmax,N(1));
bins{2}=linspace(xmin,xmax,N(2));
[Ps1,bn]=hist3(S1,'Edges' ,bins);
%[Ps1,bn]=hist3(S1,bins);

P=Ps1/NS1;
Q=1/(N(1)*N(2)) *ones(N(1),N(2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(tipe,'KL')    
    %pcolor(linspace(xmin,xmax,N),linspace(xmin,xmax,N),Ps1') 
    temp =  P(:).*log(P(:)./Q(:));
    D= nansum(temp);    
elseif strcmp(tipe,'empty_cells') 
    % teselacio en cuadrats del pla, proporcio de buits    
    D=sum(P(:)==0)/(N(1)*N(2));
elseif strcmp(tipe,'bhattacharyya')
    temp =  sqrt(P(:).*Q(:));
    D= 1-nansum(temp); 
else
    D=Nan;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%
bins{2}=linspace(xmin,xmax,N(1));
bins{1}=linspace(xmin,xmax,N(2));
[Ps1,bn]=hist3(S1,'Edges' ,bins);
%[Ps1,bn]=hist3(S1,bins);

P=Ps1/NS1;
Q=1/(N(1)*N(2)) *ones(N(2),N(1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(tipe,'KL')    
    %pcolor(linspace(xmin,xmax,N),linspace(xmin,xmax,N),Ps1') 
    temp =  P(:).*log(P(:)./Q(:));
    D=D+ nansum(temp);    
elseif strcmp(tipe,'empty_cells') 
    % teselacio en cuadrats del pla, proporcio de buits    
    D=D+ sum(P(:)==0)/((N(1)*N(2)));
elseif strcmp(tipe,'bhattacharyya')
    temp =  sqrt(P(:).*Q(:));
    D=D+  1-nansum(temp); 
else
    D=D+ Nan;

end
D=D/2 ;