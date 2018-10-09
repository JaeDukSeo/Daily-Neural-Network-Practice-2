function [iaf,csfrg]=iafrg(f,C,facfrec,nolin)

% IAFRG calcula los valores de la funcion de asignacion de informacion
% para el canal RG (experim) en el dominio discreto de frecuencias y
% contrastes definido por los vectores fila (f,C)
% -de longitudes m y n respectvamente-
%
% IAFRG da una matriz m*n tal que en cada fila, contiene los valores
% de la funcion para los diferentes contrastes (con f fija)
% (para que la csf que da sea la correcta, el primer contraste debe ser proximo a 0)
%
%
% [iafrg,csfrg]=iafrg(f,C,facfrec,[aplic_no_lineal?(0/1) ciaf_no_lin? aplic_log?]);
%


f=facfrec*f;

f=f+0.00001*(f==0);
C=C+0.0000001*(C==0);

lf=length(f);
lc=length(C);

iaf=zeros(lf,lc);
p=[0.0840 0.8345 0.6313 0.2077];

if length(nolin)==1
   nolin=[nolin nolin];
end

nolini=nolin;
nolin=nolini(1);

if ((nolini(1)==0)&(nolini(2)==1))
   nolin=1;      
end

if nolin==1
   for i=1:lf
       cu=1/(100*2537.9*sigm1d(f(i),-55.94,6.64));
       ace(i,:)=umbinc3(C,cu,p(1),p(2),p(3),p(4));
   end
   iaf=1./ace;
else
   iaf=100*2537.9*sigm1d(f,-55.94,6.64);
   iaf=iaf'*ones(1,length(C));
end
csfrg=iaf(:,1)';

if ((nolini(1)==0)&(nolini(2)==1))
   s=size(iaf);
   iafc=sum(iaf')';
   iaf=iafc*ones(1,s(2));
end

