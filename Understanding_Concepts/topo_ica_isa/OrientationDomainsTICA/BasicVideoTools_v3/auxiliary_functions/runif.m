function r=runif(m,d,N,M)

%'RUNIFL' genera una matriz aleatoria de tamaño n*m cuyos elementos
% siguen una distribucion uniforme de media mm y desviacion dd
%
% USO: r=runif(mm,dd,n,m);
%

mi=m-d*sqrt(12)/2;
ma=d*sqrt(12)+mi;
dis=ma-mi;

r=(rand(N,M)-0.5)*dis+m;