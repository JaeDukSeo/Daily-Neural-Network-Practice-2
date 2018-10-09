function ptf=proytf3(tf,tipo)

% PROYTF3 proyecta la transformada de Fourier 3D dada por fft3
% (organizada como una secuencia de matrices f2,f2 a lo largo de
% la dimension f3) sobre los planos (f1,f2), (f1,f3) o (f2,f3).
% Para seleccionar el tipo de proyeccion 't' debe valer:
%
%   t=1  : proyeccion sobre (f1,f2) sumando para todo f3
%
%   t=2  : proyeccion sobre (f1,f3) sumando para todo f2
%
%   t=3  : proyeccion sobre (f2,f3) sumando para todo f1
%
% USO: ptf=proytf3(tf,t);
%
% NOTA! se supone que cada frame es una matriz cuadrada
%

m=size(tf);
nf=m(2)/m(1);

if tipo==1
    ptf=zeros(m(1),m(1));
    for i=1:nf          
        f=sacafot(tf,m(1),m(1),i); 
        ptf=ptf+f;
    end
elseif tipo==2
    ptf=zeros(m(1),nf);
    for i=1:nf
        f=sacafot(tf,m(1),m(1),i); 
        ptf(:,i)=(sum(f))';
    end    
else
    ptf=zeros(m(1),nf);
    for i=1:nf
        f=sacafot(tf,m(1),m(1),i); 
        ptf(:,i)=(sum(f'))';
    end
end