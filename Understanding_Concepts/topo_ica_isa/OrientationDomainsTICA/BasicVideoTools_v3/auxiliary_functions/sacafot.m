function fotogramas=sacafot(sec,fil,col,N)

% SACAFOT extrae los fotogramas de las posiciones dadas por el vector v
% de la secuencia SEC con fotogramas de tamaño m*n.
% Evidentemente, si v es un numero, saca un unico fotograma (el de la posi-
% cion v).
%
% USO: fotog=sacafot(SEC,m,n,v);

s=size(sec);
fotogramas=zeros(fil,length(N)*col);
for fot=1:length(N)
    nfot=N(fot);
    fotogramas(:,col*(fot-1)+1:col*fot)=sec(:,col*(nfot-1)+1:col*nfot);
end
%fot=sec(imslice([fil col s(2)/col],N));



