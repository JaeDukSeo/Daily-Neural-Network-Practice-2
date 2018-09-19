function sec=metefot(sec,foto,N,ma)

% METEFOT inserta en la secuencia SEC el fotograma FOT en las posicion N. Es posible elegir
% si se inserta o machaca el fotograma que habia en esa posicion.
% Evidentemente, el fotograma debe ser del mismo tamaño que los del resto de la secuencia.
%
% Si se quiere insertar un fotograma en una posicion mayor que el numero de fotogramas de 
% la secuencia simplemente se añade al final (no se intercalan imagenes en negro). 
%
% USO: nueva_sec=metefot(SEC,FOT,N,machaca?(1/0));
%


ss=size(foto);
fil=ss(1);
col=ss(2);
s=size(sec);
Nfot=s(2)/col;
if N>Nfot
   sec=[sec foto];
else 
   if ma==1
       %sec(imslice([fil col Nfot],N))=foto;
       sec(:,(N-1)*col+1:N*col)=foto;
   else
       if N==1
           sec=[foto sec];     
       else
           seca=sec(:,1:(N-1)*col);
           secd=sec(:,(N-1)*col+1:s(2));
           sec=[seca foto secd]; 
       end 
   end
end