function [res]=filtra(fun,filt,p)

%'FILTRA' calcula la convolucion de las señales dadas mediante 
%el producto de sus espectros, eliminando uno de los factores de-
%pendientes del tamaño introducidos por el algoritmo de TF. De 
%otra forma, al calcular la TF-1 solo se eliminaria uno de estos 
%factores de forma que el filtrado resultaria de pendiente del 
%tamaño. 
%
%USO: [result]=filtra(señal,repuesta impuls del filtro,parte);
%
%              Seleccion de la parte del resultado:
%                      
%                      0.......Todo
%                      1.......Parte real
%



s=size(fun);
a=s(1)*s(2);

if (s(1)==1)|(s(2)==1),
    res=ifft(fft(fun).*fft(filt)/a);
    if p==1,
       res=real(res);
    end   
    if p==0,
       res=res;
    end 
else
    res=ifft2(fft2(fun).*fft2(filt)/a);
    if p==1,
       res=real(res);
    end   
    if p==0,
       res=res;
    end 
end