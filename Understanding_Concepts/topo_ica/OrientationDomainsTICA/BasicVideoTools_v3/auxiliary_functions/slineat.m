function a=slineat(pel,pos);

% SLINEAT extrae de la secuencia PEL un vector con la evolucion temporal
% del elemento [i j] de cada frame.
% 
% USO: v=slineat(PEL,[i j]);
% 
% NOTA! se supone que los frames son matrices cuadradas.

m=size(pel);
fotogramas=m(2)/m(1);

for l=1:fotogramas
      a(l)=pel(pos(1),(l-1)*m(1)+pos(2));
end