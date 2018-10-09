function [con]=sesalep(p,Dom);

% SESALEP comprueba si el punto P esta fuera de la caja
% Si se sale la respuesta es un uno, sino, un cero. 
%
% USO: respuesta=sesalep(pto,limit_caja);


if (p(1)>Dom(2))|(p(1)<Dom(1))|(p(2)>Dom(4))|(p(2)<Dom(3))|(p(3)>Dom(6))|(p(3)<Dom(5))
   con=1;
else
   con=0;
end

