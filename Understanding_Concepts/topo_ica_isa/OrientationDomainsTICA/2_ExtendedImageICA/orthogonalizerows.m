%orthogonalize rows of a matrix. 

function Wort=orthogonalizerows(W)

Wort = real((W*W')^(-0.5))*W;

return;
