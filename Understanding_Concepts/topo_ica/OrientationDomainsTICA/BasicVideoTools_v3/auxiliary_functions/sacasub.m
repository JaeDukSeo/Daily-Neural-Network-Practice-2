function B=sacasum(A,pto,lad,inte)

% SACASUB extrae una submatriz de tamaño m*n entorno al pto [i j] de 
% la matriz A.
%
% Se extrae segun:
%
%     Si m,n son pares:   
%
%          B=A(i-floor(m/2)+1:i+floor(m/2),j-floor(n/2)+1:j+floor(n/2))
%
%     Si m,n son impares:   
%
%          B=A(i-ceil(m/2)+1:i+ceil(m/2)-1,j-ceil(n/2)+1:j+ceil(n/2)-1)
%
% NOTAS:
%     
%   * Si m y n son impares el pto [i j] queda en el centro de la submatriz,
%     sino, queda mas cerca del extremo superior izquierdo. 
%
%   * En el caso de que el subbloque que queremos extraer sobrepase los li-
%     mites de A, se rellenan con ceros las partes no contenidas en A (de
%     forma que el punto [i j] de A queda en el centro del subbloque (centro
%     en el sentido anterior).
%
%   * Si las componentes [i j] del punto central no son enteras, cada elemen-
%     to del subbloque puede obtenerse interpolando entre los valores de sus
%     cuatro vecinos.
%
% USO: B=sacasub(A,[i j],[m n],interpola?);

a=size(A);
i=pto(1);
j=pto(2);
m=lad(1);
n=lad(2);
ptoent=floor(pto);
dd=pto-ptoent;

if (sum(abs(dd))==0)|(inte==0)
  i=floor(i);
  j=floor(j);  
  if floor(m/2)<m/2
     mf=i-ceil(m/2)+1;
     Mf=i+ceil(m/2)-1;
  else
     mf=i-floor(m/2)+1;
     Mf=i+floor(m/2);
  end
  if floor(n/2)<n/2
     mc=j-ceil(n/2)+1;
     Mc=j+ceil(n/2)-1;
  else
     mc=j-floor(n/2)+1;
     Mc=j+floor(n/2);
  end
  AAA=zeros(m,n);
  if (mf<1)|(Mf>a(1))|(mc<1)|(Mc>a(2))
     if (mf>a(1))|(Mf<1)|(mc>a(2))|(Mc<1)    
            B=zeros(m,n);
     elseif ((mf<1)&(mc<1))&((Mf>1)&(Mc>1))
            fi=1-mf+1;
            ci=1-mc+1;
            AAA(fi:m,ci:n)=A(1:m-fi+1,1:n-ci+1);
     elseif ((mf<1)&(Mc>a(2)))&((Mf>1)&(mc<a(2)))
            fi=1-mf+1;
            cf=n-(Mc-a(2));
            AAA(fi:m,1:cf)=A(1:m-fi+1,a(2)-cf+1:a(2));
     elseif ((Mf>a(1))&(mc<1))&((mf<a(1))&(Mc>1))
            ci=1-mc+1;
            ff=m-(Mf-a(1));
            AAA(1:ff,ci:n)=A(a(1)-ff+1:a(1),1:n-ci+1);
     elseif ((Mf>a(1))&(Mc>a(2)))&((mf<a(1))&(mc<a(2)))
            cf=n-(Mc-a(2));
            ff=m-(Mf-a(1));
            AAA(1:ff,1:cf)=A(a(1)-ff+1:a(1),a(2)-cf+1:a(2));
     elseif (mf<1)&((Mf>1)&(mc>1)&(Mc<a(2)))
            fi=1-mf+1;
            AAA(fi:m,1:n)=A(1:m-fi+1,mc:Mc);
     elseif (mc<1)&((Mc>1)&(mf>1)&(Mf<a(1)))
            ci=1-mc+1;
            AAA(1:m,ci:n)=A(mf:Mf,1:n-ci+1);
     elseif (Mf>a(1))&((mf<a(1))&(mc>1)&(Mc<a(2)))
            ff=m-(Mf-a(1));
            AAA(1:ff,1:n)=A(a(1)-ff+1:a(1),mc:Mc);
     elseif (Mc>a(2))&((mc<a(2))&(mf>1)&(Mf<a(1)))
           cf=n-(Mc-a(2));
           AAA(1:m,1:cf)=A(mf:Mf,a(2)-cf+1:a(2));
     end
     B=AAA;
  else
     B=A(mf:Mf,mc:Mc);
  end
else
  i=floor(i);
  j=floor(j);
  m=m+2;
  n=n+2; 
  if floor(m/2)<m/2
     mf=i-ceil(m/2)+1;
     Mf=i+ceil(m/2)-1;
  else
     mf=i-floor(m/2)+1;
     Mf=i+floor(m/2);
  end
  if floor(n/2)<n/2
     mc=j-ceil(n/2)+1;
     Mc=j+ceil(n/2)-1;
  else
     mc=j-floor(n/2)+1;
     Mc=j+floor(n/2);
  end
  AAA=zeros(m,n);
  if (mf<1)|(Mf>a(1))|(mc<1)|(Mc>a(2))
     if (mf>a(1))|(Mf<1)|(mc>a(2))|(Mc<1)    
            B=zeros(m,n);
     elseif ((mf<1)&(mc<1))&((Mf>=1)&(Mc>=1))
            fi=1-mf+1;
            ci=1-mc+1;
            AAA(fi:m,ci:n)=A(1:m-fi+1,1:n-ci+1);
     elseif ((mf<1)&(Mc>a(2)))&((Mf>=1)&(mc<=a(2)))
            fi=1-mf+1;
            cf=n-(Mc-a(2));
            AAA(fi:m,1:cf)=A(1:m-fi+1,a(2)-cf+1:a(2));
     elseif ((Mf>a(1))&(mc<1))&((mf<=a(1))&(Mc>=1))
            ci=1-mc+1;
            ff=m-(Mf-a(1));
            AAA(1:ff,ci:n)=A(a(1)-ff+1:a(1),1:n-ci+1);
     elseif ((Mf>a(1))&(Mc>a(2)))&((mf<=a(1))&(mc<=a(2)))
            cf=n-(Mc-a(2));
            ff=m-(Mf-a(1));
            AAA(1:ff,1:cf)=A(a(1)-ff+1:a(1),a(2)-cf+1:a(2));
     elseif (mf<1)&((Mf>=1)&(mc>=1)&(Mc<=a(2)))
            fi=1-mf+1;
            AAA(fi:m,1:n)=A(1:m-fi+1,mc:Mc);
     elseif (mc<1)&((Mc>=1)&(mf>=1)&(Mf<=a(1)))
            ci=1-mc+1;
            AAA(1:m,ci:n)=A(mf:Mf,1:n-ci+1);
     elseif (Mf>a(1))&((mf<=a(1))&(mc>=1)&(Mc<=a(2)))
            ff=m-(Mf-a(1));
            AAA(1:ff,1:n)=A(a(1)-ff+1:a(1),mc:Mc);
     elseif (Mc>a(2))&((mc<=a(2))&(mf>=1)&(Mf<=a(1)))
            cf=n-(Mc-a(2));
            AAA(1:m,1:cf)=A(mf:Mf,a(2)-cf+1:a(2));
     end
     BB=AAA;
  else
     BB=A(mf:Mf,mc:Mc);
  end
  B=zeros(m-2,n-2); 
  for k=2:m-1
      for l=2:n-1
          S1=BB(k,l);
          S2=BB(k,l+1);
          S3=BB(k+1,l);
          S4=BB(k+1,l+1);
%          (1-dd(1))*((1-dd(2))*S1+dd(2)*S2)+dd(1)*((1-dd(2))*S3+dd(2)*S4)
          B(k-1,l-1)=(1-dd(1))*((1-dd(2))*S1+dd(2)*S2)+dd(1)*((1-dd(2))*S3+dd(2)*S4);
      end 
  end
end