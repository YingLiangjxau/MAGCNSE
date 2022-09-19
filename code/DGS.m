function [ DGS ] = GSD( LDA )
    
nd=size(LDA,2);
normSum=0;
for i=1:nd
   normSum=normSum+ ((norm(LDA(:,i),2)).^2);
end

rd=1/(normSum/nd);
DGS=zeros(nd,nd);

for i=1:nd
   for j=1:nd
       sub=LDA(:,i)-LDA(:,j);
       DGS(i,j)=exp(-rd*((norm(sub,2)).^2));
   end 
end

end

