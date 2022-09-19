function [ LGS ] = GSM( LDA )

nl=size(LDA,1);
normSum=0;
for i=1:nl
   normSum=normSum+ ((norm(LDA(i,:),2)).^2);
end

rm=1/(normSum/nl);
LGS = zeros(nl,nl);

for i=1:nl
   for j=1:nl
       sub=LDA(i,:)-LDA(j,:);
       LGS(i,j)=exp(-rm*((norm(sub,2)).^2));
   end 
end

end

