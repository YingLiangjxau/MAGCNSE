function [lfs]=lncRNAfunsim(LDA,L)
lrows=size(L,1);
lfs=zeros(lrows,lrows);
for i=1:lrows
    index1=find(L(i,:)==1);
    if length(index1)==0
        lfs(i,:)=0;
    else
        for j=1:lrows
            index2=find(L(j,:)==1);
            if length(index2)==0
                lfs(i,j)=0;
            else
                sim1=zeros(1,length(index1));
                for p=1:length(index1)
                    sim1(p)=max(LDA(index2,index1(p))');
                end
                sim2=zeros(1,length(index2));
                for q=1:length(index2)
                    sim2(q)=max(LDA(index1,index2(q))');
                end
                lfs(i,j)=(sum(sim1)+sum(sim2))/(length(index1)+length(index2));
            end
        end
    end
end
for k=1:lrows
    lfs(k,k)=1;
end
end




