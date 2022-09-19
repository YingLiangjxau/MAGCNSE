> local({pkg <- select.list(sort(.packages(all.available = TRUE)),graphics=TRUE)
+ if(nchar(pkg)) library(pkg, character.only=TRUE)})

DOSE v3.18.1  For help: https://guangchuangyu.github.io/software/DOSE

If you use DOSE in published research, please cite:
Guangchuang Yu, Li-Gen Wang, Guang-Rong Yan, Qing-Yu He. DOSE: an R/Bioconductor package for Disease Ontology Semantic and Enrichment analysis. Bioinformatics 2015, 31(4):608-609

> doSim("DOID:7693","DOID:9952", measure = "Wang") #demo
[1] 0.0220824
> a=read.table("D:/DOID(lncRNADisease2).txt",header=F)
> b=read.table("D:/DOID(lncRNADisease2).txt",header=F)
> result=doSim(a,b,measure = "Wang")
> write.table (result, file ="D:/disease_semantic_similarity.txt", sep =" ", row.names =FALSE, col.names =FALSE)
> 
