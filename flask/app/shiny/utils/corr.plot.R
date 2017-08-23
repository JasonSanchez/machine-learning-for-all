corr.plot<-function(data, lim=50, sort="abs", col=c("cornflowerblue", "coral1")){
  library(dplyr)
  options(scipen = 999)
  
  #Subset numeric columns
  col.class<-sapply(data, class)
  col.class.nonchar<-names(which(col.class!="character"&col.class!="factor"))
  data.nonchar<-data[,col.class.nonchar]
  
  #Calculate correlation matrix
  cor.matrix<-cor(data.nonchar, data.nonchar, use="complete.obs")
  
  #Generate df from half matrix without diag
  cor.df<-data.frame(row=rownames(cor.matrix)[row(cor.matrix)[upper.tri(cor.matrix)]], 
                     col=colnames(cor.matrix)[col(cor.matrix)[upper.tri(cor.matrix)]], 
                     corr=cor.matrix[upper.tri(cor.matrix)])
  
  cor.df %>%
    mutate(abs=abs(corr),
           colour=ifelse(corr>0, "positive", "negative"),
           label=paste0(row, " -- ", col)) %>%
    head(lim) %>%
    ggplot(aes(reorder(label, get(sort)), corr))+
    geom_bar(stat="identity", position="identity", aes(fill=colour))+
    theme(axis.text.x=element_text(angle=90, hjust=1, vjust=0.5))+
    scale_fill_manual(values=c(positive=col[1],negative=col[2]))+
    labs(x="Features", y="Correlation")+
    theme(legend.position="none")+
    coord_flip()
}

corr.plot.outcome<-function(data, outcome, lim=50, sort="abs", col=c("cornflowerblue", "coral1")){
  library(dplyr)
  options(scipen = 999)
  
  #Subset numeric columns
  col.class<-sapply(data, class)
  col.class.nonchar<-names(which(col.class!="character"&col.class!="factor"))
  data.nonchar<-data[,col.class.nonchar]
  
  #Calculate correlation matrix
  cor.df<-data.frame(cor(dplyr::select(data.nonchar, -(get(outcome))), 
                         data[,outcome], 
                         use="complete.obs"))
  names(cor.df)<-"corr"
  
  cor.df %>%
    mutate(abs=abs(corr),
           var=rownames(cor.df),
           colour=ifelse(corr>0, "positive", "negative")) %>%
    head(lim) %>%
    ggplot(aes(reorder(var, get(sort)), corr))+
    geom_bar(stat="identity", position="identity", aes(fill=colour))+
    theme(axis.text.x=element_text(angle=90, hjust=1, vjust=0.5))+
    scale_fill_manual(values=c(positive=col[1],negative=col[2]))+
    labs(x="Features", y=paste("Correlation vs.", outcome))+
    theme(legend.position="none")+
    coord_flip()
}