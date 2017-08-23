require(ggplot2)
require(dplyr)
require(purrr)

miss.plot<-function(data, only.missing=TRUE, plot.color="coral1"){
  #Calculate percent missing data for every column in input data frame
  miss_pct<-purrr::map_dbl(data, function(x){round((sum(is.na(x)) / length(x)) * 100, 1)})
  
  #Subset miss_pct>0
  if(only.missing){
    miss_pct<-miss_pct[miss_pct > 0]
  }
  
  if(miss_pct==0){
    return("No missing values in dataset.")
  }
  
  missing.df<-data.frame(pct.missing=miss_pct, var=names(miss_pct), row.names=NULL)
  
  
  #Plot percent of missing data for variables, in descending order
  miss.plot<-missing.df %>%
    ggplot(aes(x=reorder(var, -pct.missing), y=pct.missing)) + 
    geom_bar(stat='identity', fill=plot.color) +
    labs(x='', y='% missing', title='Percent missing data by feature') +
    theme(axis.text.x=element_text(angle=90, hjust=1, vjust=0.5))
  
  return(miss.plot)
}

miss.table<-function(data){
  #Calculate percent missing data for every column in input data frame
  miss_pct<-purrr::map_dbl(data, function(x){round((sum(is.na(x)) / length(x)) * 100, 1)})
  
  #Subset miss_pct>0
  miss_pct<-miss_pct[miss_pct > 0]
  
  missing.df<-data.frame(var=names(miss_pct), pct.missing=miss_pct, row.names=NULL)
  #Same thing in tabular form
  print(sprintf("%.0f / %.0f columns have missing data", nrow(missing.df), ncol(data)))
  return (dplyr::arrange(missing.df, desc(pct.missing)))
}