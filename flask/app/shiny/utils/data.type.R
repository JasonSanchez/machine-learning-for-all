# data.type<-function(col){
data.type<-function(col){
  if(class(col)=="date"){
    return("Date")
  }else if ( any(class(col)=="character") & length(unique(col))<=10 ){
    return("Categorical")
  }else if ( any(class(col)=="factor") & length(unique(col))<=10 ){
    return("Categorical")
  }else if ( any(class(col)=="logical") ){
    return("Categorical")
  }else if ( class(col)=="integer" & length(unique(col))<=10 ){
    return("Categorical")
  }else if ( any(class(col)=="character") & length(unique(col))>10 ){
    return("Long Categorical")
  }else if ( any(class(col)=="factor") & length(unique(col))>10 ){
    return("Long Categorical")
  }else if(class(col)=="numeric"){
    return("Continuous")
  }else if( (class(col)=="integer") & (length(unique(col))>10) ){
    return("Continuous")
  }
}