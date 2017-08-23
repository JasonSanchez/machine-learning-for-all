function(input, output) {
  output$dtrain <- DT::renderDataTable(
    DT::datatable(df_train[, input$show_vars_dtrain, drop = FALSE], options = list(pageLength = 25)) %>%
      formatStyle(outcome.var, backgroundColor='#A2BFF4')
  )
  
  output$dtest <- DT::renderDataTable(
    DT::datatable(df_test[, input$show_vars_dtest, drop = FALSE], options = list(pageLength = 25))
  )
  
  
  output$summary.label <- renderText({ 
    input$show_vars
  })
  
  output$summary.plot.train <- renderPlotly({
    if(class(df_train[,input$show_vars])=="integer" | class(df_train[,input$show_vars])=="numeric"){
      p<-ggplot(df_train, aes_string(input$show_vars))
      p<-p+geom_density(alpha = 0.4, fill="cornflowerblue")
      #p<-p+labs(title = paste("Density plot of", input$show_vars))
      p<-p+xlab("Value")
      p<-p+ylab("Percent of data with given value")
      p<-p+theme(plot.margin = unit(c(0, 1, 0.5, 1.2), "lines"))
      ggplotly(p)
    }else{
      p<-ggplot(df_train, aes_string(input$show_vars))
      p<-p+geom_bar(fill="#A2BFF4")
      #p<-p+labs(title = paste("Bar chart of", input$show_vars))
      p<-p+xlab("Value")
      p<-p+ylab("Count of data with given value")
      p<-p+theme(plot.margin = unit(c(0, 1, 0.5, 1.2), "lines"))
      ggplotly(p)
    }
  })
  
  output$summary.table.train <- renderTable({
      raw.table.train<-data.frame(summary(df_train[,input$show_vars, drop = FALSE]))
      summary.table.train<-stringr::str_split_fixed(raw.table.train[,3], ":", 2)
      summary.table.train
  }, colnames = FALSE)
  
  output$relation.plot <- renderPlotly({
    if( data.type(df_train[,input$show_vars2])=="Continuous" & data.type(df_train[,outcome.var])=="Continuous" ){
      p<-ggplot(df_train, aes_string(input$show_vars2, outcome.var))
      p<-p+geom_point(color="#A2BFF4")
      p<-p+geom_smooth(method="loess")
      p<-p+theme(plot.margin = unit(c(1, 1, 0.5, 1.2), "lines"))
      p<-p+labs(title = paste("Scatterplot of", outcome.var, "vs.", input$show_vars2), x=input$show_vars2, y=outcome.var)
      ggplotly(p)
    }else if(data.type(df_train[,input$show_vars2])=="Categorical" & data.type(df_train[,outcome.var])=="Continuous"){
      p<-ggplot(df_train,aes(factor(get(input$show_vars2)), get(outcome.var)))
      p<-p+geom_boxplot(fill="#A2BFF4")
      p<-p+theme(plot.margin = unit(c(1, 1, 0.5, 1.2), "lines"))
      p<-p+labs(title = paste("Boxplot of", outcome.var, "vs.", input$show_vars2), x=input$show_vars2, y=outcome.var)
      ggplotly(p)
    }else if(data.type(df_train[,input$show_vars2])=="Continuous" & data.type(df_train[,outcome.var])=="Categorical"){
      p<-ggplot(df_train,aes(factor(get(outcome.var)), get(input$show_vars2)))
      p<-p+geom_boxplot(fill="#A2BFF4")
      p<-p+theme(plot.margin = unit(c(1, 1, 0.5, 1.2), "lines"))
      p<-p+labs(title = paste("Boxplot of", outcome.var, "vs.", input$show_vars2), x=outcome.var, y=input$show_vars2)
      p<-p+coord_flip()
      ggplotly(p)
    }else{
      dat <- data.frame(table(df_train[,input$show_vars2],df_train[,outcome.var]))
      df_train[,input$show_var2] <- factor(df_train[,input$show_vars2])
      df_train[,outcome.var] <- factor(df_train[,outcome.var])
      names(dat) <- c(input$show_vars2, outcome.var,"Count")
      p<-ggplot(data=dat, aes_string(x=input$show_vars2, y="Count", fill=outcome.var))
      p<-p+geom_bar(stat="identity", position = "dodge")
      p<-p+theme(plot.margin = unit(c(1, 1.5, 0.5, 1.2), "lines"))
      p<-p+labs(title = paste("Bar chart of count of", outcome.var, "vs.", input$show_vars2), x=input$show_vars2, fill=outcome.var )
    }
  })
}