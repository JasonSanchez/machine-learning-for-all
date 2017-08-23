library(markdown)

navbarPage("",
           tabPanel("Uploaded data",
                    fluidPage(
                      fluidRow(
                        column(2,
                          conditionalPanel(
                            'input.dataset == get(train_name)',
                            checkboxGroupInput('show_vars_dtrain', 'Columns in dataset to show:',
                                              names(df_train), selected = names(df_train))
                          ),
                          conditionalPanel(
                            'input.dataset == get(test_name)',
                            checkboxGroupInput('show_vars_dtest', 'Columns in dataset to show:',
                                               names(df_test), selected = names(df_test))
                          )
                        ),
                        
                        column(10,
                          tabsetPanel(
                            id = 'dataset',
                            tabPanel(train_name, DT::dataTableOutput('dtrain')),
                            tabPanel(test_name, DT::dataTableOutput('dtest'))
                          )
                        )
                      )
                    )
           ),
           
           tabPanel("Data distribution",
                    fluidPage(
                      fluidRow(
                        column(2,
                          radioButtons('show_vars', 'Choose column in dataset to summarize:',
                                             names(df_train[,target.vars])[-1], selected = names(df_train[,target.vars])[-1][1]),
                          br()
                        ),
                        
                        column(10, align="center",
                          h3(textOutput("summary.label")),
                          fluidRow(
                            column(8,
                              plotlyOutput("summary.plot.train")
                            ),
                            
                            column(2,
                              tableOutput("summary.table.train")
                            )
                          )
                        )
                      )
                    )
           ),
           
           tabPanel("Relationship with predictions",
                    fluidPage(
                      fluidRow(
                        column(2,
                          radioButtons('show_vars2', 'Choose column in dataset to plot against column you are trying to predict:',
                                       names(df_train[,target.vars])[-1], selected = names(df_train[,target.vars])[-1][1]),
                          br()
                        ),
                        
                        column(10,
                          plotlyOutput("relation.plot")
                        )
                      )  
                    )
           )
           
)