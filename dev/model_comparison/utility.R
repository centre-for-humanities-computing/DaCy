require(jsonlite)
require(kableExtra)
require(knitr)

metrics_from_json = function(file){
  name = str_split(file, "/", simplify=T)
  name = str_match(file, "metrics\\/\\/(.*)\\.json")
  name = name[length(name)]
  
  metrics = jsonlite::fromJSON(file, flatten = TRUE)
  metrics = tibble(
    "Framework" = name,
    "Accuracy" = metrics$tag_acc * 100,
    "Location" = metrics$ents_per_type$LOC$f * 100,
    "Organization" = metrics$ents_per_type$ORG$f * 100,
    "Person" = metrics$ents_per_type$PER$f * 100,
    "Misc" = metrics$ents_per_type$MISC$f * 100,
    "Avg F1" = metrics$ents_f * 100,
    "UAS" = metrics$dep_uas * 100,
    "LAS" = metrics$dep_las * 100,
    "WPS" = metrics$speed
  )
  return(metrics)
}


highlight_highest = function(kable_input, dataset, columns, underline_second=T, str_col_to_numeric=T){
  for (col in columns){
    idx = which(colnames(dataset) == col)
    column = dataset[[col]]
    if (class(column) == "character" & str_col_to_numeric){
      column = str_split(column, " ", simplify=T)[,1]
      column = as.numeric(column)
    }
    highest = if_else(column== max(column, na.rm=T), T, F, missing=F)
    
    if (underline_second){
      sorted = sort(column[column != max(column, na.rm=T)])
      second_highest = max(sorted, na.rm = T)
      second = if_else(column == second_highest, T, F, missing=F)
      kable_input = column_spec(kable_input, idx, bold = highest, underline = second)      
    } else{
      kable_input = column_spec(kable_input, idx, bold = highest)
    }
  }
  return(kable_input)
}


highlight_lowest = function(kable_input, dataset, columns, underline_second=T){
  for (col in columns){
    idx = which(colnames(dataset) == col)
    highest = if_else(dataset[[col]] == min(dataset[[col]], na.rm=T), T, F, missing=F)
    
    if (underline_second){
      sorted = sort(dataset[[col]], decreasing = T)
      second_highest = sorted[length(sorted)-1]
      second = if_else(dataset[[col]] == second_highest, T, F, missing=F)
      kable_input = column_spec(kable_input, idx, bold = highest, underline = second)      
    } else{
      kable_input = column_spec(kable_input, idx, bold = highest)
    }
  }
  return(kable_input)
}

