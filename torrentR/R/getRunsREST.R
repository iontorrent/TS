library(RCurl)
library(rjson)

#
# Read a file with some fields to be loaded. Format is two column tab
# delimited with first column being the name for the resulting data
# frame and the second column being the usual $ delimited list descriptor
# to access the desired field from the list object from R json object
# Example file:
#
# name	field
# run_id	id
# resultsName	resultsName
# sigSdSnr	pluginStore$separator$sigSdSnr
# meanBfSnr	pluginStore$separator$meanBfSnr
# libmetrics_id	libmetrics
# analysismetrics_id	analysismetrics
#
readFieldsFromFile = function(fileName) {
  d = read.table(fileName, header=T, sep="\t");
  fields = list();
  for ( i in 1:nrow(d)) {
    row = c(as.character(d$name[i]));
    row = c(row, unlist(strsplit(as.character(d$field[i]),'$', fixed=T)))
    fields[[i]] = row;
  }
  return (fields);
}

getListNameValue = function(d, field) {
  str = sprintf("d");
  for ( i in 1:length(field)) {
    str = sprintf("%s[['%s']]", str, field[i]);
  }
#  print(sprintf("Text is: %s",str));
  val = eval(parse(text=str));
#  print(paste("value is", val))
  return (val);
}

parseExperimentData = function(d, fields) {
  vals = rep(NA, length(fields))
  colnames = rep("x",length(fields))
  for (nameIx in 1:length(fields)) {
    value = getListNameValue(d, fields[[nameIx]][2:length(fields[[nameIx]])]);
    paste("Doing value",value);
    colnames[nameIx] = fields[[nameIx]][1];
    if (!is.null(value) && length(value) >= 1) {
     # print(paste("value is:", value));
      vals[nameIx] = value;
    }
  }
  names(vals) = colnames;
  df = data.frame(t(vals), stringsAsFactors=F);
  return (df);
}

getListFromRESTQuery = function(query, userpwd)  {
  x = getURL(query,
     userpwd = userpwd,
     verbose=F)
  d = fromJSON(x)
  if (!is.null(d$error_message)) {
    return (NA);
  }
  return (d);
}

makeDataFrameFromRESTQuery = function(query, fields, host, userpwd) {
  x = getURL(query,
     userpwd = userpwd,
     verbose=F)
  d = fromJSON(x)
  if (!is.null(d$error_message)) {
    return (list(d$error_message));
  }
  df = parseExperimentData(d, fields);
  return (df);
}

getLibMetricsUri = function(libMetricUri, fields, host, userpwd) {
  query = sprintf("http://%s%s?format=json", host, libMetricUri);
  return (makeDataFrameFromRESTQuery(query, fields, host, userpwd));
}

getAnalysisMetricsUri = function(analysisMetricUri, fields, host, userpwd) {
  query = sprintf("http://%s%s?format=json", host, analysisMetricUri);
 # print(paste("analysis query",analysisMetricUri));
  return (makeDataFrameFromRESTQuery(query, fields, host, userpwd));
}

getRunData = function(runId, fields, host, userpwd) {
  query = sprintf("http://%s/rundb/api/v1/results/%d/?format=json", host, runId);
 # print(paste("run query is:", query));
  return (makeDataFrameFromRESTQuery(query, fields, host, userpwd));
}

getExperimentsByProject = function(projectName, host, userpwd) {
  query = sprintf("http://%s/rundb/api/v1/experiment/?format=json&project=%s", host, projectName);
 # print(paste("query is: ", query));
  x = getURL(query,
     userpwd = userpwd,
     verbose=F)
  d = fromJSON(x)
  if (!is.null(d$error_message)) {
    return (list(d$error_message));
  }
  return(d[[2]]);
}

getRunsRESTByProject = function(projectName, fields, lmFields, amFields, host, userpwd) {
  d = getExperimentsByProject(projectName, host, userpwd);
 # print(paste("got:", length(d), "runs"));
  runIds = c();
  for (i in 1:length(d)) {
    s = strsplit(d[[i]]$results, '/')
    id = as.numeric(s[[1]][length(s[[1]])]);
    runIds = c(runIds, id);
  }
 # print(paste("run ids:" ,runIds))
  dd = getRunsREST(runIds, fields, lmFields, amFields, host, userpwd);
  return (dd);
}

getRunsREST = function(runIds, fields, lmFields, amFields, host="host.com", userpwd="name:password") {
  results = list(length(runIds));
  for (i in 1:length(runIds)) {
   # print(paste("Doing run: ", runIds[i]));
    rf = getRunData(runIds[i], fields, host, userpwd);
    if (!is.na(rf$analysismetrics_id[1]) & !is.na(rf$libmetrics_id[1])) {
     # print(paste("Doing analysis: ", runIds[i], "with uri:",rf$analysismetrics_id[1] ))
      af = getAnalysisMetricsUri(as.character(rf$analysismetrics_id[1]), amFields, host=host, userpwd=userpwd);
     # print(paste("Doing libmetrics: ", runIds[i]))
      lf = getLibMetricsUri(as.character(rf$libmetrics_id[1]), lmFields, host=host, userpwd=userpwd);
      results[[i]] = data.frame(rf,af,lf);
    } else {
     # print(paste("Bad ids for run:",runIds[i]));
    }
  }
  df = results[[1]];
  if (length(results) > 1) {
    for (i in 2:length(results)) {
      df = rbind(df, results[[i]])
    }
  }
  return(df);
}

getRunsListREST = function(runIds, host="host.com", userpwd="name:password") {
  results = list(length(runIds));
  runs = list();
  libmetrics = list();
  analysis = list();
  experiments = list()
  for (i in 1:length(runIds)) {
    print(paste("Doing run: ", runIds[i]));
    query = sprintf("http://%s/rundb/api/v1/results/%d/?format=json", host, runIds[i]);
    runs[[i]] = getListFromRESTQuery(query, userpwd);
    if (length(runs[[i]]$libmetrics) >= 1 && !is.na(runs[[i]]$libmetrics)) {
     # print(paste("Doing analysis: ", runIds[i], "with uri:",runs[[i]]$libmetrics ))
      query = sprintf("http://%s%s?format=json", host, runs[[i]]$libmetrics);
      libmetrics[[i]] = getListFromRESTQuery(query, userpwd);
    } else {
      libmetrics[[i]] = NA;
    }
    if (length(runs[[i]]$analysismetrics) >= 1 && !is.na(runs[[i]]$analysismetrics)) {
      query = sprintf("http://%s%s?format=json", host, runs[[i]]$analysismetrics);
      analysis[[i]] = getListFromRESTQuery(query, userpwd);
    } else {
      analysis[[i]] = NA;
    }
    if (length(runs[[i]]$experiment) >= 1 && !is.na(runs[[i]]$experiment)) {
      query = sprintf("http://%s%s?format=json", host, runs[[i]]$experiment);
      experiments[[i]] = getListFromRESTQuery(query, userpwd);
    } else {
      experiments[[i]] = NA;
    }
  }
  return(list(runs=runs, libmetrics=libmetrics,analysis=analysis, experiments=experiments));
}

getRunsListRESTByProject = function(projectName, host, userpwd) {
  d = getExperimentsByProject(projectName, host, userpwd);
 # print(paste("got:", length(d), "runs"));
  runIds = c();
  for (i in 1:length(d)) {
    s = strsplit(d[[i]]$results, '/')
    id = as.numeric(s[[1]][length(s[[1]])]);
    runIds = c(runIds, id);
  }
 # print(paste("run ids:" ,runIds))
  dd = getRunsListREST(runIds, host, userpwd);
  return (dd);
}

calcAdditionalFaveMetrics = function(df) {
  df$live_to_100Q17 = as.numeric(df$i100Q17_reads) / as.numeric(df$live);
  df$live_to_q7_alignments = as.numeric(df$q7_alignments) / as.numeric(df$live);
  df$reads_to_q7_alignments = as.numeric(df$q7_alignments) / as.numeric(df$totalNumReads);
  df$i50Q17_to_i100Q17 = as.numeric(df$i100Q17_reads) / as.numeric(df$i50Q17_reads);
  return (df);
}

getBasicFields = function() {
fields <-
list(c("run_id", "id"), c("resultsName", "resultsName"), c("sigSdSnr", 
"pluginStore", "separator", "sigSdSnr"), c("meanBfSnr", "pluginStore", 
"separator", "meanBfSnr"), c("libmetrics_id", "libmetrics"), 
    c("analysismetrics_id", "analysismetrics"), c("timmed_filtered_100Q17", 
    "pluginStore", "filterAndTrim", "100q17", "filtered trimmed"
    ), c("unfiltered_untrimmed_100Q17", "pluginStore", "filterAndTrim", 
    "100q17", "unfiltered untrimmed"), c("bubble_percent", "pluginStore", 
    "bubbleRun", "covered_percent"), c("spatial_50Q17ConvSlope", 
    "pluginStore", "spatialPlots", "50Q17ConvSlope"), c("spatial_100Q17Slope", 
    "pluginStore", "spatialPlots", "100Q17Slope"))
lmFields <-
list(c("cf", "cf"), c("ie", "ie"), c("dr", "dr"), c("totalNumReads", 
"totalNumReads"), c("q7_alignments", "q7_alignments"), c("q10_alignments", 
"q10_alignments"), c("i50Q7_reads", "i50Q7_reads"), c("i50Q10_reads", 
"i50Q10_reads"), c("i50Q17_reads", "i50Q17_reads"), c("i50Q20_reads", 
"i50Q20_reads"), c("i100Q7_reads", "i100Q7_reads"), c("i100Q10_reads", 
"i100Q10_reads"), c("i100Q17_reads", "i100Q17_reads"), c("i100Q20_reads", 
"i100Q20_reads"), c("q17_mapped_bases", "q17_mapped_bases"), 
    c("q17_mean_alignment_length", "q17_mean_alignment_length"
    ), c("sysSNR", "sysSNR"))
amFields <-
list(c("bead", "bead"), c("dud", "dud"), c("empty", "empty"), 
    c("ignored", "ignored"), c("lib", "lib"), c("tf", "tf"), 
    c("live", "live"), c("pinned", "pinned"), c("sysCF", "sysCF"
    ), c("sysDR", "sysDR"), c("sysIE", "sysIE"))
return(list(fields,lmFields,amFields));
}

