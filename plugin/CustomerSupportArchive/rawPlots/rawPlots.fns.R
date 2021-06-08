medSmooth = function(x, window=3) {
  y = x;
  for (i in (window+1):(length(x)-window - 1)) {
    y[i] = median(x[(i-window):(i+window)]);
  }
  y
}

meanSmooth = function(x, window=3) {
  y = x;
  for (i in 1:length(x)) {
    minX = max(1,i-window);
    maxX = min(length(x),i+window);
    y[i] = mean(x[minX:maxX]);
  }
  y
}

loadDat = function(datFile, bf, regions, smooth=0) {
  dat = readDat(
    datFile,
    minCol=unlist(lapply(regions,function(z){z$minCol})),
    maxCol=unlist(lapply(regions,function(z){z$maxCol})),
    minRow=unlist(lapply(regions,function(z){z$minRow})),
    maxRow=unlist(lapply(regions,function(z){z$maxRow})),
    uncompress=FALSE
  )

  idx = dat$row * dat$nCol + dat$col;
  maskLib = bf$maskLib[idx + 1] 
  maskTF = bf$maskTF[idx + 1]
  maskEmpty = bf$maskEmpty[idx + 1] + bf$maskReference[idx +1] # This should be maskReference, fix torrentR
  maskEmpty[maskEmpty > 1] = 1;
  
#   maskEmpty[bf$maskReference[idx + 1] == 1] = 1;
  maskPinned = bf$maskPinned[idx + 1]
  signals = list();
  for (i in 1:length(regions)) {
    s = dat$col >= regions[[i]]$minCol & dat$col < regions[[i]]$maxCol  & dat$row >= regions[[i]]$minRow & dat$row < regions[[i]]$maxRow
    mIdx = dat$col[s] + dat$row[s] * bf$nCol;
    mLib = maskLib[s];
    mTF = maskTF[s];
    mEmpty = maskEmpty[s];
    mPinned = maskPinned[s];
    name  = rep(regions[[i]]$name, length(mEmpty));
    type = rep("Other", length(mEmpty));
    type[mLib == 1] = "Lib";
    type[mTF == 1] = "TF";
    type[mEmpty == 1] = "EmptyRef";
    type[mPinned == 1] = "Pinned";
    sig = list(maskLib=mLib, Region=name, maskTF=mTF, maskEmpty=mEmpty, Type=(type), index=mIdx, time=(dat$frameStart+dat$frameEnd)/2, signal=dat$signal[s,])
    if (smooth > 0) {
      sig$signal = t(apply(sig$signal, 1, medSmooth, window=smooth));
    }
    signals[[i]] = sig;
  }

  retSig = signals[[1]];
  if (length(signals) > 1) {
    for (i in 2:length(signals)) {
      retSig$maskLib = c(retSig$maskLib, signals[[i]]$maskLib)
      retSig$maskTF = c(retSig$maskTF, signals[[i]]$maskTF)
      retSig$maskEmpty = c(retSig$maskEmpty, signals[[i]]$maskEmpty)
      retSig$Type = c(as.character(retSig$Type), as.character(signals[[i]]$Type))
      retSig$index = c(retSig$index, signals[[i]]$index)
      retSig$Region = c(retSig$Region, signals[[i]]$Region)
      retSig$signal = rbind(retSig$signal, signals[[i]]$signal);
    }
  }
  return(retSig);
}

normalizeDat = function(tDat1, tDat2) {
  f1 = apply(tDat1$signal[tDat1$maskEmpty == 1,], 2, median);
  f1 = abs(f1);
  f1 = f1 + 10;

  f2 = apply(tDat2$signal[tDat2$maskEmpty == 1,], 2, median);
  f2 = abs(f2);
  f2 = f2 + 10;

  ff = f1 / f2;
  ffm = matrix(ff, nrow = nrow(tDat1$sig), ncol=ncol(tDat1$sig), byrow=T);
  tDat1$signal = tDat1$signal / ffm;
  tDat1
}

normalizeDatToDist = function(tDat1, tDist, fudge=20, percentile=.5) {
  f1 = apply(tDat1$signal[tDat1$maskEmpty == 1,], 2, quantile, probs=c(percentile));
  f1[f1 <= 0] = 1;
  ff = (f1 + fudge) / (tDist + fudge);
  ffm = matrix(ff, nrow = nrow(tDat1$sig), ncol=ncol(tDat1$sig), byrow=T);
  tDat1$signal = tDat1$signal / ffm;
  tDat1
}

calcSignal = function(sample, reference, smooth=smooth) {
  if (smooth > 0) {
    sample$signal = t(apply(sample$signal, 1, medSmooth, window=smooth));
    reference$signal = t(apply(reference$signal, 1, medSmooth, window=smooth));
  }
  sample.norm = normalizeDat(sample, reference);
  sample.norm$signal = sample.norm$signal - reference$signal;
  sample.norm;
}

plotSig = function(tpSig, title="Plot", toPlot=c(), ylim=c(), xlim=c(), doRegionFacet=F) {

  notPinned = tpSig$Type != "Pinned";

  tpSig$Type[tpSig$Type != "EmptyRef"] = "Other";

  tp = data.frame(Type=as.factor(as.character(tpSig$Type[notPinned])), Index=tpSig$index[notPinned], Region=tpSig$Region[notPinned], tpSig$signal[notPinned,])
  colnames(tp) = sub("X","", colnames(tp))
  sig.mx = melt(tp, c("Type","Index","Region"), c(4:ncol(tp)))
  colnames(sig.mx) = c("Type","Index","Region","Time","Counts");
  sig.mx$Type = as.factor(as.character(sig.mx$Type));
  sig.mx$Time = tpSig$time[as.numeric(sig.mx$Time)];
  sig.mx$Counts = as.numeric(sig.mx$Counts);
  g = ggplot(sig.mx, aes(x=Time,y=Counts,colour=Region))
  td = g + geom_line(aes(group=Index, colour=Region, fill=Region), data=sig.mx, alpha=.3) +
     stat_summary(fun.data="median_hilow", geom="smooth", linetype=2, size=1.2, fill=I("black"), alpha=.4, conf.int=.5) +
     stat_summary(aes(y=Counts,x=Time,group=Region,colour=Region),fun.y=median, geom="line", linetype=2, size=1.2, alpha=1 ) +
    scale_colour_brewer(palette="Set1") +
    labs(title=title)
  if (doRegionFacet) {
    td = td + facet_grid(Type ~ Region)
  }
  else {
    td = td + facet_grid(Type ~ .)
  }
  if (length(ylim) > 0) {
    td = td + ylim(ylim);
  }
  if (length(xlim) > 0) {
    td = td + xlim(xlim);
  }
  try(print(td));
}

plotEmptyVsSig = function(tpSig, title="Plot", toPlot=c(), ylim=c(), xlim=c()) {
  notPinned = tpSig$Type != "Pinned";
  tpSig$Type[tpSig$Type != "EmptyRef"] = "Other";

  tp = data.frame(Type=as.factor(as.character(tpSig$Type[notPinned])), Index=tpSig$index[notPinned], Region=tpSig$Region[notPinned], tpSig$signal[notPinned,])
  #tp = data.frame(Type=as.factor(as.character(tpSig$Type[notPinned])), Index=tpSig$index[notPinned], Region=tpSig$Region[notPinned], tpSig$signal[notPinned])

  colnames(tp) = sub("X","", colnames(tp))
  sig.mx = melt(tp, c("Type","Index","Region"), c(4:ncol(tp)))
  colnames(sig.mx) = c("Type","Index","Region","Time","Counts");
  sig.mx$Type = as.factor(as.character(sig.mx$Type));
  sig.mx$Time = tpSig$time[as.numeric(sig.mx$Time)];
  sig.mx$Counts = as.numeric(sig.mx$Counts);
  g = ggplot(sig.mx, aes(x=Time,y=Counts,colour=Type))
  td = g + geom_line(aes(group=Index, colour=Type, fill=Type), data=sig.mx, alpha=.3) +
    stat_summary(fun.data="median_hilow", geom="smooth", linetype=2, size=.5, fill=I("black"), alpha=.5, conf.int=.5) +
    scale_colour_brewer(palette="Set1") +
    labs(title=title)
  td = td + facet_grid(. ~ Region)
  if (length(ylim) > 0) {
    td = td + ylim(ylim);
  }
  if (length(xlim) > 0) {
    td = td + xlim(xlim);
  }
  try(print(td));
}

bgSub = function(sig, percentile=.5) {

  sel = sig$Type == "EmptyRef";
  if (length(sel[sel]) == 0) {
    sel = rep(T,length(sig$Type)) 
  }
  empties = sig$signal[sel,]
  summaries = by(empties, sig$Region[sel], colMeans)
  sig.bg = sig;
  for (i in 1:length(summaries)) {
    sig.bg$signal[(names(summaries))[i] == sig.bg$Region,] = t(apply(sig.bg$signal[(names(summaries))[i] == sig$Region,], 1, function(x) { return (x - summaries[[i]]); }))
  }
  return(sig.bg);
}

BFBgSub = function(sig, percentile=.5) {
  summaries = by(sig$signal, sig$Region, colMeans)
  sig.bg = sig;
  for (i in 1:length(summaries)) {
    sig.bg$signal[(names(summaries))[i] == sig.bg$Region,] = t(apply(sig.bg$signal[(names(summaries))[i] == sig$Region,], 1, function(x) { return (x - summaries[[i]]); }))
  }
  return(sig.bg);
}

BFEmptySub = function(sig, percentile=.5) {
  summaries = by(sig$signal[sig$Type == "EmptyRef",], sig$Region[sig$Type == "EmptyRef"], colMeans)
  sig.bg = sig;
  for (i in 1:length(summaries)) {
    sig.bg$signal[(names(summaries))[i] == sig.bg$Region,] = t(apply(sig.bg$signal[(names(summaries))[i] == sig$Region,], 1, function(x) { return (x - summaries[[i]]); }))
  }
  sig.bg$diffMean = by(sig.bg$signal, sig$Region, colMeans)
  return(sig.bg);
}

sumSignal = function(sample, sumRange=c()) {
  if (length(sumRange) == 0) {
    sumRange = c(1:ncol(sample$signal));
  }
  sumSig = apply(sample$signal[,sumRange], 1, sum)
  sigDf = data.frame(Type = sample$Type, Signal=sumSig)
  sigDf;
}

plotSigDensity = function(sig, title="Plot", toPlot=c("EmptyRef","Lib")) {
  sig = sig[sig$Type %in% toPlot,]
  sig$Type = as.factor(as.character(sig$Type));
  g = ggplot(sig, aes(x=Signal, colour=Type, group=Type));
  try(print(g + geom_density(fill=NA) + scale_colour_brewer(palette="Set1") + labs(title=title)))
  meanDiff = mean(sig$Signal[sig$Type == "TF"]) - mean(sig$Signal[sig$Type == "EmptyRef"]);
  sd = sqrt(var(sig$Signal[sig$Type == "TF"]) + var(sig$Signal[sig$Type == "EmptyRef"]));
  snr = meanDiff / sd;
  snr;
}

sumSignal = function(sample, sumRange=c()) {
  if (length(sumRange) == 0) {
    sumRange = c(1:ncol(sample$signal));
  }
  sumSig = apply(sample$signal[,sumRange], 1, sum)
  sigDf = data.frame(Type = sample$Type, Signal=sumSig)
  sigDf;
}

expandRange2 = function(rng, factor) {
  rng2 = c(0,0);
  rng2[1] =  rng[1] - ( abs(rng[1]) * factor);
  rng2[2] = rng[2] + (abs(rng[2]) * factor);
  return(rng2);
}
