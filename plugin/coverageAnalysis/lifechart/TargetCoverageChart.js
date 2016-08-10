// html for chart container and filters bar - note these are invisible and moved into position later
document.write('\
<div id="TC-chart" class="unselectable" style="border:2px solid #666;page-break-inside:avoid;display:none">\
  <div id="TC-titlebar" class="grid-header" style="min-height:24px;border:0">\
    <span id="TC-collapsePlot" style="float:left" class="ui-icon ui-icon-triangle-1-n" title="Collapse View"></span>\
    <span id="TC-titletext" class="table-title" style="float:none">Target Coverage Chart</span>\
    <span id="TC-PlotOptions">\
      <span class="TC-shy flyhelp" id="TC-plotLabel" style="padding-left:20px">Plot:</span>\
      <select class="TC-selectParam TC-shy txtSelect" id="barAxis">\
       <option value=0 selected="selected">Total Reads</option>\
       <option value=2 style="display:none">Read Depths</option>\
       <option value=1>Strand Reads</option>\
      </select>\
    </span>\
    <span class="TC-shy flyhelp" id="TC-overlayLabel" style="padding-left:10px">Overlay:</span>\
    <select class="TC-selectParam TC-shy txtSelect" id="overPlot">\
     <option value=0 selected="selected">None</option>\
     <option value=1>Target GC%</option>\
     <option value=2>Target Length</option>\
     <option value=3>Strand Bias</option>\
     <option value=4>GC/AT Bias</option>\
    </select>\
    <input class="TC-shy" id="TC-unzoomToggle" type="button" value="Zoom Out" style="margin-left:10px;width:70px">\
    <span id="TC-toggleControls" style="float:right" class="TC-shy ui-icon ui-icon-search" title="Show/Hide the view/filter options panel"></span>\
    <span id="TC-help" style="float:right;margin-left:0;margin-right:0" class="TC-shy ui-icon ui-icon-help"></span>\
    <span id="TC-message" class="message"></span>\
  </div>\
  <div id="TC-noncanvas" style="background:#EEF;border-top:2px solid #666">\
   <div id="TC-plotspace" style="padding:4px">\
    <div id="TC-placeholder" style="width:100%"></div>\
   </div>\
  </div>\
</div>\
<div id="TC-controlpanel" style="display:none;padding:4px;border-top:solid 1px #666">\
  <table style="float:none;width:auto"><tr>\
    <td class="nwrap">Viewing Options:</td>\
    <td class="nwrap"><span class="flyhelp" id="TC-logAxisLabel">Log Axis</span>:\
      <input type="checkbox" id="TC-logAxis" checked="checked"></td>\
    <td class="nwrap"><span class="flyhelp" id="TC-autoZoomLabel">Automatic Zoom</span>:\
      <input type="checkbox" id="TC-autoZoom" checked="checked"></td>\
    <td class="nwrap"><span class="flyhelp" id="TC-showLegendLabel">Show Legend</span>:\
      <input type="checkbox" id="TC-showLegend" checked="checked"></td>\
    <td class="nwrap"><span class="flyhelp" id="TC-numPointsLabel">Bars/Points</span>:\
      <input class="txtSearch" type="text" id="TC-numPoints" value=100 size=4>&nbsp;<span id="TC-numBars"></span></td>\
    <td><input id="TC-export" type="button" value="Export"></td>\
  </tr></table>\
  <table><tr>\
    <td id="TC-filtertitle" class="nwrap">Data Filters:</td>\
    <td class="nwrap"><span class="flyhelp" id="TC-filterDepthLabel">Depth</span>:\
      <input type="text" class="txtSearch" id="TC-filterCovMin" value="0" size=4>&nbsp;-&nbsp;<input type="text" class="txtSearch" id="TC-filterCovMax" value="" size=4></td>\
    <td id="TC-chromFilter" class="nwrap"><span class="flyhelp" id="TC-filterChromLabel">Chrom/Contig</span>:\
      <select class="txtSelect" id="TC-selectChrom"></select></td>\
    <td class="nwrap"><span class="flyhelp" id="TC-filterGeneSymLabel">Attribute</span>:\
      <input class="txtSearch" type="text" id="TC-filterGeneSym" value="" size=19></td>\
    <td><input id="TC-clearFilters" type="button" value="Clear" style="width:45px"></td>\
  </tr></table>\
</div>\
<div id="TC-tooltip" style="display:none">\
  <div><span id="TC-tooltip-close" title="Close" class="help-box ui-icon ui-icon-close"></span></div>\
  <div><span id="TC-tooltip-zoomout" title="Zoom out from this region" class="help-box ui-icon ui-icon-zoomout"></span></div>\
  <div><span id="TC-tooltip-center" title="Center view on this region" class="help-box ui-icon ui-icon-arrowthick-2-e-w"></span></div>\
  <div><span id="TC-tooltip-zoomin" title="Zoom in on this region" class="help-box ui-icon ui-icon-zoomin"></span></div>\
  <div id="TC-tooltip-body"></div>\
  <div id="TC-tooltip-controls" class="controlbox" style="display:none">\
    <input type="button" id="TC-OpenInIGV" value="View in IGV">\
    <input type="button" id="TC-OpenInRCC" value="View in Reference Coverage Chart">\
  </div>\
</div>\
<div id="TC-mask" class="grid-mask"></div>\
<div id="TC-dialog" class="tools-dialog" style="display:none">\
  <div id="TC-dialog-title" class="title">Export Targets in View</div>\
  <div id="TC-dialog-content" class="content">...</div>\
  <div id="TC-dialog-buttons" class="buttons">\
    <input type="button" value="OK" id="TC-exportOK">\
    <input type="button" value="Cancel" onclick="$(\'#TC-dialog\').hide();$(\'#TC-mask\').hide();">\
  </div>\
</div>\
<div id="TC-helptext" class="helpblock" style="display:none"></div>\
<input type="hidden" id="TC-ViewRequest"/>\
');

$(function () {

  // check placer element exists
  if( !$('#TargetCoverageChart').length ) return;

  // check browser environment
  var fixIE = (typeof G_vmlCanvasManager != 'undefined');
  var useFlash = (typeof FlashCanvas != 'undefined');
  var useExCan = (fixIE && !useFlash);

  // minimum sizes for chart widget
  var def_minWidth = 625;
  var def_minHeight = 200;

  // configure widget size and file used from placement div attributes
  var coverageFile = $("#TargetCoverageChart").attr("datafile");
  if( coverageFile == undefined || coverageFile == "" ) {
    //alert("ERROR on page: TargetCoverageChart widget requires attribute 'datafile' is set.");
    $('#TargetCoverageChart').hide();
    return;
  }
  var initCovFile = $("#TargetCoverageChart").attr("initfile");
  if( initCovFile == undefined ) initCovFile = '';
  var tsvOverride = (initCovFile == '') ? '' : initCovFile;

  // default customization set for target seq with target base coverage & averaging
  var ampliconReads = false; // just changes description text
  var baseCoverage = true;   // false => number of reads rather than base reads
  var enableReadDepthPlot = false; // input enabled display option

  // customization is controlled by just one flag for now indicating amplicon reads
  // but more flags could be added later or geven ability to switch between two data sets
  var amplicons = $("#TargetCoverageChart").attr("amplicons");
  if( amplicons == undefined || amplicons == '' ) amplicons = 0;
  if( amplicons > 0 && amplicons <= 3 ) {
    ampliconReads = true;
    baseCoverage = false;
  }
  var transcriptBed = (amplicons == 2 || amplicons == 3) ? 1 : 0;
  var contigChart = (amplicons == 3 || amplicons == 4);
  var reportPassingCov = false; // transcriptBed;
  // lengthNormal => make averages by total length (number of bases) ELSE divide by bin size
  var lengthNormal = baseCoverage || !ampliconReads;

  var autolegend = $("#TargetCoverageChart").attr("autolegend");
  autolegend = (autolegend != undefined);
  var startHideLegend = $("#TargetCoverageChart").attr("hidelegend");
  startHideLegend = (startHideLegend != undefined);

  var startCollapsed = $("#TargetCoverageChart").attr("collapse");
  startCollapsed = (startCollapsed != undefined);

  // possible input options
  showPlotOptions = true;
  autoJumpToGene = true;

  if( transcriptBed ) {
    $('#TC-chromFilter').hide();
    $('#TC-OpenInRCC').hide();
  }
  var tmp = $('#TargetCoverageChart').width();
  if( tmp < def_minWidth ) tmp = def_minWidth;
  $("#TC-chart").width(tmp);
  tmp = $('#TargetCoverageChart').height();
  if( tmp < def_minHeight ) tmp = def_minHeight;
  $("#TC-chart").height(tmp);
  $("#TC-placeholder").height(tmp-36);
  $("#TargetCoverageChart").css('height','auto');

  $("#TC-controlpanel").appendTo('#TC-titlebar');
  $("#TC-chart").appendTo('#TargetCoverageChart');
  $("#TC-chart").show();
  $('#TC-chart').css("height","auto");

  // some default values for plot display
  var def_minPoints = 11;
  var def_numPoints = 100;
  var def_hugeCov = 100000000;
  var disableTitleBar = false;
  var placeholder = $("#TC-placeholder");
  var timeout = null;

  var dblclickUnzoomFac = 10;

  var resiz_def = {
    alsoResize: "#TC-placeholder",
    minWidth:def_minWidth,
    minHeight:def_minHeight,
    handles:"e,s,se",
    resize:function(e,u){ updatePlot(); }
  };
  $('#TC-chart').resizable(resiz_def);

  placeholder.bind("mouseleave", function() {
    if( !lastHoverBar.sticky ) hideTooltip();
  });

  $("#TC-collapsePlot").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#TC-plotspace').is(":visible") ) {
      $(this).attr("class","ui-icon ui-icon-triangle-1-s");
      $(this).attr("title","Expand view");
      $('#TC-controlpanel').slideUp();
      $('.TC-shy').fadeOut(400);
      $('#TC-chart').resizable('destroy');
      $('#TC-noncanvas').slideUp('slow');
      hideTooltip();
    } else {
      $(this).attr("class","ui-icon ui-icon-triangle-1-n");
      $(this).attr("title","Collapse view");
      $('.TC-shy').fadeIn(400);
      $('#TC-noncanvas').slideDown('slow',function(){
        $('#TC-chart').resizable(resiz_def);
      });
    }
    $("#TC-chart").css('height','auto');
  });

  $("#TC-toggleControls").click(function(e) {
    if( disableTitleBar ) return;
    $('#TC-chart').css("height","auto");
    if( $('#TC-controlpanel').is(":visible") ) {
      $('#TC-controlpanel').slideUp();
    } else {
      $('#TC-controlpanel').slideDown();
    }
  });

  $("#TC-help").click( function() {
    var offset = $("#TC-help").offset();
    var ypos = offset.left - $('#TC-helptext').width();
    $("#TC-help").removeAttr("title");
    $('#TC-helptext').css({
      position: 'absolute', display: 'none',
      top: offset.top+16, left: ypos+8
    }).appendTo("body").slideDown();
  });

  $("#TC-help").hover( null, function() {
    $('#TC-helptext').fadeOut(200);
    $("#TC-help").attr( "title", "Click for help." );
  });

  function zoomViewOnBin(binNum,zoomIn) {
    // Always perform zoom out if binNum < 0
    if( plotStats.numPlots <= 0 ) return false;
    var overzoom = plotStats.minX > 0 || plotStats.maxX < plotStats.numPoints;
    var srt = tsvFilter.clipleft;
    var end = tsvFilter.clipright;
    var siz = end - srt;
    if( binNum >= 0 && zoomIn ) {
      if( overzoom ) return false;
      if( plotStats.binnedData ) {
        zoomToRange( binNum, binNum+1 );
        return true;
      }
      return centerViewOnBin(binNum);
    } else {
      if( !plotStats.zoom ) return false;
      // return to previous view if in over-zoom
      if( overzoom ) {
        plotStats.minX = 0;
        plotStats.maxX = plotStats.numPoints;
        plotStats.targetsRepresented = plotStats.numPoints;
        updatePlot();
        return true;
      } else if( binNum >= 0 ) {
        // override limits by selected bin if provided
        var cbin = (plotStats.numPoints-1) >> 1;
        var csrt = dataTable[binNum][DataField.pos_start];
        var cend = dataTable[binNum][DataField.pos_end];
        var csiz = cend - csrt - 1;
        csrt -= cbin * csiz;
        cend += (cbin * siz) - 1;
        if( csrt > 0 && cend <= plotStats.chromLength ) {
          srt = csrt;
          end = cend;
          siz = csiz;
        }
      }
      // zoomout to dblclickUnzoomFac from center of current view
      siz *= dblclickUnzoomFac;
      srt = 0.5*(end+srt-siz);
    }
    return windowView(srt,end,siz);
  };

  function centerViewOnBin(binNum) {
    if( plotStats.numPlots <= 0 || binNum < 0 ) return false;
    if( !plotStats.zoom ) return false;
    // binNum 0-based so 49 (bin#50) is center of 100 bins and 50 (bin#51) is center of 101 bins
    var cbin = (plotStats.numPoints-1) >> 1;
    if( binNum == cbin ) return true;
    var srt = tsvFilter.clipleft;
    var end = tsvFilter.clipright;
    var siz = end - srt;
    srt += (binNum - cbin) * (siz / plotStats.numPoints);
    return windowView(srt,end,siz);
  }

  function windowView(srt,end,siz) {
    // ensure new view of data is within range and correct flags are set, etc.
    if( srt < 0 ) srt = 0;
    end = srt + siz;
    if( end > 100 ) {
      end = 100;
      srt = end - siz;
      if( srt < 0 ) srt = 0;
    }
    if( tsvFilter.clipleft == srt && tsvFilter.clipright == end ) {
      return false; // no update needed
    } else if( srt <= 0 && end >= 100 ) {
      unzoomData();
    } else {
      tsvFilter.clipleft = srt;
      tsvFilter.clipright = end;
      zoomData();
    }
    return true;
  }

  //$('#TC-chart').noContext();

  function rightClickMenu(e) {
    alert("r-click");
  }

  // attempt to disable defalt context menu and enable r-click
  if( useFlash ) {
    // only works for flashcanvas pro!
    FlashCanvas.setOptions( {disableContextMenu : true} );
    // instead use area outside of canvas
    //$('#TC-noncanvas').noContext();
    //$('#TC-noncanvas').rightClick(rightClickMenu);
  } else {
    //$('#TC-chart').noContext();
    //$('#TC-chart').rightClick(rightClickMenu);
  }

  var plotStats = {
    defNumPoints : def_numPoints,
    minNumPoints : def_minPoints,
    maxNumPoints : 1000,
    targetsTotal : 0,
    targetsSelected : 0,
    targetsRepresented : 0,
    targetBinSize : 0,
    binnedData : false,
    numFields : 0,
    numPlots : 0,
    numPoints : 0,
    zoomMinLoad : 0,
    zoom: false,
    minX : 0,
    maxX : 0,
    minY : 0,
    maxY : 0,
    totalMinY : 0,
    totalMaxY : 0,
    strandMinY : 0,
    strandMaxY : 0,
    tooltipZero : 0,
    chrList : ""
  };

  var plotParams = {
    resetYScale: false,
    logAxis : true,
    showLegend : true,
    autoJumpToGene : autoJumpToGene,
    numPoints : def_numPoints,
    aveBase : 1,
    barAxis : 0,
    overPlot : amplicons == 1 ? 1 : 0,
    zoomMode : 1
  };

  var tsvFilter = {
    options : (transcriptBed ? '-c' : ''),
    dataFile : '',
    chrom : '',
    gene : '',
    covmin : 0,
    covmax : def_hugeCov,
    maxrows : 100,
    clipleft : 0,
    clipright : 100,
    numrec : 0
  };

  var DataField = {
    contig_id : 0,
    pos_start : 1,
    pos_end : 2,
    target_id : 3,
    gene_id : 4,
    target_gc : 5,
    cov_length : 6,
    uncov_5p : 7,
    uncov_3p : 8,
    reads_tot : 9,
    reads_fwd : 10,
    reads_rev : 11,
    bin_size : 1,
    bin_length : 2,
    sum_gcbias : 4,
    cov_20x : 12,
    cov_100x : 13,
    cov_500x : 14
  };

  var LegendLabels = {
    targType : "Target",
    meanType : "Average ",
    rcovType : "Reads",
    rcovOrig : "Reads",
    allReads : "Total Reads",
    fwdReads : "Forward Reads",
    revReads : "Reverse Reads",
    allBaseReads : "Total Base Read Depth",
    fwdBaseReads : "Forward Base Reads Depth",
    revBaseReads : "Reverse Base Reads Depth",
    allBaseRead_cov : "% Target length covered",
    allBaseReads_u3p : "% Target length uncovered at 3'",
    allBaseReads_u5p : "% Target length uncovered at 5'",
    fwdBaseReads_u3p : "% Target length uncovered at 3'",
    revBaseReads_u5p : "% Target length uncovered at 5'",
    covDepth_1x : "% Base Coverage at <20x",
    covDepth_20x : "% Base Coverage at 20x",
    covDepth_100x : "% Base Coverage at 100x",
    covDepth_500x : "% Base Coverage at 500x",
    allReads_pss : "% Passing Reads",
    fwdReads_pss: "Forward % Passing Reads",
    revReads_pss : "Reverse % Passing Reads",
    allReads_e2e : "% End-to-end Reads",
    fwdReads_e2e : "Forward % End-to-end Reads",
    revReads_e2e  : "Reverse % End-to-end Reads",
    percentGC : "GC Content",
    targLen : "Target Length",
    fwdBias : "Fwd Strand Bias",
    gcBias : "GC/AT Bias"
  }

  var ColorSet = {
    allReads : "rgb(128,160,192)",
    allReads_sd2 : "rgb(64,80,160)",
    allReads_shd : "rgb(0,0,128)",
    fwdReads : "rgb(240,120,100)",
    revReads : "rgb(100,240,120)",
    fwdReads_shd: "rgb(160,32,32)",
    revReads_shd : "rgb(32,160,32)",
    covDepth_1x : "rgb(255,220,96)",
    covDepth_20x : "rgb(237,194,64)",
    covDepth_100x : "rgb(180,143,32)",
    covDepth_500x : "rgb(129,105,0)",
    allReads_pss : "% Passing Reads",
    percentGC : "rgb(200,62,128)",
    targLen : "rgb(64,126,200)",
    fwdBias : "rgb(220,96,200)",
    gcBias : "rgb(190,190,190)"
  }

  function customizeChart() {
    LegendLabels.targType = ampliconReads ? "Amplicon" : "Target";
    if( amplicons == 3 ) LegendLabels.targType = "Contig";
    if( amplicons == 4 ) LegendLabels.targType = "Chromosome";
    LegendLabels.meanType = (baseCoverage || ampliconReads) ? "Average " : "Normalized ";
    LegendLabels.rcovType = baseCoverage ? (plotParams.aveBase ? "Base Read Depth" : "Total Base Reads") : "Assigned Reads";
    LegendLabels.rcovOrig = baseCoverage ? "base reads" : "assigned reads";
    LegendLabels.allBaseRead_cov = "% "+LegendLabels.targType+" length covered";
    LegendLabels.allBaseReads_u3p = "% "+LegendLabels.targType+" length uncovered at 3'";
    LegendLabels.allBaseReads_u5p = "% "+LegendLabels.targType+" length uncovered at 5'";
    LegendLabels.fwdBaseReads_u3p = "% "+LegendLabels.targType+" length uncovered at 3'";
    LegendLabels.revBaseReads_u5p = "% "+LegendLabels.targType+" length uncovered at 5'";

    var rtp = " " + LegendLabels.rcovType.toLowerCase();
    var trg = LegendLabels.targType.toLowerCase();

    $('#TC-titletext').text(LegendLabels.targType+' Coverage Chart');
    $('#TC-filterDepthLabel').text(ampliconReads ? 'Reads' : 'Depth');

    // cusomize primary help text
    var barShading = ampliconReads ? "number of amplicons covered " + (reportPassingCov ? "at 70% of their length" : "from end to end")
       : "proportion of uncovered target regions at the 3' and/or 5' ends";
    $('#TC-helptext').html(
      "This chart shows the representation ("+LegendLabels.rcovType.toLowerCase()+ ") of individual targeted regions.<br/><br/>"+
      "Typically there are more individual regions than space available for display and a<br/>"+
      "data bar (or point) will represent a number of binned "+trg+"s, with corresponding<br/>"+
      "values representing totals or averages for those "+trg+"s. Data bars may be shaded<br/>"+
      "to depict the (average) "+barShading+".<br/><br/>"+
      "The plot area may be re-sized by dragging the borders of the chart or hidden<br/>"+
      "using the Collapse View button in the upper right corner.<br/><br/>"+
      "Moving the mouse over data bar or point in the graph area will show some minimal<br/>"+
      "information about the data plotted. Clicking on the same data will produce a more<br/>"+
      "detailed information box that remains until dismissed.<br/><br/>"+
      "Click and drag the mouse to select a region in the plot space to zoom in.<br/>"+
      "You may also zoom in to a set of binned targets by double-clicking on a specific<br/>"+
      "data bar. Note that additional controls become available via the information box<br/>"+
      "when only data for an individual target is displayed. (Zoom in if necessary.)<br/>"+
      "Double-click in the white-space around the plotted data to zoom-out (by 10x).<br/>"+
      "Or, click on the Zoom Out button to return to the view at maximum zoom out.<br/><br/>"+
      "You may change how the data is viewed using the Plot and Overlay selectors and<br/>"+
      "other controls on the options panel, opened using the search icon in the title bar.<br/><br/>"+
      "Look for additional fly-over help on or near the controls provided." );

    // add fly-over help to controls here in case need to customize for chart data
    customizePlotOptions();
    $("#TC-overlayLabel").attr( "title",
      "Select a particular property of the (binned) targets to plot over the coverage (bar) data to see " +
      "if there is an obvious correlation between target(s) representation and that property. " +
      "Note that this is the averaged value if the bar represents binned data for multiple "+trg+"s. " +
      "For example, Target GC would be the percentage of G and C bases in the combined "+trg+" sequences. " +
      "In some cases, a correlation might be more apparent looking over a binned set of targets (that have " +
      "similar representation) than individual targets." );
    $("#TC-unzoomToggle").attr( "title",
      "Click this button to 'zoom out' the view to show coverage for all"+rtp+"(the initial view). " +
      "This button only has an effect if the view was previously zoomed-in to show a subset of the "+trg+"s." );
    $("#TC-autoZoomLabel").attr( "title", 
      "Select how the y-axis zoom works when selecting a set of "+trg+"s (x-axis data) to zoom in to view. " +
      "When checked the y-axis range will be automatically set to the largest"+rtp+
      "value of the "+trg+"s currently in view. This mode is particularly useful to magnify the view for "+trg+
      "s with relatively low representation. When unchecked the range is set to the largest"+rtp+
      "value for all "+trg+"s in the set selected by the "+LegendLabels.targType+
      " Filters, regardless of which "+trg+"s are currently in view." );
    var logMsg = baseCoverage ? "BRD+1" : "AR+1";
    $("#TC-logAxisLabel").attr( "title", "Check to display the "+LegendLabels.rcovType+" axis with log10 scaling."+
      " Log axis values are log10("+logMsg+") to give positive numbers and so 0 bar height => 0 reads." );
    $("#TC-showLegendLabel").attr( "title", "Select whether the legend is displayed over the plot area." );
    $("#TC-numPointsLabel").attr( "title",
      "This value specifies the maximum number of data bars and overlay points to display. Should there be more "+trg+
      "s than this number, their data will be binned evenly in to the selected number of data bars. The number of "+trg+
      "s and both total and average values are reported by the fly-over help for each data bar. If there are less "+trg+
      "s than this number only those "+trg+"s are shown and the actual number represented is shown in parentheses. " +
      "You may wish to reset this to the total number of "+trg+" if you wish to see all targets in one plot " +
      "without binninget to any value between 10 and 1,000, although setting a value great than 200 is not recommended "+
      "as this may make selecting individual data for review difficult." );
    $("#TC-export").attr( "title", "Click to open Export "+LegendLabels.targType+"s in View dialog." );
    $("#TC-dialog-title").html( "Export "+LegendLabels.targType+"s in View" );
    $("#TC-filterDepthLabel").attr( "title",
      "Filter data presented using a range of"+rtp+" (the y-axis range). There are two settings for the " +
      "minimum and maximum thresholds. After typing a number press tab or enter to update the view. " +
      "Tip: Uncheck the Log Axis option when using this filter, since it filters on the non-logged data." );
    $("#TC-filterChromLabel").attr( "title",
      "Use this selector to select a particular chromosome (or contig) of the reference to filter to only "+trg+
      "s on this chromosome, or to set to no filter by selecting the 'ALL' value. If there is only one chromosome in your "+
      "reference this value is set and cannot be changed." );
    $("#TC-filterGeneSymLabel").attr( "title",
      "Type in a gene symbol or name to filter the view to just "+trg+"s in that gene or having that "+trg+" ID. "+
      "Alternatively you filter on any single attribute specified in the "+trg+"s file by giving the name and value, "+
      "e.g. 'Pool=1' or 'GENE_ID=TBP'. Press the enter or tab key to perform the filter. "+
      "The attibute text typed in must match that of the "+trg+"(s) exactly but is not case-sensitive." );
    $("#TC-clearFilters").attr( "title",
      "Click this button to clear all specified filters: Coverage data is presented for all "+trg+"s." );
    $("#TC-help").attr( "title", "Click for help." );
  }

  function customizePlotOptions() {
    // separate function as dependent on coverage file read - for backwards compatibility
    if( !showPlotOptions ) {
      $('#TC-PlotOptions').hide();
    }
    $('#TC-PlotOptions').show();
    var rtp = " " + LegendLabels.rcovType.toLowerCase();
    $("#TC-plotLabel").attr( "title",
      "Select the how the data is plotted.\n'Total Reads' shows bar plots of"+rtp+"to both DNA strands.\n"+
      (enableReadDepthPlot ? "'Read Depths' shows the same bar plots colored by percentage of targets covered at specific base read depths.\n" : "")+
      "'Strand Reads' shows bar plots of forward and reverse DNA strand"+rtp+"separately, above and below the 0 reads line." );
    if( enableReadDepthPlot ) {
      $("#barAxis option[value=2]").show();
    } else {
      $("#barAxis option[value=2]").hide();
    }
  }

  // (re)initiailize page from default user options
  customizeChart();
  updateGUIPlotParams();

  // --------- initialize plot bindings - controls inside plot area ----------

  var fieldIds = [];
  var dataTable = [];
  var plotData = [];
  var options = [];
  var plotObj = null;
  var canvas = null;

  var lastHoverBar = {binNum:-1, isRev:false, clickItem:null, postZoom:false, sticky:false, label:''};

  // plotselected (area) is mapped to zoom - which is quite complicated for this app.
  placeholder.bind("plotselected", function(event, ranges) {
    if( plotParams.zoomMode == 2 ) {
      plotStats.minY = options.yaxes[0].min = ranges.yaxis.from;
      plotStats.maxY = options.yaxes[0].max = ranges.yaxis.to;
    }
    zoomToRange( ranges.xaxis.from, ranges.xaxis.to );
  });

  function zoomToRange(xLeft,xRight) {
    plotStats.zoom = true;
    lastHoverBar.postZoom = true;
    // determine if zoom requires new binning request
    if( plotStats.binnedData ) {
      var scl = (tsvFilter.clipright - tsvFilter.clipleft) / plotStats.numPoints;
      tsvFilter.clipright = tsvFilter.clipleft + scl * xRight;
      tsvFilter.clipleft += scl * xLeft;
      // check and adjust zoom to loading no less than required number of bins
      var clip = tsvFilter.clipright - tsvFilter.clipleft;
      if( clip < plotStats.zoomMinLoad ) {
        // spread the excess clip equally to the zoom area, accounting for boundaries at 0 and 100%
        var diff = 0.5 * (plotStats.zoomMinLoad - clip);
        tsvFilter.clipleft -= diff;
        tsvFilter.clipright += diff;
        if( tsvFilter.clipleft < 0 ) {
          tsvFilter.clipleft = 0;
          tsvFilter.clipright = plotStats.zoomMinLoad;
        } else if( tsvFilter.clipright > 100 ) {
          tsvFilter.clipleft = 100 - plotStats.zoomMinLoad;
          tsvFilter.clipright = 100;
        }
      }
      zoomData();
    } else {
      var clip = xRight - xLeft;
      if( clip >= plotStats.minNumPoints ) {
        plotStats.minX = options.xaxis.min = Math.floor(xLeft);
        plotStats.maxX = options.xaxis.max = Math.ceil(xRight);
      } else {
        var diff = 0.5 * (plotStats.minNumPoints - clip);
        plotStats.minX = Math.floor(0.5+xRight-diff);
        plotStats.maxX = plotStats.minX + plotStats.minNumPoints;
        if( plotStats.minX < 0 ) {
          plotStats.maxX -= plotStats.minX;
          plotStats.minX = 0;
        }
        if( plotStats.maxX > plotStats.numPoints ) {
          plotStats.minX -= plotStats.maxX - plotStats.numPoints;
          plotStats.maxX = plotStats.numPoints;
        }
      }
      plotStats.targetBinSize = 1;
      plotStats.targetsRepresented = Math.floor(plotStats.maxX - plotStats.minX);
      updatePlot();
    }
  }

  placeholder.bind("plothover", function(event, pos, item) {
    var hoverTip = !lastHoverBar.sticky;
    if( cursorOverPlot(pos.x,pos.y) ) {
      if( cursorOverItem(pos,item) ) {
        setCursor('pointer');
        if( hoverTip ) showTooltip(item,pos,false);
      } else {
        setCursor('crosshair');
        if( hoverTip ) hideTooltip();
      }
    } else {
      setCursor('default');
      if( hoverTip ) hideTooltip();
    }
  });

  var numClicks = 0;
  var clickTimer = null;
  placeholder.bind("plotclick", function(e,pos,item) {
    // manual implement of dblclick since single click always fires
    if( ++numClicks > 1 ) {
      clearTimeout(clickTimer);
      numClicks = 0;
      var binNum = item ? Math.floor(pos.x) : -1;
      if( zoomViewOnBin( binNum, true ) ) hideTooltip();
      return;
    }
    // ignore false triggering due to mouse selection for zoom
    if( lastHoverBar.postZoom ) {
      lastHoverBar.postZoom = false;
      numClicks = 0;
      return;
    }
    // defer click event to enable dblclick catch
    clickTimer = setTimeout( function() {
      if( cursorOverItem(pos,item) ) {
        showTooltip(item,pos,true);
        lastHoverBar.clickItem = item;
        if( item ) plotObj.highlight(item.series,item.datapoint);
      } else {
        hideTooltip();
      }
      numClicks = 0;
    }, 250 );
  });

  placeholder.bind("mouseleave", function() {
    setCursor('default');
  });

  function cursorOverPlot(x,y) {
    return plotStats.numPlots > 0 && x >= plotStats.minX && x <= plotStats.maxX && y >= plotStats.minY && y <= plotStats.maxY;
  }

  function cursorOverItem(pos,item) {
    if( pos.x >= plotStats.numPoints ) return false;
    return item || Math.abs(pos.y) < plotStats.tooltipZero;
  }

  function hideTooltip() {
    if( plotObj ) plotObj.unhighlight();
    $("#TC-tooltip").hide();
    $('#TC-tooltip-controls').hide();
    lastHoverBar.binNum = -1;
    lastHoverBar.clickItem = null;
    lastHoverBar.sticky = false;
    lastHoverBar.label = '';
  }

  function showTooltip(item,pos,sticky) {
    renderTooltip( item, Math.floor(pos.x), (pos.y < 0 ? -1 : 1), sticky, pos.pageX, pos.pageY );
  }

  // create tooptip relative to plot area - Flot object item may be NULL
  function renderTooltip(item,binNum,dir,sticky,pageX,pageY) {
    // clear previous non-sticky toolip timeout - just in case
    if( timeout != null ) {
      clearTimeout(timeout);
      timeout = null;
    }
    // get label for checking need to resolve selection issues
    // - if in not in Strand Reads view, dir < 0 => reverse, dir > 0 => forward, dir == 0 => all
    var isRev = (dir < 0);
    var forceReportAll = (dir == 0 && plotParams.barAxis == 1);
    var label, bgColor;
    if( overlayPoint(item) ) {
      label = item.series.label;
      bgColor = item.series.color;
    } else if( plotParams.barAxis == 0 || forceReportAll ) {
      label = baseCoverage ? LegendLabels.allBaseReads : LegendLabels.allReads;
      bgColor = ColorSet.allReads;
    } else if( plotParams.barAxis == 1 ) {
      if( baseCoverage ) {
        label = isRev ? LegendLabels.revBaseReads : LegendLabels.fwdBaseReads;
      } else {
        label = isRev ? LegendLabels.revReads : LegendLabels.fwdReads;
      }
      bgColor = isRev ? ColorSet.revReads : ColorSet.fwdReads;
    } else {
      label = baseCoverage ? LegendLabels.allBaseReads : LegendLabels.allReads;
      bgColor = ColorSet.covDepth_1x;
    }
    // resolve for overlapping points selection - override supplied binNum for points selection
    var clickBar = dataBar(label);
    if( item && !clickBar ) binNum = item.dataIndex;
    if( binNum < 0 ) binNum = 0;
    if( binNum >= plotStats.numPoints ) binNum = plotStats.numPoints-1;
    // do not output the current message (e.g. if move cursor)
    if( lastHoverBar.binNum == binNum && lastHoverBar.sticky == sticky &&
        lastHoverBar.isRev == isRev && lastHoverBar.label == label ) return;
    hideTooltip();
    lastHoverBar.binNum = binNum;
    lastHoverBar.isRev = isRev;
    lastHoverBar.sticky = sticky;
    lastHoverBar.label = label;
    // generate text message content
    $('#TC-tooltip-body').html( sticky ? tooltipMessage(label,binNum) : tooltipHint(label,binNum) );
    // position and display message box
    var whiteText = (label === LegendLabels.targLen || label === LegendLabels.fwdBias || label === LegendLabels.gcBias || label === LegendLabels.percentGC);
    var posx = pageX+10;
    var posy = pageY-10;
    var minTipWidth = 0;
    if( sticky ) {
      if( clickBar ) minTipWidth = plotParams.barAxis ? 230 : 210;
      var cof = $('#TC-chart').offset();
      var ht = $('#TC-tooltip').height();
      var ymax = cof.top + $('#TC-chart').height() - ht;
      posy = pageY - $('#TC-tooltip').height()/2;
      if( posy > ymax ) posy = ymax;
      if( posy < cof.top-4 ) posy = cof.top-4;
      var xmid = cof.left + $('#TC-chart').width()/2;
      if( pageX > xmid ) posx = pageX - $('#TC-tooltip').width() - 26;
    }
    $('#TC-tooltip').css({
      position: 'absolute', left: posx, top: posy, minWidth: minTipWidth,
      background: bgColor, padding: '3px '+(sticky ? '7px' : '4px'),
      color: whiteText ? "white" : "black",
      border: (sticky ? 2 : 1)+'px solid #444',
      opacity: sticky ? 1: 0.7
    }).appendTo("body").fadeIn(sticky ? 10 : 100);
    if( !sticky ) {
      timeout = setTimeout( function() { hideTooltip(); }, 200 );
    }
  }

  $('#TC-tooltip-close').click( function() {
    hideTooltip();
  });

  $('#TC-tooltip-zoomout').click( function() {
    // NOTE: zoom-out from filtered target not possible, as neighbors not loaded/predictable
    if( zoomViewOnBin( lastHoverBar.binNum, false ) ) hideTooltip();
  });

  $('#TC-tooltip-center').click( function() {
    // NOTE: zoom-out from filtered target not possible, as neighbors not loaded/predictable
    if( centerViewOnBin( lastHoverBar.binNum ) ) hideTooltip();
  });

  $('#TC-tooltip-zoomin').click( function() {
    if( zoomViewOnBin( lastHoverBar.binNum, true ) ) hideTooltip();
  });

  function binBar(bin) {
    return plotStats.binnedData && dataTable[bin][DataField.target_id] === '0';
  }

  function dataBar(id) {
    if( baseCoverage ) {
      return (id === LegendLabels.fwdBaseReads || id === LegendLabels.revBaseReads || id === LegendLabels.allBaseReads);
    }
    return (id === LegendLabels.fwdReads || id === LegendLabels.revReads || id === LegendLabels.allReads);
  }

  function overlayPoint(item) {
    if( item == undefined || item == null ) return false;
    var id = item.series.label;
    return (id === LegendLabels.percentGC || id === LegendLabels.targLen ||
            id === LegendLabels.fwdBias || id === LegendLabels.gcBias);
  }

  function tooltipHint(id,bin) {
    $('#TC-tooltip-close').hide();
    $('#TC-tooltip-zoomout').hide();
    $('#TC-tooltip-center').hide();
    $('#TC-tooltip-zoomin').hide();
    if( dataBar(id) ) {
      var br = "<br/>";
      if( binBar(bin) ) {
        return dataTable[bin][DataField.bin_size] + ' ' + LegendLabels.targType + 's';
      }
      var i, targID = dataTable[bin][DataField.target_id];
      while( (i = targID.indexOf(',')) > 0 ) {
        targID = targID.substr(0,i)+br+targID.substr(i+1);
      }
      return targID;
    }
    var targLen = binBar(bin) ? dataTable[bin][DataField.bin_length] :
      dataTable[bin][DataField.pos_end]-dataTable[bin][DataField.pos_start]+1;
    if( id === LegendLabels.percentGC ) {
      return (targLen > 0 ? sigfig(100 * dataTable[bin][DataField.target_gc] / targLen) : 0)+'%';
    } else if( id === LegendLabels.targLen ) {
      return commify(binBar(bin) ? sigfig(targLen / dataTable[bin][DataField.bin_size]) : targLen);
    } else if( id === LegendLabels.fwdBias ) {
      var fwdReads = dataTable[bin][DataField.reads_fwd];
      var totalReads = fwdReads + dataTable[bin][DataField.reads_rev];
      return sigfig(totalReads > 0 ? 100 * fwdReads / totalReads : 50)+'%';
    } else if( id === LegendLabels.gcBias ) {
      var gcbias = 100 * (binBar(bin) ? dataTable[bin][DataField.sum_gcbias]/dataTable[bin][DataField.bin_size]
                                      : Math.abs(dataTable[bin][DataField.target_gc]/targLen - 0.5) );
      return gcbias.toFixed(1)+'%';
    }
    return '?';
  }

  function tooltipMessage(id,bin) {
    $('#TC-tooltip-close').show();
    $('#TC-tooltip-zoomout').toggle(plotStats.targetsSelected > plotStats.targetsRepresented);
    $('#TC-tooltip-center').toggle(plotStats.targetsSelected > plotStats.targetsRepresented);
    $('#TC-tooltip-zoomin').toggle(plotStats.targetsRepresented > plotParams.numPoints);
    var targLen = binBar(bin) ? dataTable[bin][DataField.bin_length] :
      dataTable[bin][DataField.pos_end]-dataTable[bin][DataField.pos_start]+1;
    var lenCov = dataTable[bin][DataField.cov_length];  // or number of read overlaps with target
    var fwdReads = dataTable[bin][DataField.reads_fwd]; // base or assigned reads
    var revReads = dataTable[bin][DataField.reads_rev]; // base or assigned reads
    var uc3p = dataTable[bin][DataField.uncov_3p]; // or number of fwd e2e
    var uc5p = dataTable[bin][DataField.uncov_5p]; // or number of rev e2e
    // derived data
    var totalReads = fwdReads + revReads;
    var pcFwd = totalReads > 0 ? 100 * fwdReads / totalReads : 50;
    var pcCov = targLen > 0 ? sigfig(100 * lenCov / targLen) : 0;
    var pcGC  = targLen > 0 ? sigfig(100 * dataTable[bin][DataField.target_gc] / targLen) : 0;
    var pcU3p, pcU5p; // % uncovered end lengths or % end-to-end reads
    if( baseCoverage ) {
      pcU3p = targLen > 0 ? sigfig(100 * uc3p / targLen) : 0;
      pcU5p = targLen > 0 ? sigfig(100 * uc5p / targLen) : 0;
    } else {
      // note reverse of field values: uc5p == fwd_e2e
      pcU3p = fwdReads > 0 ? sigfig(100 * uc5p / fwdReads) : 0;
      pcU5p = revReads > 0 ? sigfig(100 * uc3p / revReads) : 0;
    }
    var dir = '';
    var barData = dataBar(id);
    if( id == LegendLabels.fwdReads || id == LegendLabels.fwdBaseReads ) {
      dir = "forward ";
      totalReads = fwdReads;
    } else if( id == LegendLabels.revReads || id == LegendLabels.revBaseReads ) {
      dir = "reverse ";
      totalReads = revReads;
    }
    // customize message fields
    var leadStr = binBar(bin) ? "Total "+LegendLabels.targType.toLowerCase() : LegendLabels.targType;
    var readStr = dir+LegendLabels.rcovOrig.toLowerCase()+': ';
    var meanStr = LegendLabels.meanType+dir+LegendLabels.rcovType.toLowerCase()+": ";
    var e2eStr = (reportPassingCov ? "Passing coverage " : "End-to-end ");
    // create 'average' - dependent on bar and axis
    var meanReads;
    if( lengthNormal ) {
      meanReads = targLen > 0 ? (totalReads / targLen) : 0;
    } else {
      meanReads = binBar(bin) ? totalReads/dataTable[bin][DataField.bin_size] : totalReads;
    }
    // compose message
    var br = "<br/>";
    var msg = "(Bin#"+(bin+1)+")"+br;
    var contigType = amplicons == 3 ? "Contig: " : (transcriptBed ? "Transcript: " : "Choromosome: ");
    if( binBar(bin) && dataTable[bin][DataField.bin_size] > 1 ) {
      var nbins = dataTable[bin][DataField.bin_size];
      msg += LegendLabels.targType+"s represented: "+nbins+br;
      msg += contigType+dataTable[bin][DataField.contig_id]+br;
      if( barData ) {
        msg += "Average "+LegendLabels.targType.toLowerCase()+" length: "+commify(sigfig(targLen/nbins))+br;
      }
    } else {
      var i, targID = dataTable[bin][DataField.target_id];
      while( (i = targID.indexOf(',')) > 0 ) {
        targID = targID.substr(0,i)+br+'+  '+targID.substr(i+1);
      }
      msg += leadStr+" ID: "+targID+br;
      if( amplicons < 3 ) {
        msg += contigType+dataTable[bin][DataField.contig_id]+br;
        msg += "Location: "+commify(dataTable[bin][DataField.pos_start])+"-"+commify(dataTable[bin][DataField.pos_end])+br;
      }
      gstr = dataTable[bin][DataField.gene_id]
      if( gstr != '' && gstr != '.' ) {
        msg += (gstr.indexOf('=') > 0 ? "Attributes" : "Gene Sym")+": ";
        if( gstr.length > 50 ) gstr = gstr.substring(0,50)+"...";
        msg += gstr+br;
      }
      if( barData ) {
        msg += LegendLabels.targType+" length: "+commify(targLen)+br;
      }
    }
    if( barData || id === LegendLabels.percentGC || id === LegendLabels.gcBias )
      msg += leadStr+" GC content: "+pcGC+"%"+br;
    if( barData ) {
      if( baseCoverage ) {
        // moved to below for new reports
        if( !enableReadDepthPlot ) {
          msg += leadStr+" length covered: "+pcCov+"%"+br;
        }
      } else {
        msg += leadStr+" overlapping reads: "+commify(lenCov)+br;
      }
      msg += leadStr+" "+readStr+commify(totalReads)+br;
      if( lengthNormal || binBar(bin) ) {
        msg += meanStr+commify(sigfig(meanReads))+br;
      }
    }
    if( id === LegendLabels.fwdReads || id === LegendLabels.fwdBaseReads ) {
      msg += "Percent "+readStr+sigfig(pcFwd)+'%'+br;
      if( baseCoverage ) {
        msg += "Uncovered target 5' length: "+sigfig(pcU5p)+"%"+br;
        msg += "Uncovered target 3' length: "+sigfig(pcU3p)+"%"+br;
      } else {
        msg += e2eStr+readStr+sigfig(pcU3p)+"%"+br;
      }
    } else if( id === LegendLabels.revReads || id === LegendLabels.revBaseReads ) {
      msg += "Percent "+readStr+sigfig(100-pcFwd)+'%'+br;
      if( baseCoverage ) {
        msg += "Uncovered target 5' length: "+sigfig(pcU5p)+"%"+br;
        msg += "Uncovered target 3' length: "+sigfig(pcU3p)+"%"+br;
      } else {
        msg += e2eStr+readStr+sigfig(pcU5p)+"%"+br;
      }
    } else if( id === LegendLabels.allReads || id === LegendLabels.allBaseReads ) {
      msg += "Percent forward "+readStr+sigfig(pcFwd)+'%'+br;
      if( baseCoverage ) {
        msg += "Uncovered target 5' length: "+sigfig(pcU5p)+"%"+br;
        msg += "Uncovered target 3' length: "+sigfig(pcU3p)+"%"+br;
      } else {
        pcU3p = totalReads > 0 ? sigfig(100 * (uc3p+uc5p) / totalReads) : 0;
        msg += e2eStr+readStr+sigfig(pcU3p)+"%"+br;
      }
    } else if( id === LegendLabels.targLen ) {
      if( binBar(bin) ) {
        var aveTargLen = targLen / dataTable[bin][DataField.bin_size];
        msg += "Average "+LegendLabels.targType.toLowerCase()+" length: "+commify(sigfig(aveTargLen))+br;
      } else {
        msg += LegendLabels.targType+" length: "+commify(targLen)+br;
      }
    } else if( id === LegendLabels.fwdBias ) {
      msg += "Percent forward "+readStr+sigfig(pcFwd)+'%'+br;
      msg += pcFwd >= 50 ? "Forward" : "Reverse";
      pcFwd = Math.abs(2*pcFwd-100);
      msg += " bias: "+sigfig(pcFwd)+'%'+br;
    } else if( id === LegendLabels.gcBias || id === LegendLabels.percentGC ) {
      var gcbias = 100 * (binBar(bin) ? dataTable[bin][DataField.sum_gcbias]/dataTable[bin][DataField.bin_size]
                                      : Math.abs(dataTable[bin][DataField.target_gc]/targLen - 0.5) );
      if( binBar(bin) ) msg += "Average ";
      msg += "GC or AT bias: "+gcbias.toFixed(1)+"%"+br;
    }
    if( enableReadDepthPlot && barData ) {
      if( baseCoverage ) {
        msg += "Target base coverage at 1x: "+sigfig(pcCov)+"%"+br;
      }
      msg += "Target base coverage at 20x: "+sigfig(100*dataTable[bin][DataField.cov_20x]/targLen)+'%'+br;
      msg += "Target base coverage at 100x: "+sigfig(100*dataTable[bin][DataField.cov_100x]/targLen)+'%'+br;
      msg += "Target base coverage at 500x: "+sigfig(100*dataTable[bin][DataField.cov_500x]/targLen)+'%'+br;
    }
    // show tooltip controls if appropriate
    if( barData && !binBar(bin) ) {
      $('#TC-tooltip-controls').show();
    } else {
      $('#TC-tooltip-controls').hide();
    }
    return msg;
  }

  $('#TC-OpenInIGV').click( function() {
    var bin = lastHoverBar.binNum;
    if( bin < 0 ) return;
    window.open( linkIGV( getDisplayRegion(bin,50) ) );
  });

  // grab update request from external source (to TC)
  $('#TC-ViewRequest').change(function() {
    // ignore request if widget currently disabled
    if( disableTitleBar ) return;
    // force chart into view when request recieved - circumvents display issues if newly ploted while collapsed
    if( !$('#TC-plotspace').is(":visible") ) {
      $("#TC-collapsePlot").click();
    }
    if( $('#TC-filterGeneSym').val() == this.value ) {
      // if already in focus just force tooltip to be shown
      renderTooltip( null, 0, 0, true, 203, 1195 );
    } else {
      // disable sending back focus to RC Chart
      autoJumpToGene = false;
      $('#TC-filterGeneSym').val(this.value).change();
      autoJumpToGene = plotParams.autoJumpToGene;
    }
  });

  function getDisplayRegion(bin,viewBuffer,binEnd) {
    if( bin < 0 ) return '';
    if( binEnd == undefined || binEnd == null ) binEnd = bin;
    var chr = dataTable[bin][DataField.contig_id];
    var srt = dataTable[bin][DataField.pos_start];
    var end = dataTable[binEnd][DataField.pos_end];
    if( srt <= 0 ) return chr;
    if( end <= 0 ) return chr+':'+end;
    srt -= viewBuffer;
    if( srt < 1 ) srt = 1;
    end += viewBuffer;
    if( end > plotStats.chromLength ) end = plotStats.chromLength;
    return chr+':'+srt+'-'+end;
  }

  function linkIGV(region) {
    var locpath = window.location.pathname.substring(0,window.location.pathname.lastIndexOf('/'));
    var igvURL = window.location.protocol + "//" + window.location.host + "/auth" + locpath + "/igv.php3";
    var launchURL = window.location.protocol + "//" + window.location.host + "/IgvServlet/igv";
    return launchURL + "?locus="+region+"&sessionURL="+igvURL;
  }

  $('#TC-OpenInRCC').click( function() {
    var bin = lastHoverBar.binNum;
    if( bin < 0 ) return;
    var region = dataTable[bin][DataField.contig_id]+':'+dataTable[bin][DataField.pos_start]+'-'+dataTable[bin][DataField.pos_end];
    $("#RC-ViewRequest").val( getDisplayRegion(bin,50) ).change();
  });

  function sigfig(val) {
    val = parseFloat(val);
    var av = Math.abs(val);
    if( av == parseInt(av) ) return val.toFixed(0);
    if( av >= 100 ) return val.toFixed(0);
    if( av >= 10 ) return val.toFixed(1);
    if( av >= 1 ) return val.toFixed(2);
    if( av >= 0.1 ) return val.toFixed(3);
    return val.toFixed(3);
  }

  function commify(val) {
    var jrs = val.toString();
    var dps = "";
    var i = jrs.indexOf('.');
    if( i >= 0 ) {
      dps = jrs.substring(i);
      jrs = jrs.substring(0,i);
    }
    return jrs.replace(/(\d)(?=(\d\d\d)+(?!\d))/g, "$1,")+dps;
  }

  function setCursor(curs) {
    if( curs == null || curs == "" )
       curs = 'default';
    if( useFlash && canvas != null ) {
      // prevents tooltip from appearing!
      //FlashCanvas.setCursor(canvas,curs);
    } else {
      document.body.style.cursor = curs;
    }
  }

  // --------- Set up charting controls outside of plot area ---------

  $('#TC-snapShot').click(function() {
    if( canvas != null ) {
      canvas2png(canvas);
    }
  });

  function updateGUIPlotParams() {
    $('.TC-selectParam#barAxis').val(plotParams.barAxis);
    $('.TC-selectParam#overPlot').val(plotParams.overPlot);
    $('#TC-logAxis').attr('checked',(plotParams.logAxis));
    $('#TC-autoZoom').attr('checked',(plotParams.zoomMode == 1));
    $('#TC-showLegend').attr('checked',plotParams.showLegend);
    $('#TC-selectChrom').val(tsvFilter.chrom);
    $('#TC-filterGeneSym').val(tsvFilter.gene);
    $('#TC-filterCovMin').val(tsvFilter.covmin);
    $('#TC-filterCovMax').val(tsvFilter.covmax < def_hugeCov ? tsvFilter.covmax : '');
  }

  $('#TC-clearFilters').click(function() {
    if( clearFilters(true) ) unzoomData();
  });

  function clearFilters(allFilters) {
    // only reset if any filter is not at its default
    // allFilters forces all to be reset, otherwise cumulative filters are not changed
    var numReset = 0;
    if( allFilters ) {
      if( tsvFilter.covmin != 0 ) {
        $('#TC-filterCovMin').val(tsvFilter.covmin = 0);
        ++numReset;
      }
      if( $('#TC-filterCovMax').val() != '' ) {
        $('#TC-filterCovMax').val('');
        tsvFilter.covmax = def_hugeCov;
        ++numReset;
      }
      if( tsvFilter.chrom != '' ) {
        $('#TC-selectChrom').val('ALL');
        tsvFilter.chrom = '';
        ++numReset;
      }
    }
    if( tsvFilter.gene != '' ) {
      $('#TC-filterGeneSym').val(tsvFilter.gene = '');
      ++numReset;
    }
    tsvFilter.numrec = 0;
    return numReset;
  }

  $('#TC-autoZoom').change(function() {
    plotParams.zoomMode = ($(this).attr("checked") == "checked") ? 1 : 0;
    updatePlot();
  });

  $('#TC-numPoints').change(function() {
    var val = this.value.trim();
    val = (val == '') ? plotStats.defNumPoints : Math.floor(val);
    if( isNaN(val) ) {
      val = plotParams.numPoints;
    } else if( val < plotStats.minNumPoints ) {
      val = plotStats.minNumPoints;
    } else if( val > plotStats.maxNumPoints ) {
      val = plotStats.maxNumPoints;
    }
    this.value = val;
    if( val != plotParams.numPoints ) {
      plotParams.numPoints = val;
      unzoomData();
    }
  });
    
  $('#TC-filterCovMin').change(function() {
    var val = this.value.trim();
    if( val == '' || isNaN(val) || val <= 0 ) val = 0;
    this.value = val;
    if( val > tsvFilter.covmax ) {
      $('#TC-filterCovMax').val('');
      tsvFilter.covmax = def_hugeCov;
    }
    if( val != tsvFilter.covmin ) {
      clearFilters(false);
      tsvFilter.covmin = val;
      unzoomData();
    }
  });
    
  $('#TC-filterCovMax').change(function() {
    var val = this.value.trim();
    if( val == '' || isNaN(val) ) {
      val = def_hugeCov;
      this.value = '';
    } else {
      if( val <= 0 ) val = 0;
      this.value = val;
      if( val < tsvFilter.covmin ) {
        $('#TC-filterCovMin').val('0');
        tsvFilter.covmin = 0;
      }
    }
    if( val != tsvFilter.covmax ) {
      clearFilters(false);
      tsvFilter.covmax = val;
      unzoomData();
    }
  });

  $('#TC-selectChrom').change(function() {
    var val = this.value;
    if( val == 'ALL' ) val = '';
    if( val != tsvFilter.chrom ) {
      clearFilters(false);
      tsvFilter.chrom = val;
      this.value = val ? val : 'ALL';
      unzoomData();
      if( contigChart && val ) {
        renderTooltip( null, 0, 0, true, 203, 1195 );
      } 
    }
  });

  $('#TC-filterGeneSym').change(function() {
    var val = this.value.trim();
    if( val != tsvFilter.gene ) {
      clearFilters(true);
      tsvFilter.gene = val;
      this.value = val;
      unzoomData();
      if( autoJumpToGene && val != '' ) {
        viewTargetsInRRC();
      }
      if( plotStats.numPoints == 1 ) {
        renderTooltip( null, 0, 0, true, 203, 1195 );
        if( contigChart ) {
          $('#TC-selectChrom').val(val);
          tsvFilter.chrom = val;
        }
      }
    }
  });

  // ok to call so long as all bins loaded are on same contig, e.g. same gene
  function viewTargetsInRRC() {
    if( plotStats.numPoints <= 0 ) return;
    var srtBin = 0, minPos = -1
    var endBin = 0, maxPos = 0;
    for( var i = 0; i < plotStats.numPoints; ++i ) {
      // ignore binned data incase more than [100] targets
      if( binBar(i) ) continue;
      if( dataTable[i][DataField.pos_start] < minPos || minPos < 0 ) {
        minPos = dataTable[i][DataField.pos_start];
        srtBin = i;
      }
      if( dataTable[i][DataField.pos_end] > maxPos ) {
        maxPos = dataTable[i][DataField.pos_end];
        endBin = i;
      }
    }
    if( minPos < 0 ) return;
    var buf = Math.floor((maxPos-minPos)/2000)*100;
    if( buf < 50 ) buf = 50;
    $("#RC-ViewRequest").val( getDisplayRegion(srtBin,buf,endBin) ).change();
  }

  $('.TC-selectParam').change(function() {
    plotParams[this.id] = parseInt(this.value);
    plotParams.resetYScale = this.id == 'barAxis';
    autoShowLegend();
    updatePlot();
  });

  function autoShowLegend() {
    if( autolegend && !plotParams.showLegend ) {
      $('#TC-showLegend').attr('checked', plotParams.showLegend = true );
    }
  }

  $('#TC-logAxis').change(function() {
    plotParams.logAxis = ($(this).attr("checked") == "checked");
    updatePlot();
  });

  $('#TC-showLegend').change(function() {
    plotParams.showLegend = ($(this).attr("checked") == "checked");
    updatePlot();
  });

  $("#TC-unzoomToggle").click(function() {
    // remove specific Gene/Attribute to allow zoom out
    if( tsvFilter.gene != '' ) {
      $('#TC-filterGeneSym').val(tsvFilter.gene = '');
      tsvFilter.numrec = 0;
      plotStats.zoom = true;
    }
    if( plotStats.zoom ) unzoomData();
  });

  function unzoomToFile(filename) {
    tsvFilter.dataFile = filename;
    plotStats.minNumPoints = def_minPoints;
    plotParams.numPoints = plotStats.defNumPoints = def_numPoints;
    plotParams.zoomMode = 1;
    $('#TC-numPoints').val(plotParams.numPoints);
    plotStats.chrList = '';
    tsvFilter.numrec = -1; // flags fetch of chromosome list for selection
    unzoomData(true); // capture data bounds
    plotStats.targetsTotal = plotStats.targetsSelected;
    // check for small data sets or allow initfile to define the number of bins
    plotParams.numPoints = plotStats.numPoints;
    $('#TC-numPoints').val(plotStats.numPoints);
    if( tsvOverride != '' ) {
       plotStats.defNumPoints = plotStats.numPoints;
    }
    tsvOverride = '';
    setChromSearch();
  }

  function setChromSearch() {
    var selObj = $('#TC-selectChrom');
    selObj.empty();
    selObj.css('width','66px');
    if( plotStats.chrList == '' ) return;
    // allow ':' in contig ids as separator, unless '&' is present (the later being new and the old for backwards compat)
    var splitChr = plotStats.chrList.indexOf('&') > 0 ? '&' : ':';
    var chrs = plotStats.chrList.split(splitChr);
    if( chrs.length > 1 ) {
      selObj.append("<option value='ALL'>ALL</option>");
    }
    var mclen = 0;
    for( var i = 0; i < chrs.length-1; ++i ) {
      selObj.append("<option value='"+chrs[i]+"'>"+chrs[i]+"</option>");
      if( chrs[i].length > mclen ) mclen = chrs[i].length;
    }
    if( mclen > 6 ) selObj.css('width','');
  }

  function loadData() {
    tsvFilter.maxrows = plotParams.numPoints;
    loadTSV();
    updatePlotStats();
  }

  function zoomData() {
    loadData();
    updatePlot();
  }

  function unzoomData() {
    tsvFilter.clipleft = 0;
    tsvFilter.clipright = 100;
    plotStats.zoom = false;
    if( tsvFilter.dataFile == '' ) return;
    loadData();
    updatePlot(true); // capture data bounds
  }

  // load data using PHP to dataTable[] using options in tsvFilter{}
  function loadTSV() {
    var src = (tsvOverride != '') ? tsvOverride : 'lifechart/target_coverage.php3';
    dataTable = [];
    $('#TC-message').text('Loading...');
    $.ajaxSetup( {dataType:"text",async:false} );
    $.get(src, tsvFilter, function(mem) {
      var lines = mem.split("\n");
      $.each(lines, function(n,row) {
        var fields = $.trim(row).split('\t');
        if( n == 0 ) {
          fieldIds = fields;
          if( fields[0].substr(0,5).toLowerCase() == 'error' ) alert(row);
          plotStats.targetsSelected = Math.floor(fieldIds.shift());
          if( tsvFilter.numrec < 0 ) {
            plotStats.chrList = fieldIds.shift();
          }
        } else if( fields[0] != "" ) {
          // important to convert numeric fields to numbers for performance
          fields[1] = +fields[1];
          fields[2] = +fields[2];
          for( var i = 5; i < fields.length; ++i ) { fields[i] = +fields[i]; }
          dataTable.push( fields );
        }
      });
    }).error(function(){
      alert("An error occurred while loading from "+(tsvOverride != '' ? tsvOverride : tsvFilter.dataFile));
      $('#TC-message').text('');
    }).success(function(){
      $('#TC-message').text('');
    });
  }

  function updatePlotStats() {
    plotStats.numPoints = dataTable.length;
    plotStats.numFields = fieldIds.length;
    plotStats.minX = 0;
    plotStats.maxX = plotStats.numPoints;

    var numRep = 0.01 * plotStats.targetsSelected * (tsvFilter.clipright - tsvFilter.clipleft);
    plotStats.targetBinSize = numRep / tsvFilter.maxrows;
    plotStats.targetsRepresented = parseInt(numRep+0.5);
    plotStats.binnedData = (plotStats.targetBinSize > 1.0000001);
    if( !plotStats.binnedData ) plotStats.targetBinSize = 1;

    // set the maximum zoom level to ensure number of points loaded
    plotStats.zoomMinLoad = 100 * plotStats.numPoints / plotStats.targetsSelected;

    // detect new (filter) load - set static values and speedup subsequent loads
    if( tsvFilter.numrec == 0 ) {
      tsvFilter.numrec = plotStats.targetsSelected;
    }

    // check for small (filtered) data sets
    plotStats.minNumPoints = plotStats.numPoints < def_minPoints ? plotStats.numPoints : def_minPoints;
    if( plotParams.numPoints != plotStats.numPoints ) {
      $('#TC-numBars').text('('+plotStats.numPoints+')');
    } else {
      $('#TC-numBars').text('');
    }

    // enable Read Depth plot if the extra fields are available
    enableReadDepthPlot = (plotStats.numFields > 12);
    customizePlotOptions();
  }

  function roundAxis( maxVal ) {
    if( maxVal == 0 ) return 0;
    var sgn = maxVal < 0 ? -1 : 1;
    maxVal = Math.abs(maxVal);
    var b = Math.pow( 10, Math.round(Math.log(maxVal)/Math.LN10)-1 );
    return sgn * b * Math.floor(1+maxVal/b);
  }

  function percentFormat(val, axis) {
    return ''+val.toFixed(axis.tickDecimals)+'%';
  }

  function sigfigFormat(val, axis) {
    return sigfig(val);
  }

  function absFormat(val, axis) {
    return ''+Math.abs(val.toFixed(axis.tickDecimals));
  }

  function log10(val) {
    return Math.log(val+1)/Math.LN10;
  }

  function updatePlot(captureScale) {
    if( captureScale === undefined || captureScale === null ) captureScale = false;
    plotData = [];
    if( plotStats.numFields <= 1 ) {
      return;
    }
    var nBar = plotStats.targetsRepresented > tsvFilter.maxrows ? tsvFilter.maxrows : plotStats.targetsRepresented;
    var xLabel = commify(plotStats.targetsRepresented) + (plotStats.binnedData ? " Binned" : " Individual")
      + " " + LegendLabels.targType + "s of " + commify(plotStats.targetsSelected) + " Selected";
    options = {
      grid: {minBorderMargin:0, hoverable:true, clickable:true, backgroundColor:"#F8F8F8"},
      selection: {mode:plotParams.zoomMode == 2 ? "xy" : "x"},
      legend: {position:plotParams.barAxis == 0 ? "nw" : "sw"},
      series: {axis:1, bars:{show:true,align:"left"}, line:{show:false}},
      xaxis: {ticks:0, axisLabel:xLabel, axisLabelFontSizePixels:18, min:plotStats.minX, max:plotStats.maxX },
      yaxis: {tickFormatter:absFormat, axisLabelFontSizePixels:16},
      xaxes: {}, yaxes: []
    };
    var nplot = 0;
    var d1 = [];
    var d2 = [];
    var d3 = [];
    var d4 = [];
    var ymin = 0, ymax = 0;
    var dmin = 0, dmax = 0;
    var binSiz = DataField.bin_size;
    var binLen = DataField.bin_length;
    var posSrt = DataField.pos_start;
    var posEnd = DataField.pos_end;
    var uncov3p  = DataField.uncov_3p; // or rev_e2e
    var uncov5p  = DataField.uncov_5p; // or fwd_e2e
    var fwdReads = DataField.reads_fwd; // base reads or assigned reads
    var revReads = DataField.reads_rev; // base reads or assigned reads
    var cov20x  = DataField.cov_20x;
    var cov100x = DataField.cov_100x;
    var cov500x = DataField.cov_500x;
    var logAxis = plotParams.logAxis;
    var barScale;
    for( var i = 0; i < plotStats.numPoints; ++i ) {
      var numReads = dataTable[i][fwdReads]+dataTable[i][revReads];
      var targLen = binBar(i) ? dataTable[i][binLen] : dataTable[i][posEnd]-dataTable[i][posSrt]+1;
      var ucov3 = dataTable[i][uncov3p];
      var ucov5 = dataTable[i][uncov5p];
      if( lengthNormal ) {
        barScale = targLen > 0 ? 1/targLen : 0;
      } else {
        barScale = binBar(i) ? 1/dataTable[i][binSiz] : 1;
      }
      var axisScale = plotParams.aveBase ? barScale : 1;
      var frds = dataTable[i][fwdReads] * axisScale;
      var rrds = dataTable[i][revReads] * axisScale;
      var trds = frds + rrds;
      // always track the plotted directional max/min for setting post call
      if( plotStats.minX <= i && i <= plotStats.maxX ) {
        if( frds > dmax ) dmax = frds;
        if( -rrds < dmin ) dmin = -rrds;
        if( trds > ymax ) ymax = trds;
      }
      var fcov =  (logAxis ? log10(frds) : frds);
      var rcov = -(logAxis ? log10(rrds) : rrds);
      var ncov = logAxis ? log10(trds) : trds;
      if( plotParams.barAxis == 0 ) {
        if( baseCoverage ) {
          ucov5 *= ncov * barScale;
          ucov3 = ncov * (1 - ucov3 * barScale);
        } else {
          // fractional passing coverage shows as ratio (not log)
          ucov5 = numReads == 0 ? 0 : ncov * (ucov3+ucov5)/numReads;
        }
        if( baseCoverage ) {
          d1.push( [i,ucov3] );
          d2.push( [i,ncov] );
          d3.push( [i,ucov5] );
        } else {
          d1.push( [i,ncov] );
          d2.push( [i,ucov5] );
        }
      } else if( plotParams.barAxis == 1 ) {
        if( baseCoverage ) {
          ucov3 = fcov * ucov3 * barScale;
          ucov5 = rcov * ucov5 * barScale;
          //ucov3 = fcov * (1 - ucov3 * barScale);
          //ucov5 = rcov * (1 - ucov5 * barScale);
        } else {
          // fractional passing coverage shows as ratio (not log)
          // Note: the order of the data fields is opposite to that for base coverage
          f_e2e = ucov5;
          r_e2e = ucov3;
          ucov3 = dataTable[i][fwdReads] == 0 ? 0 : fcov * f_e2e/dataTable[i][fwdReads];
          ucov5 = dataTable[i][revReads] == 0 ? 0 : rcov * r_e2e/dataTable[i][revReads];
        }
        d1.push( [i,fcov] );
        d2.push( [i,ucov3] );
        d3.push( [i,rcov] );
        d4.push( [i,ucov5] );
      } else {
        // base coveage stats always normalized by total target length
        var lenScale = targLen > 0 ? 1/targLen : 0;
        d1.push( [i,ncov] );
        d2.push( [i,ncov*dataTable[i][cov20x]*lenScale] );
        d3.push( [i,ncov*dataTable[i][cov100x]*lenScale] );
        d4.push( [i,ncov*dataTable[i][cov500x]*lenScale] );
      }
    }
    // collect the range bounds
    if( captureScale ) {
      plotStats.totalMinY = roundAxis(ymin);
      plotStats.totalMaxY = roundAxis(ymax);
      plotStats.strandMinY = roundAxis(dmin);
      plotStats.strandMaxY = roundAxis(dmax);
    }
    // set absolute man/max depending on plot type (i.e. if negative bars are drawn)
    ymin = plotParams.barAxis == 1 ? dmin : ymin;
    ymax = plotParams.barAxis == 1 ? dmax : ymax;
    if( plotStats.zoom && !plotParams.resetYScale ) {
      // always adjust zoom if max/min increase due to re-binning
      if( ymin < plotStats.minY ) plotStats.minY = ymin;
      if( ymax > plotStats.maxY ) plotStats.maxY = ymax;
      if( plotParams.zoomMode == 2 ) {
         ymin = plotStats.minY;
         ymax = plotStats.maxY;
      } else if( plotParams.zoomMode == 0 ) {
         ymin = plotParams.barAxis == 1 ? plotStats.strandMinY : plotStats.totalMinY;
         ymax = plotParams.barAxis == 1 ? plotStats.strandMaxY : plotStats.totalMaxY;
      }
    } else {
      plotParams.resetYScale = false;
      plotStats.minY = ymin;
      plotStats.maxY = ymax;
    }
    var logLeg = logAxis ? "Log " : "";
    if( plotParams.barAxis == 0 ) {
      if( baseCoverage ) {
        plotData.push( { label: LegendLabels.allBaseReads_u3p, color: ColorSet.allReads, data: d1 } );
        plotData.push( { color: ColorSet.allReads_sd2, data: d2 } );
        plotData.push( { label: LegendLabels.allBaseReads_u5p, color: ColorSet.allReads_shd, data: d3 } );
      } else {
        plotData.push( { label: logLeg+LegendLabels.allReads, color: ColorSet.allReads, data: d1 } );
        plotData.push( { label: (reportPassingCov ? LegendLabels.allReads_pss : LegendLabels.allReads_e2e),
          color: ColorSet.allReads_shd, data: d2 } );
      }
    } else if( plotParams.barAxis == 1 ) {
      if( baseCoverage ) {
        plotData.push( { label: logLeg+LegendLabels.fwdBaseReads, color: ColorSet.fwdReads, data: d1 } );
        plotData.push( { label: LegendLabels.fwdBaseReads_u3p, color: ColorSet.fwdReads_shd, data: d2 } );
        plotData.push( { label: logLeg+LegendLabels.revBaseReads, color: ColorSet.revReads, data: d3 } );
        plotData.push( { label: LegendLabels.revBaseReads_u5p, color: ColorSet.revReads_shd, data: d4 } );
      } else {
        plotData.push( { label: logLeg+LegendLabels.fwdReads, color: ColorSet.fwdReads, data: d1 } );
        plotData.push( { label: (reportPassingCov ? LegendLabels.fwdReads_pss : LegendLabels.fwdReads_e2e),
          color: ColorSet.fwdReads_shd, data: d2 } );
        plotData.push( { label: logLeg+LegendLabels.revReads, color: ColorSet.revReads, data: d3 } );
        plotData.push( { label: (reportPassingCov ? LegendLabels.revReads_pss : LegendLabels.revReads_e2e),
          color: ColorSet.revReads_shd, data: d4 } );
      }
    } else {
      plotData.push( { label: LegendLabels.covDepth_1x, color: ColorSet.covDepth_1x, data: d1 } );
      plotData.push( { label: LegendLabels.covDepth_20x, color: ColorSet.covDepth_20x, data: d2 } );
      plotData.push( { label: LegendLabels.covDepth_100x, color: ColorSet.covDepth_100x, data: d3 } );
      plotData.push( { label: LegendLabels.covDepth_500x, color: ColorSet.covDepth_500x, data: d4 } );
    }
    var ytitle = LegendLabels.rcovType;
    // account for limits on log axis and round
    if( logAxis ) {
      ytitle = "log10("+ytitle+")";
      ymin = ymin < 0 ? -log10(-ymin) : log10(ymin);
      ymax = log10(ymax);
    }
    ymin = roundAxis(ymin);
    ymax = roundAxis(ymax);
    var dplace = baseCoverage ? null : 0;
    options.yaxes.push( {position:"left", axisLabel:ytitle, min:ymin, max:ymax, tickDecimals:dplace} );
    options.legend.show = plotParams.showLegend;
    plotStats.tooltipZero = 0.01*(ymax-ymin);
    ++nplot;

    // Add 2nd yaxis plot
    if( plotParams.overPlot > 0 ) {
      var aLabel, pLabel, pColor;
      var d5 = [];
      var dmin = 0;
      var dmax = 100;
      var formatter = percentFormat;
      if( plotParams.overPlot == 1 ) {
        aLabel = "Target GC%";
        pLabel = LegendLabels.percentGC;
        pColor = ColorSet.percentGC;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          var targLen = binBar(i) ? dataTable[i][binLen] : dataTable[i][posEnd]-dataTable[i][posSrt]+1;
          var pcGC = targLen > 0 ? 100 * dataTable[i][DataField.target_gc] / targLen : 0;
          d5.push( [(i+0.5),pcGC] );
        }
      } else if( plotParams.overPlot == 2 ) {
        aLabel = "Target Length";
        pLabel = LegendLabels.targLen;
        pColor = ColorSet.targLen;
        dmax = 0;
        formatter = absFormat;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          var targLen = binBar(i) ? dataTable[i][binLen] / dataTable[i][binSiz] : dataTable[i][posEnd]-dataTable[i][posSrt]+1;
          d5.push( [(i+0.5),targLen] );
          if( targLen > dmax ) dmax = targLen;
          if( targLen < dmin || dmin == 0 ) dmin = targLen;
        }
      } else if( plotParams.overPlot == 3 ) {
        aLabel = "Forward Reads";
        pLabel = LegendLabels.fwdBias;
        pColor = ColorSet.fwdBias;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          var fwdReads = dataTable[i][DataField.reads_fwd];
          var revReads = dataTable[i][DataField.reads_rev];
          var totalReads = fwdReads + revReads;
          var pcFwd = totalReads > 0 ? 100 * fwdReads / totalReads : 50;
          d5.push( [(i+0.5),pcFwd] );
        }
      } else if( plotParams.overPlot == 4 ) {
        aLabel = "Target |GC% - 50%|";
        pLabel = LegendLabels.gcBias;
        pColor = ColorSet.gcBias;
        dmax = 0;
        var targLen, gcbias;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          if( binBar(i) ) {
            gcbias = 100 * dataTable[i][DataField.sum_gcbias] / dataTable[i][DataField.bin_size];
          } else {
            targLen = dataTable[i][posEnd]-dataTable[i][posSrt]+1;
            gcbias = 100 * Math.abs(dataTable[i][DataField.target_gc]/targLen - 0.5);
          }
          d5.push( [(i+0.5),gcbias] );
          if( gcbias > dmax ) dmax = gcbias;
        }
      }
      plotData.push( {
        label: pLabel, color: pColor, data: d5, yaxis: 2, bars: {show:false}, points: {show:true}, shadowSize: 0 } );
      options.yaxes.push( {position:"right", axisLabel:aLabel, min:dmin, max:dmax, tickFormatter: formatter} );
      options.grid.aboveData = true;
      if( plotParams.overPlot == 1 || plotParams.overPlot == 3 ) {
        options.grid.markings = [ {color: pColor, linewidth: 1, y2axis: {from:50,to:50}} ];
      }
      ++nplot;
    }

    plotStats.numPlots = nplot;
    hideTooltip();
    plotObj = $.plot(placeholder, plotData, options);
    canvas = plotObj.getCanvas();
  }

  $('#TC-export').click( function() {
    // check the total number of targets currently in-view
    var numSelected = plotStats.targetsRepresented;
    var targType = LegendLabels.targType.toLowerCase()+'s';
    var $content = $('#TC-dialog-content');
    $content.html('Total number of '+targType+' in view: '+commify(numSelected)+'<br/>');
    if( numSelected == 0 ) {
      $content.append('<p>You must have '+targType+' in view to export.</p>');
      $('#TC-exportOK').hide();
    } else {
      $content.append('<p>\
        <input type="radio" name="TS-exportTool" id="TC-ext1" value="table" checked="checked"/>\
          <label for="TC-ext1">Download as tsv table file.</label><br/>\
        <input type="radio" name="TS-exportTool" id="TC-ext2" value="bed"/>\
          <label for="TC-ext2">Download as a 4-column bed file.</label></p>' );
      $('#TC-exportOK').show();
    }
    // open dialog over masked out table
    var pos = $('#TC-export').offset();
    var x = pos.left+6+($('#TC-export').width()-$('#TC-dialog').width())/2;
    var y = pos.top+$('#TC-export').height()+8;
    $('#TC-dialog').css({ left:x, top:y });
    pos = $('#TC-chart').offset();
    var hgt = $('#TC-chart').height()+4; // extra for borders
    var wid = $('#TC-chart').width()+4;
    $('#TC-mask').css({ left:pos.left, top:pos.top, width:wid, height:hgt });
    $('#TC-mask').show();
    $('#TC-dialog').show();
  });

  $('#TC-exportOK').click(function(e) {
    $('#TC-dialog').hide();
    $('#TC-mask').hide();
    // the following doesn't work when including DOC and RCC code and but not displaying (exiting immediately)
    var op = $("input[@name='TS-exportTool']:checked").val();
    if( op == "table" ) {
      exportTSV("-a");
    } else if( op == "bed" ) {
      exportTSV("-b");
    }
  });

  function exportTSV(expOpt) {
    var clipLeft = tsvFilter.clipleft;
    var clipRight = tsvFilter.clipright;
    var maxrows = plotStats.targetsRepresented;
    if( plotStats.targetsRepresented < plotStats.numPoints ) {
      // temporaily modify file extraction region for over-zoom 
      var lam = plotStats.minX / plotStats.numPoints;
      clipLeft = (1 - lam) * tsvFilter.clipleft + lam * tsvFilter.clipright;
      lam = plotStats.maxX / plotStats.numPoints;
      clipRight = (1 - lam) * tsvFilter.clipleft + lam * tsvFilter.clipright;
    }
    expOpt += ' ' + tsvFilter.options;
    window.open( "lifechart/target_coverage.php3"+
      "?options="+expOpt.trim()+"&dataFile="+tsvFilter.dataFile+
      "&chrom="+tsvFilter.chrom+"&gene="+tsvFilter.gene+
      "&covmin="+tsvFilter.covmin+"&covmax="+tsvFilter.covmax+
      "&clipleft="+clipLeft+"&clipright="+clipRight+
      "&maxrows="+maxrows+"&numrec="+tsvFilter.numrec );
  }

  // autoload - after everything is defined
  if( startHideLegend ) $('#TC-showLegend').attr('checked',plotParams.showLegend = false);
  unzoomToFile(coverageFile);
  if( !startHideLegend ) autoShowLegend();

  // collapse view after EVRYTHING has been drawn in open chart (to avoid flot issues)
  if( startCollapsed ) {
    $("#TC-collapsePlot").attr("class","ui-icon ui-icon-triangle-1-s");
    $("#TC-collapsePlot").attr("title","Expand view");
    $('#TC-controlpanel').hide();
    $('.TC-shy').hide();
    $('#TC-chart').resizable('destroy');
    $('#TC-noncanvas').hide();
  }

});
