// html for chart container and filters bar - note these are invisible and moved into position later
document.write('\
<div id="RC-chart" style="border:2px solid #666;display:none">\
  <div id="RC-titlebar" class="grid-header" style="min-height:24px;border:0">\
    <span id="RC-collapsePlot" style="float:left" class="ui-icon ui-icon-triangle-1-n" title="Collapse View"></span>\
    <span class="table-title" style="float:none">Reference Coverage Chart</span>\
    <span class="RC-shy flyhelp" id="RC-plotLabel" style="padding-left:20px">Plot:</span>\
    <select class="RC-selectParam txtSelect RC-shy" id="barAxis">\
     <option value=0 selected="selected">Total Base Reads</option>\
     <option value=1>Strand Base Reads</option>\
    </select>\
    <span class="RC-shy flyhelp" id="RC-overlayLabel" style="padding-left:10px">Overlay:</span>\
    <select class="RC-selectParam txtSelect RC-shy" id="overPlot">\
     <option value=0 selected="selected">None</option>\
     <option value=1>On-Target</option>\
     <option value=2>Strand Bias</option>\
    </select>\
    <input class="RC-shy" id="RC-unzoomToggle" type="button" value="Zoom In" style="margin-left:10px;width:70px">\
    <span id="RC-toggleControls" style="float:right" class="RC-shy ui-icon ui-icon-search" title="Show/Hide the view/filter options panel"></span>\
    <span id="RC-help" style="float:right;margin-left:0;margin-right:0" class="RC-shy ui-icon ui-icon-help"></span>\
    <span id="RC-message" class="message"></span>\
  </div>\
  <div id="RC-noncanvas" style="background:#EEF;border-top:2px solid #666">\
   <div id="RC-plotspace" style="padding:4px">\
    <div id="RC-placeholder" style="width:100%"></div>\
    <div id="RC-cursorCoords" style="color:#bbd"></div>\
   </div>\
  </div>\
</div>\
<div id="RC-controlpanel" class="filter-panel" style="display:none">\
  <table><tr>\
    <td class="nwrap">Viewing Options:</td>\
    <td class="nwrap" id="RC-offScaleOutlierControl" style="display:none">\
      <span class="flyhelp" id="RC-offScaleOutlierLabel">Plot Outlier Off-scale</span>:\
      <input type="checkbox" id="RC-offScaleOutlier" checked="unchecked"></td>\
    <td class="nwrap"><span class="flyhelp" id="RC-showLegendLabel">Show Legend</span>:\
      <input type="checkbox" id="RC-showLegend" checked="checked"></td>\
    <td class="nwrap"><span class="flyhelp" id="RC-numPointsLabel">Bars/Points</span>:\
      <input type="text" class="numSearch" id="RC-numPoints" value=200 size=4>&nbsp;<span id="RC-numBars"></span></td>\
  </tr></table>\
  <table><tr>\
    <td class="nwrap">Reference Region:</td>\
    <td class="nwrap"><span class="flyhelp" id="RC-filterChromLabel">Chrom/Contig</span>:\
      <select class="txtSelect" id="RC-selectChrom" style="width:66px"></select>&nbsp;<span id="RC-numChroms"></span></td>\
    <td class="nwrap"><span class="flyhelp" id="RC-filterChromRangeLabel">Range</span>:\
      <input type="text" class="numSearch" id="RC-chromRange" value="" size=20></td>\
    <td class="nwrap">\
      <input type="button" id="RC-OpenIGV" value="View in IGV" style="width:76px"></td>\
  </tr></table>\
</div>\
<div id="RC-tooltip" style="display:none">\
  <div><span id="RC-tooltip-close" title="Close" class="help-box ui-icon ui-icon-close"></span></div>\
  <div><span id="RC-tooltip-zoomout" title="Zoom out from this region" class="help-box ui-icon ui-icon-zoomout"></span></div>\
  <div><span id="RC-tooltip-center" title="Center view on this region" class="help-box ui-icon ui-icon-arrowthick-2-e-w"></span></div>\
  <div><span id="RC-tooltip-zoomin" title="Zoom in on this region" class="help-box ui-icon ui-icon-zoomin"></span></div>\
  <div id="RC-tooltip-body"></div>\
</div>\
<div id="RC-helptext" class="helpblock" style="display:none">\
This chart shows the base coverage due to reads aligned across the whole reference.<br/><br/>\
For a reference of multiple contigs (chromosomes), the initial view will show total<br/>\
coverage as a data bar per contig. If there are many contigs each bar itself may<br/>\
represent a binned average over a smaller number of contigs. Click-and-drag or<br/>\
double-click the mouse pointer to zoom in on a range of contigs. Where data bars<br/>\
represent single contigs the double-click will zoom the view to base coveage across<br/>\
that contig. A single contig may also be viewed using the drop-down selector on the<br/>\
options panel; opened by clicking on the spy-glass icon at the top right of the chart.<br/><br/>\
With a single contig (chromosome) in view, typically each data bar (or overlay point)<br/>\
will represent the average base coveage over a range of the reference genome.<br/>\
Click-and-drag or double-click with the mouse to zoom in to regions along the contig,<br/>\
or type in a specific range to view usiing the Range field of the options panel.<br/><br/>\
Double-click in the white-space around the plotted data to zoom out (by 10x).<br/>\
Click on the Zoom Out button to return to the coverage view across the whole contig<br/>\
currently in view, or to zoom out to the (full) coverage-per-contig view.<br/><br/>\
Hover the mouse pointer over a data bar (or overlay point) to review minimal information<br/>\
or click on the data to show a detailed information box that remains until dismissed.<br/><br/>\
More viewing options are available using the Plot and Overlay selectors in the title bar<br/>\
and other controls on the options panel. Depending on the total number of contigs, the<br/>\
Zoom In button may also be available to view coverage over all contigs as a single stand.<br/>\
Look for additional fly-over help on or near the controls provided.\
</div>\
<input type="hidden" id="RC-ViewRequest"/>\
');

$(function () {

  // check placer element exists
  if( !$('#ReferenceCoverageChart').length ) return;

  // check browser environment
  var fixIE = (typeof G_vmlCanvasManager != 'undefined');
  var useFlash = (typeof FlashCanvas != 'undefined');
  var useExCan = (fixIE && !useFlash);

  // minimum sizes for chart widget
  var def_minWidth = 620;
  var def_minHeight = 200;

  // configure widget size and file used from placement div attributes
  var bbcFile = $("#ReferenceCoverageChart").attr("bbcfile");
  if( bbcFile == undefined  || bbcFile == "" ) {
    //alert("ERROR on page: ReferenceCoverageChart widget requires attribute 'bbcfile' is set.");
    $('#ReferenceCoverageChart').hide();
    return;
  }
  var chrcovFile = $("#ReferenceCoverageChart").attr("chrcovfile");
  if( chrcovFile == undefined || chrcovFile == "" ) {
    //alert("ERROR on page: ReferenceCoverageChart widget requires attribute 'chrcovfile' is set.");
    $('#ReferenceCoverageChart').hide();
    return;
  }
  var cbcFile = $("#ReferenceCoverageChart").attr("cbcfile");
  if( cbcFile == undefined ) cbcFile = '';
  var wgncovFile = $("#ReferenceCoverageChart").attr("wgncovfile");
  if( wgncovFile == undefined ) wgncovFile = '';

  var wholeGenome = $("#ReferenceCoverageChart").attr("genome");
  wholeGenome = (wholeGenome != undefined);

  var startCollapsed = $("#ReferenceCoverageChart").attr("collapse");
  startCollapsed = (startCollapsed != undefined);

  var startShowOptions = $("#ReferenceCoverageChart").attr("showoptions");
  startShowOptions = (startShowOptions != undefined);

  var startOutlierOffScale = $("#ReferenceCoverageChart").attr("outlieroffscale");
  startOutlierOffScale = (startOutlierOffScale != undefined);

  var tmp = $('#ReferenceCoverageChart').width();
  if( tmp < def_minWidth ) tmp = def_minWidth;
  $("#RC-chart").width(tmp);
  tmp = $('#ReferenceCoverageChart').height();
  if( tmp < def_minHeight ) tmp = def_minHeight;
  $("#RC-chart").height(tmp);
  $("#RC-placeholder").height(tmp-36);
  $("#ReferenceCoverageChart").css('height','auto');

  $("#RC-controlpanel").appendTo('#RC-titlebar');
  $("#RC-chart").appendTo('#ReferenceCoverageChart');
  $("#RC-chart").show();
  $('#RC-chart').css("height","auto");

  // some default values for plot display
  var def_minPoints = 11;  // odd so one is in the middle
  var def_numPoints = 200;
  var def_tinyValue = 0.0004;
  var def_outlierFactor = 4;
  var disableTitleBar = false;
  var placeholder = $("#RC-placeholder");
  var timeout = null;

  var dblclickUnzoomFac = 10;

  var resiz_def = {
    alsoResize: "#RC-placeholder",
    minWidth:def_minWidth,
    minHeight:def_minHeight,
    handles:"e,s,se",
    resize:function(e,u){ updatePlot(); }
  };
  $('#RC-chart').resizable(resiz_def);

  placeholder.bind("mouseleave", function() {
    if( !lastHoverBar.sticky ) hideTooltip();
  });

  $("#RC-collapsePlot").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#RC-plotspace').is(":visible") ) {
      $(this).attr("class","ui-icon ui-icon-triangle-1-s");
      $(this).attr("title","Expand view");
      $('#RC-controlpanel').slideUp();
      $('.RC-shy').fadeOut(400);
      $('#RC-chart').resizable('destroy');
      $('#RC-noncanvas').slideUp('slow');
      hideTooltip();
    } else {
      $(this).attr("class","ui-icon ui-icon-triangle-1-n");
      $(this).attr("title","Collapse view");
      $('.RC-shy').fadeIn(400);
      $('#RC-noncanvas').slideDown('slow',function(){
        $('#RC-chart').resizable(resiz_def);
      });
    }
    $("#RC-chart").css('height','auto');
  });

  $("#RC-toggleControls").click(function(e) {
    if( disableTitleBar ) return;
    $('#RC-chart').css("height","auto");
    if( $('#RC-controlpanel').is(":visible") ) {
      $('#RC-controlpanel').slideUp();
    } else {
      $('#RC-controlpanel').slideDown();
    }
  });

  $("#RC-help").click( function() {
    var offset = $("#RC-help").offset();
    var ypos = offset.left - $('#RC-helptext').width();
    $("#RC-help").removeAttr("title");
    $('#RC-helptext').css({
      position: 'absolute', display: 'none',
      top: offset.top+16, left: ypos+8
    }).appendTo("body").slideDown();
  });

  $("#RC-help").hover( null, function() {
    $('#RC-helptext').fadeOut(200);
    $("#RC-help").attr( "title", "Click for help." );
  });

  $('#RC-placeholder').dblclick(function(e) {
    if( zoomViewOnBin( lastHoverBar.binNum, true ) ) hideTooltip();
  });

  function zoomViewOnBin(binNum,zoomIn) {
    // Always perform zoom out if binNum < 0
    if( plotStats.numPlots <= 0 ) return false;
    var overzoom = plotStats.minX > 0 || plotStats.maxX < plotStats.numPoints;
    if( binNum >= 0 && zoomIn ) {
      if( overzoom ) return false;
      var chr = dataTable[binNum][DataField.contig_id];
      if( plotStats.binnedChroms ) {
        var i = chr.indexOf(' - ');
        if( i >= 0 ) {
          plotStats.zmSrtChrom = chr.substr(0,i);
          plotStats.zmEndChrom = chr.substr(i+3);
        } else {
          plotStats.zmSrtChrom = plotStats.zmEndChrom = chr;
        }
        unzoomData(); // requires file reload
        return true;
      }
      if( plotParams.dblCenter && plotStats.zoomChrom ) {
        return centerViewOnBin(binNum);
      }
      var srt = dataTable[binNum][DataField.pos_start];
      var end = dataTable[binNum][DataField.pos_end];
      zoomDataRange(chr,srt,end);
    } else if( plotStats.zoomChrom ) {
      // set zoom out range by limits of current view
      var chr = covFilter.chrom;
      var srt = covFilter.pos_srt;
      var end = covFilter.pos_end;
      // return to previous view if in over-zoom
      if( overzoom ) {
        plotStats.minX = 0;
        plotStats.maxX = plotStats.numPoints;
        setMaxZoomXtitle(0,plotStats.maxX-1);
        updatePlot(false);
        return true;
      } else if( binNum >= 0 && !zoomIn ) {
        // override limits by selected bin if provided
        var cbin = Math.floor(plotStats.numPoints/2);
        var csrt = dataTable[binNum][DataField.pos_start];
        var cend = dataTable[binNum][DataField.pos_end];
        var csiz = cend - csrt + 1;
        csrt -= (cbin * csiz) + 1;
        cend += cbin * csiz;
        if( csrt > 0 && cend <= plotStats.chromLength ) {
          chr = dataTable[binNum][DataField.contig_id];
          srt = csrt;
          end = cend;
        }
      }
      var siz = dblclickUnzoomFac*(end-srt+1);
      srt = Math.floor(0.5*(end+srt-siz));
      end = srt + siz - 1;
      zoomDataRange(chr,srt,end);
    } else if( plotStats.zmSrtChrom != "" ) {
      // flag this since requires a reload to find outer contig range
      plotStats.zmOutChrom = true;
      unzoomData();
    } else {
      return false;
    }
    return true;
  }

  function centerViewOnBin(binNum) {
    if( plotStats.numPlots <= 0 || binNum < 0 ) return false;
    // binNum 0-based so 99 (bin#100) is center of 200 bins and 100 (bin#101) is center of 201 bins
    var cbin = (plotStats.numPoints-1) >> 1;
    if( binNum == cbin ) return true;
    // check for zoomed multi-contig views
    if( plotStats.zmSrtChrom != "" ) return false;
    // skip for fully zoomed-out views
    if( !plotStats.zoomChrom ) return false;
    // to prevent zoom out view changing on re-center the current bin start position for the shift rather than it's center
    var srt = dataTable[0][DataField.pos_start];
    var end = dataTable[plotStats.numPoints-1][DataField.pos_end];
    var siz = end - srt + 1;
    srt += (binNum - cbin) * (siz / plotStats.numPoints);
    end = srt + siz - 1;
    var chr = dataTable[binNum][DataField.contig_id];
    zoomDataRange(chr,srt,end);
    return true;
  }

  //$('#RC-chart').noContext();

  function rightClickMenu(e) {
    alert("r-click");
  }

  // attempt to disable defalt context menu and enable r-click
  if( useFlash ) {
    // only works for flashcanvas pro!
    FlashCanvas.setOptions( {disableContextMenu : true} );
    // instead use area outside of canvas
    //$('#RC-noncanvas').noContext();
    //$('#RC-noncanvas').rightClick(rightClickMenu);
  } else {
    //$('#RC-chart').noContext();
    //$('#RC-chart').rightClick(rightClickMenu);
  }

  var plotStats = {
    xTitle : "Reference Range",
    defNumPoints : def_numPoints,
    minNumPoints : def_minPoints,
    maxNumPoints : 1000,
    maxXaxisLabels : 25,
    multiChrom : false,
    binnedChroms : false,
    zmSrtChrom : "",
    zmEndChrom : "",
    zmOutChrom : false,
    zoomInMode : false,
    zoomChrom: false,
    chromsInView : 0,
    targetsRepresented : 0,
    targetBinSize : 0,
    binnedData : false,
    onTargets : false,
    numFields : 0,
    numPlots : 0,
    numPoints : 0,
    wgnNumPoints : 0,
    minX : 0,
    maxX : 0,
    minY : 0,
    maxY : 0,
    tooltipZero : 0,
    totalChroms : 0,
    chromLength : 0,
    chrList : "",
    chrLens : {}
  };

  var plotParams = {
    resetYScale: false,
    showLegend : true,
    offScaleOutlier : false,
    numPoints : def_numPoints,
    aveBase : 1,
    barAxis : 0,
    overPlot : 0,
    zoomMode : 1,
    dblCenter : false
  };

  var covFilter = {
    inputfile : '',
    bbcfile : bbcFile,
    cbcfile : cbcFile,
    chrom : '',
    pos_srt : 0,
    pos_end : 0,
    maxrows : 200,
    clipleft : 0,
    clipright : 100,
    numrec : 0
  };

  var DataField = {
    contig_id : 0,
    pos_start : 1,
    pos_end : 2,
    fwd_reads : 3,
    rev_reads : 4,
    fwd_ont : 5,
    rev_ont : 6
  };

  var LegendLabels = {
    allReads : "Total Reads",
    fwdReads : "Forward Reads",
    revReads : "Reverse Reads",
    offReads : "Off-target",
    ontReads : "On-target",
    fwdOffReads : "Forward Off-target",
    fwdOntReads : "Forward On-target",
    revOffReads : "Reverse Off-target",
    revOntReads : "Reverse On-target",
    precentOntarg : "Percent On-Target",
    fwdBias : "Fwd Strand Bias"
  }

  var ColorSet = {
    allReads : "rgb(128,160,192)",
    fwdReads : "rgb(240,160,96)",
    revReads : "rgb(64,200,96)",
    offReads : "rgb(128,160,192)",
    ontReads : "rgb(0,0,128)",
    fwdOffReads : "rgb(255,160,96)",
    fwdOntReads : "rgb(220,96,32)",
    revOffReads : "rgb(96,200,160)",
    revOntReads : "rgb(32,160,64)",
    precentOntarg : "rgb(224,96,96)",
    fwdBias : "rgb(200,96,160)"
  }

  function setUnzoomTitle(zoomin) {
    var txt;
    if( zoomin) {
      txt = "'Zoom in' to a special view of coverage over the reference as a single sequence with all chromsomes (or contigs) "+
      "laid end-to-end. Coverage across each is distributed over a number of bins that is approximately proportional to the "+
      "relative sizes of the chromsome, with each represented by at least one bin.";
    } else {
      txt = "'Zoom out' to the maximum distance for the current view. If zoomed in on a single chromosome, this "+
      "will zoom out to the view for coverage across the whole of that chromosome. Otherwise it will return to the last "+
      "view for multiple contigs or back to the initial view for coverage across all contigs in the reference. "+
      "Note that this is a different operation from double-clicking the mouse in plot area (outside of any data bar). "+
      "That operation will zoom out to a region that is 10 times the current region size (from a zoomed-in view).";
    }
    $("#RC-unzoomToggle").val(zoomin ? "Zoom In" : "Zoom Out");
    $("#RC-unzoomToggle").attr( "title", txt );
  }

  function customizeChart() {
    // add fly-over help to controls here in case need to customize for chart data
    $("#RC-plotLabel").attr( "title",
      "Select the how the data is plotted.\n'Total Reads' shows bar plots of base reads aligned "+
      "to both DNA strands, whereas 'Strand Reads' plots the numbers of forward and reverse (DNA strand) base reads "+
      "separately, above and below the 0 reads line. If a set of target regions was specified, the numbers of reads "+
      "that were inside or outside of these regions are shown by stacked colors of each data bar." );
    $("#RC-overlayLabel").attr( "title",
      "Select a particular property of the data to plot as an overlay of points parallel to each data bar. " +
      "For example, the Strand Bias can be overlayed with the Total Reads plot to see if there is any obvious strand "+
      "bias for the reference regions in view. Whether there is specific or general strand bias "+
      "may depend on the current zoom level to those regions (i.e. the region size represented by the binned data). "+
      "Adding an overlay plot is also useful for identifying very low coverage regions, since these points are not "+
      "plotted for any region with 0 base coverage." );
    $("#RC-offScaleOutlierLabel").attr( "title",
      "Select to indicate that a single outlier data point is plotted off-scale. "+
      "The y-axis is re-scaled so that the over-represented contig (or range) does not hide the relative representation "+
      "of other data curently in view. This option only becomes available when the maximum value (height) of any "+
      "data point (bar) is at least "+def_outlierFactor+" times greater than all others in view." );
    $("#RC-showLegendLabel").attr( "title", "Select whether the legend is displayed over the plot area." );
    $("#RC-numPointsLabel").attr( "title",
      "This value specifies the maximum number of data bars and overlay points to display. Typically each bar (or point) "+
      "plotted will represent the binned totals and averages of many individual base regions along the genome. If there is "+
      "less data to plot than this value, e.g. when in the maximum zoom-out mode showing coverage per chromosome, the "+
      "number of bars actually plotted is displayed in parentheses. This value may be set to any value 10 and 1000, "+
      "although values greater than 200 are not recommended as this may make selection of individual bars difficult.\n"+
      "Note that the number of bars represented in the 'Zoom In' view, where coverage is visualized along the whole genome "+
      "with a proportional number of bins per chromosome, is fixed at 200 regardless of the current Bars/Points setting." );
    $("#RC-filterChromLabel").attr( "title",
      "Use this selector to select a particular chromosome (or contig) of the reference to view or to go to an overview " +
      "across the whole reference by selecting the 'ALL' value. " +
      "You may also change to a Chrom/Contig selection by typing its full name (or ID) in the Range field." );
    $("#RC-filterChromRangeLabel").attr( "title",
      "Use this selector to select a particular region of the currently selected chromosome to bring into view. "+
      "The range may be in the form <start>-<end> where 'start' and 'end' are positional base coordinates (1-based). "+
      "These numbers may include commas (which are ignored) but may not include other characters, such as space or dot. "+
      "After typing a value, press enter or tab or click on another field to execute your selection.\n"+
      "The actual range requested may be 'corrected' if it exceeds the size of the current chromosome or is adjusted to "+
      "statisfy the specified Bars/Points setting. In partcular, if only one number is provided the view will be centered "+
      "at this location with (e.g.) 200 base locations in view. Large ranges are typically adjusted to the nearest 1KB "+
      "range (or greater round-off) because of the way regions of the genome are pre-binned for this chart for performance.\n"+
      "If the value of the Range field is empty then it will be filled in with the whole range for the selected chromosome. "+
      "However, the Range entry is completely ignorred if the current Chrom/Contig setting is 'ALL'. You may also include "+
      "the chromosome name in the range by typing, e.g. chr1:100000-200000 or chr1:250000, which brings up the specified "+
      "location without having to first select the chromosome using the Chrom/Contig selector." );
    $("#RC-OpenIGV").attr( "title",
      "Click this button to open an instance of IGV (Integrated Genome Viewer) with whatever Chrom/Contig and Range selection "+
      "is currently in view. Your target/amplicon regions are also uploaded as a separate annotation track if a target regions "+
      "file was specified." );
    $("#RC-help").attr( "title", "Click for help." );
    setUnzoomTitle(true);
  }

  // (re)initiailize page from default user options
  customizeChart();
  updateGUIPlotParams();

  // --------- initialize plot bindings - controls inside plot area ----------


  var fieldIds = [];
  var dataTable = [];
  var plotData = [];
  var options = [];
  var pointMap = [];
  var plotObj = null;
  var canvas = null;

  var lastHoverBar = {binNum:-1, isRev:false, clickItem:null, postZoom:false, sticky:false, label:'' };

  // plotselected (area) is mapped to zoom - which is quite complicated for this app.
  placeholder.bind("plotselected", function(event, ranges) {
    if( plotParams.zoomMode == 2 ) {
      plotStats.minY = options.yaxes[0].min = ranges.yaxis.from;
      plotStats.maxY = options.yaxes[0].max = ranges.yaxis.to;
    }
    zoomViewToBinRange( Math.floor(ranges.xaxis.from), Math.floor(ranges.xaxis.to) );
  });

  function zoomViewToBinRange(binSrt,binEnd) {
    // correct for selections dragged beyond end of last bin
    if( binEnd >= plotStats.numPoints ) binEnd = plotStats.numPoints - 1;
    lastHoverBar.postZoom = true;
    // deal with range zooms across multiple and single contig views
    if( plotStats.multiChrom ) {
      plotStats.zmSrtChrom = dataTable[binSrt][DataField.contig_id];
      plotStats.zmEndChrom = dataTable[binEnd][DataField.contig_id];
      if( plotStats.binnedChroms ) {
        var i = plotStats.zmSrtChrom.indexOf(' - ');
        if( i >= 0 ) {
          plotStats.zmSrtChrom = plotStats.zmSrtChrom.substr(0,i);
        }
        i = plotStats.zmEndChrom.indexOf(' - ');
        if( i >= 0 ) {
          plotStats.zmEndChrom = plotStats.zmEndChrom.substr(i+3);
        }
      }
      if( plotStats.zmSrtChrom != plotStats.zmEndChrom ) {
        setUnzoomTitle(false);
        unzoomData(); // resets plotStats.zoomChrom
        return;
      }
    }
    // escape multi-contig view since only one contig is now selected
    plotStats.zmSrtChrom = "";
    plotStats.zmEndChrom = "";
    plotStats.multiChrom = false;
    plotStats.zoomInMode = false;
    // check for zoom on a chromosome view
    if( plotStats.binnedData ) {
      // ensure only the middle chromosome is selected
      if( covFilter.chrom == '' ) {
        var midBin = Math.round(0.5*(binSrt+binEnd));
        var chr = dataTable[midBin][DataField.contig_id];
        while( dataTable[binSrt][DataField.contig_id] != chr ) ++binSrt;
        while( dataTable[binEnd][DataField.contig_id] != chr ) --binEnd;
      }
      var chr = dataTable[binSrt][DataField.contig_id];
      var srt = dataTable[binSrt][DataField.pos_start];
      var end = dataTable[binEnd][DataField.pos_end];
      zoomDataRange(chr,srt,end);
    } else {
      // recenter at max zoom, or zoom out at over zoom
      var clip = binEnd - binSrt + 1;
      if( clip >= plotStats.minNumPoints ) {
        plotStats.minX = options.xaxis.min = binSrt;
        plotStats.maxX = options.xaxis.max = binEnd + 1;
      } else {
        var diff = 0.5 * (plotStats.minNumPoints - clip);
        plotStats.minX = Math.floor(0.5+binSrt-diff);
        plotStats.maxX = plotStats.minX + plotStats.minNumPoints;
        if( plotStats.minX < 0 ) {
          plotStats.maxX -= plotStats.minX;
          plotStats.minX = 0;
        }
        if( plotStats.maxX > plotStats.numPoints ) {
          plotStats.minX -= plotStats.maxX - plotStats.numPoints;
          plotStats.maxX = plotStats.numPoints;
        }
        binSrt = plotStats.minX;
        binEnd = plotStats.maxX - 1;
      }
      plotStats.targetBinSize = 1;
      setMaxZoomXtitle(binSrt,binEnd);
      updatePlot();
    }
  }

  function setMaxZoomXtitle(binSrt,binEnd) {
    plotStats.targetsRepresented = binEnd - binSrt + 1;
    var chr = dataTable[binSrt][DataField.contig_id];
    plotStats.xTitle = chr + ': ' + commify(dataTable[binSrt][DataField.pos_start]) + ' - ' + commify(dataTable[binEnd][DataField.pos_end]);
    plotStats.xTitle += '  (' + commify(plotStats.targetsRepresented) + ' bases)';
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

  placeholder.bind("plotclick", function(e,pos,item) {
    // ignore false triggering due to mouse selection for zoom
    if( lastHoverBar.postZoom ) {
      lastHoverBar.postZoom = false;
      return;
    }
    if( cursorOverItem(pos,item) ) {
      showTooltip(item,pos,true);
      lastHoverBar.clickItem = item;
      if( item ) plotObj.highlight(item.series,item.datapoint);
    } else {
      hideTooltip();
    }
  });

  placeholder.bind("mouseleave", function() {
    setCursor('default');
  });

  function cursorOverPlot(x,y) {
    return plotStats.numPlots > 0 && x >= plotStats.minX && x < plotStats.maxX && y >= plotStats.minY && y <= plotStats.maxY;
  }

  function cursorOverItem(pos,item) {
    if( pos.x >= plotStats.numPoints ) return false;
    return item || Math.abs(pos.y) < plotStats.tooltipZero;
  }

  function hideTooltip() {
    if( plotObj ) plotObj.unhighlight();
    $("#RC-tooltip").hide();
    lastHoverBar.binNum = -1;
    lastHoverBar.clickItem = null;
    lastHoverBar.sticky = false;
    lastHoverBar.label = '';
  }

  function showTooltip(item,pos,sticky) {
    if( timeout != null ) {
      clearTimeout(timeout);
      timeout = null;
    }
    // item not passed for 0-width bars: if passed check for overlay points
    var isRev = pos.y < 0;
    var label = '', bgColor;
    if( item )
    {
      var tmp = item.series.label;
      if( tmp == LegendLabels.fwdBias || tmp == LegendLabels.precentOntarg ) label = tmp;
    }
    if( label != '' ) {
      bgColor = item.series.color;
    } else if( plotParams.barAxis == 0 ) {
      label = LegendLabels.allReads;
      bgColor = ColorSet.allReads;
    } else {
      label = isRev ? LegendLabels.revReads : LegendLabels.fwdReads;
      bgColor = isRev ? ColorSet.revReads : ColorSet.fwdReads;
    }
    var binNum = Math.floor(pos.x);
    if( binNum >= plotStats.numPoints ) binNum = plotStats.numPoints-1;
    if( lastHoverBar.binNum == binNum && lastHoverBar.sticky == sticky &&
        lastHoverBar.isRev == isRev && lastHoverBar.label == label ) return;
    hideTooltip();
    // correct for over-approximate bin selection for point hover with missing data points
    var clickBar = dataBar(label);
    if( !clickBar ) {
      // if item is available try to map points throu
      if( item != null && pointMap.length > 0 ) binNum = pointMap[item.dataIndex];
      if( dataTable[binNum][DataField.fwd_reads]+dataTable[binNum][DataField.rev_reads] == 0 ) return;
    }
    lastHoverBar.binNum = binNum;
    lastHoverBar.isRev = isRev;
    lastHoverBar.sticky = sticky;
    lastHoverBar.label = label;
    $('#RC-tooltip-body').html( sticky ? tooltipMessage(label,binNum) : tooltipHint(label,binNum));
    var posx = pos.pageX+10;
    var posy = pos.pageY-10;
    var minTipWidth = 0;
    if( sticky ) {
      if( clickBar ) minTipWidth = plotParams.barAxis ? 230 : 210;
      var cof = $('#RC-chart').offset();
      var ht = $('#RC-tooltip').height();
      var ymax = cof.top + $('#RC-chart').height() - ht;
      posy = pos.pageY - $('#RC-tooltip').height()/2;
      if( posy > ymax ) posy = ymax;
      if( posy < cof.top-4 ) posy = cof.top-4;
      var xmid = cof.left + $('#RC-chart').width()/2;
      if( pos.pageX > xmid ) posx = pos.pageX - $('#RC-tooltip').width() - 16;
    }
    $('#RC-tooltip').css({
      position: 'absolute', left: posx, top: posy, minWidth: minTipWidth,
      background: bgColor, padding: (sticky ? 3 : 4)+'px',
      border: (sticky ? 2 : 1)+'px solid #444',
      opacity: sticky ? 1: 0.7
    }).appendTo("body").fadeIn(sticky ? 10 : 100);
    if( !sticky ) {
      timeout = setTimeout( function() { hideTooltip(); }, 200 );
    }
  }

  $('#RC-tooltip-close').click( function() {
    hideTooltip();
  });

  $('#RC-tooltip-zoomout').click( function() {
    if( zoomViewOnBin( lastHoverBar.binNum, false ) ) hideTooltip();
  });

  $('#RC-tooltip-center').click( function() {
    if( centerViewOnBin( lastHoverBar.binNum ) ) hideTooltip();
  });

  $('#RC-tooltip-zoomin').click( function() {
    if( zoomViewOnBin( lastHoverBar.binNum, true ) ) hideTooltip();
  });

  function dataBar(id) {
    return (id === LegendLabels.fwdReads || id === LegendLabels.revReads || id === LegendLabels.allReads);
  }

  function tooltipHint(id,bin) {
    $('#RC-tooltip-close').hide();
    $('#RC-tooltip-zoomout').hide();
    $('#RC-tooltip-center').hide();
    $('#RC-tooltip-zoomin').hide();
    if( dataBar(id) ) {
      var chr = dataTable[bin][DataField.contig_id];
      if( dataTable[bin][DataField.pos_start] < 0 ) {
        return chr;
      } else if( dataTable[bin][DataField.pos_start] == 1 && dataTable[bin][DataField.pos_end] == plotStats.chrLens[chr] ) {
        return chr;
      } else if( dataTable[bin][DataField.pos_start] == dataTable[bin][DataField.pos_end] ) {
        return chr+":"+dataTable[bin][DataField.pos_start];
      }
      return chr+":"+dataTable[bin][DataField.pos_start]+"-"+dataTable[bin][DataField.pos_end];
    }
    var totalReads = dataTable[bin][DataField.fwd_reads] + dataTable[bin][DataField.rev_reads];
    if( id === LegendLabels.precentOntarg ) {
      var onTarget = dataTable[bin][DataField.fwd_ont]+dataTable[bin][DataField.rev_ont];
      return (totalReads > 0 ? sigfig(100 * onTarget / totalReads) : 0)+'%';
    } else if( id === LegendLabels.fwdBias ) {
      return sigfig(totalReads > 0 ? 100 * dataTable[bin][DataField.fwd_reads] / totalReads : 50)+'%';
    }
    return '?';
  }

  function tooltipMessage(id,bin) {
    $('#RC-tooltip-close').show();
    $('#RC-tooltip-zoomout').show();
    $('#RC-tooltip-center').show();
    $('#RC-tooltip-zoomin').show();
    var br = "<br/>";
    var i = id.indexOf(' ');
    var dirStr = id.substr(0,i);
    var dir = dirStr.charAt(0);
    var binChrom = dataTable[bin][DataField.pos_start] < 0;
    var regionLen = dataTable[bin][DataField.pos_end];
    if( !binChrom ) regionLen += 1-dataTable[bin][DataField.pos_start];
    var numReads = dataTable[bin][DataField.fwd_reads]+dataTable[bin][DataField.rev_reads];
    var msg = "<span style='white-space:nowrap'>"+id+" in bin#"+(bin+1)+".</span>"+br;
    if( binChrom ) {
      msg += "Contigs: "+dataTable[bin][DataField.contig_id]+br;
      msg += "Number of contigs: "+(-dataTable[bin][DataField.pos_start])+br;
      msg += "Total contig length: "+regionLen+br;
    } else {
      msg += "Contig: "+dataTable[bin][DataField.contig_id]+br;
      msg += "Region: "+dataTable[bin][DataField.pos_start]+"-"+dataTable[bin][DataField.pos_end]+br;
      msg += "Region length: "+regionLen+br;
    }
    if( id == LegendLabels.fwdBias ) {
      var bias = numReads >  0 ? 100 * dataTable[bin][DataField.fwd_reads] / numReads : 50;
      if( numReads >  0 ) msg += "Forward reads: "+sigfig(bias)+'%'+br;
      msg += bias >= 50 ? "Forward" : "Reverse";
      bias = Math.abs(2*bias-100);
      return msg + " bias: "+sigfig(bias)+'%'+br;
    } else if( id == LegendLabels.precentOntarg ) {
      msg += "Total base reads: "+numReads+br;
      var onTarget = dataTable[bin][DataField.fwd_ont]+dataTable[bin][DataField.rev_ont];
      var pcOntarg = numReads >  0 ? 100 * onTarget / numReads : 0;
      if( numReads >  0 ) msg += "On-target reads: "+sigfig(pcOntarg)+'%'+br;
      return msg;
    }
    var dirReads = dir == 'T' ? numReads : dataTable[bin][dir == 'F' ? DataField.fwd_reads : DataField.rev_reads];
    var aveReads = dirReads / (regionLen);
    msg += dirStr + " base reads: "+dirReads+br;
    msg += "Average base read depth:  "+sigfig(aveReads)+br;
    if( dir == 'T' ) {
      var bias = numReads >  0 ? 100 * dataTable[bin][DataField.fwd_reads] / numReads : 0;
      msg += "Percent forward reads: "+sigfig(bias)+'%'+br;
    } else {
      var bias = numReads >  0 ? 100 * dirReads / numReads : 0;
      msg += "Percent "+(dir == 'F' ? "forward" : "reverse")+" reads: "+sigfig(bias)+'%'+br;
    }
    if( plotStats.onTargets ) {
      var onTarget = dataTable[bin][DataField.fwd_ont]+dataTable[bin][DataField.rev_ont];
      var dirTarg = dir == 'T' ? onTarget : dataTable[bin][dir == 'F' ? DataField.fwd_ont : DataField.rev_ont];
      var pcOntarg = dirReads >  0 ? 100 * dirTarg / dirReads : 0;
      msg += dirStr + " on-target base reads: "+dirTarg+br;
      msg += dirStr + " off-target base reads: "+(dirReads-dirTarg)+br;
      msg += dirStr + " on-target fraction: "+sigfig(pcOntarg)+'%'+br;
    }
    return msg;
  }

  function sigfig(val) {
    var av = Math.abs(val);
    if( av == 0 ) return "0";
    if( av >= 100 ) return val.toFixed(0);
    if( av >= 10 ) return val.toFixed(1);
    if( av >= 1 ) return val.toFixed(2);
    if( av >= 0.1 ) return val.toFixed(3);
    return val.toFixed(3);
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

  $('#snapShot').click(function() {
    if( canvas != null ) {
      canvas2png(canvas);
    }
  });

  function updateGUIPlotParams() {
    $('.RC-selectParam#barAxis').val(plotParams.barAxis);
    $('.RC-selectParam#overPlot').val(plotParams.overPlot);
    $('#RC-numPoints').val(plotParams.numPoints);
    $('#RC-offScaleOutlier').attr('checked',plotParams.offScaleOutlier);
    $('#RC-showLegend').attr('checked',plotParams.showLegend);
    $('#RC-selectChrom').val(covFilter.chrom);
  }

  $('#RC-offScaleOutlier').change(function() {
    plotParams.offScaleOutlier = ($(this).attr("checked") == "checked");
    updatePlot();
  });

  $('#RC-showLegend').change(function() {
    plotParams.showLegend = ($(this).attr("checked") == "checked");
    updatePlot();
  });

  $('#RC-numPoints').change(function() {
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
      covFilter.maxrows = plotParams.numPoints = val;
      zoomData();
    }
  });
    
  $('#RC-selectChrom').change(function() {
    updateContigView(this.value);
  });

  $('#RC-chromRange').change(function() {
    // first check for a valid (leading) chromosome ID
    var val = $.trim(this.value);
    var chr = val;
    var selChr = $('#RC-selectChrom option[value="'+chr+'"]').length;
    if( selChr > 0 ) {
      val = '';
    } else {
      var j = val.indexOf(':');
      if( j >= 0 ) {
        chr = $.trim(val.substring(0,j));
        val = $.trim(val.substring(j+1));
        if( $('#RC-selectChrom option[value='+chr+']').length == 0 ) chr = '';
      } else {
        chr = '';
      }
    }
    if( chr != '' ) {
      if( val == '' ) {
        this.value = '';
        updateContigView(chr);
        return;
      }
      setContig(chr);
      setUnzoomTitle(false);
    } else if( covFilter.chrom == '' ) {
      // range is reset if no contig selected
      this.value = '';
      return;
    }
    // parse range string and reset to valid range
    plotStats.zoomChrom = setContigRange(val);
    zoomData();
  });

  function setContigRange(rangeStr) {
    // range becomes whole contig on parsing error or empty string
    // return true if a slice of the contig is selected, or false if whole contig in view
    var val = $.trim(rangeStr.replace(/,/g,''));
    var pos = val.split('-');
    var srt = 1;
    var end = plotStats.chromLength;
    // short circuit for invalid format of start location
    if( pos.length < 1 || isNaN(pos[0]) ) {
      $('#RC-chromRange').val(commify(srt)+'-'+commify(end));
      covFilter.pos_srt = srt;
      covFilter.pos_end = end;
      return false;
    }
    // center view around bases at single position
    if( pos.length == 1 ) {
      srt = parseInt(pos[0]);
      end = srt;
      srt -= plotStats.numPoints/2;
      srt = Math.floor(srt);
      if( srt < 1 ) srt = 1;
      end = srt + plotStats.numPoints - 1;
      if( end > plotStats.chromLength ) {
        end = plotStats.chromLength;
        srt = end - plotStats.numPoints + 1;
        if( srt < 1 ) srt = 1;
      }
    } else {
      if( pos[0] == '' || isNaN(pos[0]) ) srt = 1;
      else srt = parseInt(pos[0]);
      if( pos[1] == '' || isNaN(pos[1]) ) end = plotStats.chromLength;
      else end = parseInt(pos[1]);
      if( end < srt ) {
        var swp = end;
        end = srt;
        srt = swp;
      }
    }
    // correct for typical 1-off numbering
    if( srt % 1000 == 0 ) ++srt;
    // expand range if insufficient
    if( (end-srt+1) < plotStats.numPoints ) {
      srt = Math.floor(0.5*(end+srt+1-plotStats.numPoints));
      if( srt < 1 ) srt = 1;
      end = srt + plotStats.numPoints - 1;
      if( end > plotStats.chromLength ) {
        end = plotStats.chromLength;
        srt = end - plotStats.numPoints + 1;
        if( srt < 1 ) srt = 1;
      }
    }
    $('#RC-chromRange').val(commify(srt)+'-'+commify(end));
    covFilter.pos_srt = srt;
    covFilter.pos_end = end;
    return (srt > 1 || end < plotStats.chromLength);
  }

  function updateContigView(chr) {
    if( chr == '' ) return;
    setContig(chr);
    if( chr == 'ALL' ) {
      covFilter.inputfile = wgncovFile != '' ? wgncovFile : chrcovFile;
      covFilter.numrec = 0;
    }
    unzoomData();
    setUnzoomTitle(false);
  }

  function setContig(chr) {
    if( chr == '' ) return;
    if( chr == 'ALL' ) {
      plotStats.zmSrtChrom = "";
      plotStats.zmEndChrom = "";
      covFilter.chrom = '';
      plotStats.chromLength = 0;
    } else if( chr != covFilter.chrom ) {
      covFilter.chrom = chr;
      plotStats.chromLength = plotStats.chrLens[chr];
    }
    $('#RC-selectChrom').val(chr);
  }

  $('.RC-selectParam').change(function() {
    plotParams[this.id] = parseInt(this.value);
    plotParams.resetYScale = this.id == 'barAxis';
    autoHideLegend();
    updatePlot();
  });

  function autoHideLegend() {
    if( plotParams.showLegend && !plotStats.onTargets && plotParams.barAxis == 0 && plotParams.overPlot == 0 ) {
      $('#RC-showLegend').attr('checked', plotParams.showLegend = false );
    } else if( !plotParams.showLegend ) {
      $('#RC-showLegend').attr('checked', plotParams.showLegend = true );
    }
  }
  
  $("#RC-unzoomToggle").click(function() {
    // check for zoom out after zoom in mode and not at max zoom out already
    if( plotStats.multiChrom && plotStats.zmSrtChrom == "" ) {
      plotStats.zoomInMode = false;
    }
    if( plotStats.zoomInMode || this.value == "Zoom In" ) {
      plotStats.zmSrtChrom = "";
      plotStats.zmEndChrom = "";
      plotStats.zmOutChrom = false;
      unzoomToFile( wgncovFile );
      plotStats.wgnNumPoints = plotStats.numPoints;
      plotStats.zoomInMode = true;
      setUnzoomTitle(false);
      return;
    }
    // this means zoomed in on a contig
    if( plotStats.zoomChrom ) {
      if( covFilter.pos_srt > 1 || covFilter.pos_end < plotStats.chromLength ) {
        // make exception for single, especially small, genomes
        if( plotStats.totalChroms == 1 && plotStats.wgnNumPoints == plotParams.numPoints && wgncovFile != '' ) {
          unzoomToFile( wgncovFile );
          plotStats.zoomInMode = true;
        } else {
          unzoomData();
        }
        return;
      }
    }
    // always zoom out to full view if in multiChrom view
    if( plotStats.multiChrom ) {
      plotStats.zmSrtChrom = "";
      plotStats.zmEndChrom = "";
    }
    plotStats.chrList = '';
    plotStats.chrLens = {};
    covFilter.chrom = '';
    unzoomToFile( chrcovFile );
    if( wgncovFile != '' ) {
      setUnzoomTitle(true);
    }
  });

  $('#RC-OpenIGV').click(function() {
    window.open( linkIGV( getDisplayRegion(0) ) );
  });

  function getDisplayRegion(viewBuffer) {
    var chr = $('#RC-selectChrom').val();
    if( chr == '' || chr == 'ALL' ) return '';
    var srt = covFilter.pos_srt;
    var end = covFilter.pos_end;
    if( srt <= 0 ) return chr;
    if( end <= 0 ) return chr+':'+end;
    // expand the region slightly for better viewing in context
    srt -= viewBuffer;
    if( srt < 1 ) srt = 1;
    end += viewBuffer;
    if( end > plotStats.chromLength ) end = plotStats.chromLength;
    return chr+':'+srt+'-'+end;
  }

  function linkIGV(region) {
    if( region == undefined || region == null || region == "" ) {
      var i = plotStats.chrList.indexOf(':');
      if( i > 0 ) { region = plotStats.chrList.substr(0,i); }
    }
    var locpath = window.location.pathname.substring(0,window.location.pathname.lastIndexOf('/'));
    var igvURL = window.location.protocol + "//" + window.location.host + "/auth" + locpath + "/igv.php3";
    var launchURL = window.location.protocol + "//" + window.location.host + "/IgvServlet/igv";
    return launchURL + "?locus="+region+"&sessionURL="+igvURL;
  }

  // grab update request from external source (to RC)
  $('#RC-ViewRequest').change(function() {
    // ignore request if widget currently disabled
    if( disableTitleBar ) return;
    // force chart into view when request recieved - circumvents display issues if newly ploted while collapsed
    if( !$('#RC-plotspace').is(":visible") ) {
      $("#RC-collapsePlot").click();
    }
    $('#RC-chromRange').val(this.value).change();
  });

  function unzoomToFile(filename) {
    covFilter.inputfile = filename;
    covFilter.chrom = '';
    plotStats.minNumPoints = def_minPoints;
    plotParams.numPoints = plotStats.defNumPoints = def_numPoints;
    plotParams.zoomMode = 1;
    $('#RC-numPoints').val(plotParams.numPoints);
    unzoomData(true);
    setChromSearch();
  }

  function setChromSearch() {
    var selObj = $('#RC-selectChrom');
    selObj.empty();
    selObj.css('width','66px');
    if( plotStats.chrList == '' ) return;
    var chrs = plotStats.chrList.split(':');
    plotStats.totalChroms = chrs.length - 1;
    $('#RC-numChroms').text( '('+plotStats.totalChroms+')' );
    if( plotStats.totalChroms > 1 ) {
      selObj.append("<option value='ALL'>ALL</option>");
    }
    var mclen = 0;
    for( var i = 0; i < plotStats.totalChroms; ++i ) {
      selObj.append("<option value='"+chrs[i]+"'>"+chrs[i]+"</option>");
      if( chrs[i].length > mclen ) mclen = chrs[i].length;
    }
    if( mclen > 6 ) selObj.css('width','');
    // if only one chromosome then the selection cannot be changed => ensure it is the one selected
    if( plotStats.totalChroms == 1 ) {
      covFilter.chrom = chrs[0];
      plotStats.chromLength = plotStats.chrLens[chrs[0]];
    }
  }

  function loadData() {
    loadTSV();
    updateContigRange();
    updatePlotStats();
  }

  function updateContigRange() {
    plotStats.multiChrom = false;
    plotStats.binnedChroms = false;
    if( dataTable.length <= 1 ) return;
    plotStats.multiChrom = (dataTable[0][DataField.contig_id] != dataTable[dataTable.length-1][DataField.contig_id]);
    if( plotStats.multiChrom || plotStats.zmOutChrom ) {
      zoomcontigs();
      plotStats.multiChrom = (dataTable[0][DataField.contig_id] != dataTable[dataTable.length-1][DataField.contig_id]);
      plotStats.binnedChroms = bincontigs();
    }
  }

  function zoomcontigs() {
    // focus on range of contigs after loading
    // Also ensure the Zoom In button is only available at max zoom out
    if( plotStats.zmSrtChrom == "" && plotStats.zmEndChrom == "" ) {
      if( wgncovFile != '' ) {
        setUnzoomTitle(true);
      }
      return;
    } else if( plotStats.zoomInMode ) {
      setUnzoomTitle(false);
    }
    var srtRange = 0;
    var endRange = dataTable.length;
    var i = 0;
    for( ; i < dataTable.length; ++i ) {
      if( dataTable[i][0] == plotStats.zmSrtChrom ) {
        srtRange = i;
        break;
      }
    }
    // look backwards for end contig for sake of concat contigs zoom
    for( i = dataTable.length-1; i >= 0; --i ) {
      if( dataTable[i][0] == plotStats.zmEndChrom ) {
        endRange = i+1;
        break;
      }
    }
    // check if this is to be a zoom-out operation
    var newcv = endRange - srtRange; // default zoom in range - to specified contigs
    if( plotStats.zmOutChrom ) {
      plotStats.zmOutChrom = false;  // reset default operation to zoom in
      if( plotStats.numPoints < plotStats.maxXaxisLabels ) {
        // zoom out from over-zoom to max labels
        newcv = plotStats.maxXaxisLabels;
      } else {
        // zoom out by factor
        newcv *= dblclickUnzoomFac;
      }
    } else if( plotStats.zoomInMode ) {
      // no zoom-in restrictions for concat contigs view
    } else if( newcv < plotStats.maxXaxisLabels ) {
      if( plotStats.numPoints > plotStats.maxXaxisLabels ) {
        // zoom in to max labels before over-zoom
        newcv = plotStats.maxXaxisLabels;
      } else if( newcv < plotStats.minNumPoints ) {
        // zoom in to max zoom in level
        newcv = plotStats.minNumPoints;
      }
    }
    // zoom to new contig range
    if( newcv >= dataTable.length ) {
      plotStats.zmSrtChrom = "";
      plotStats.zmEndChrom = "";
      return;
    }
    srtRange = parseInt(0.5*(endRange + srtRange - newcv));
    if( srtRange < 0 ) srtRange = 0;
    endRange = srtRange + newcv;
    if( endRange > dataTable.length ) {
      endRange = dataTable.length;
      srtRange = endRange - newcv;
    }
    // need to reset these for correct level of zoom-out, etc.
    plotStats.zmSrtChrom = dataTable[srtRange][DataField.contig_id];
    plotStats.zmEndChrom = dataTable[endRange-1][DataField.contig_id];
    // reduce data:- could later just set limits of loaded contigs for viewing
    i = 0;
    for( var j = srtRange; j < endRange; ++j ) {
      dataTable[i++] = dataTable[j];
    }
    dataTable = dataTable.slice( 0, i );
  }

  function bincontigs() {
    // resolve issues of having too many contigs by binning
    if( dataTable.length <= plotParams.numPoints ) return false;
    var numFields = dataTable[0].length;
    var binSize = dataTable.length / plotParams.numPoints;
    var binIndx = 0;
    var binCnt = 0;
    var binMod = 0;
    for( var i = 0; i < dataTable.length; ++i ) {
      binMod += 1.0;
      if( ++binCnt == 1 ) {
        dataTable[binIndx] = dataTable[i];
      } else {
        for( var j = 2; j < numFields; ++j ) {
          dataTable[binIndx][j] += dataTable[i][j];
        }
      }
      if( binMod >= binSize ) {
        if( binCnt > 1 ) {
          dataTable[binIndx][0] += ' - ' + dataTable[i][0];
          dataTable[binIndx][1] = -binCnt;  // indicates size of bin
        }
        binMod -= binSize;
        binCnt = 0;
        ++binIndx;
      }
    }
    // finish up last bin if round-off error in binMod
    if( binCnt > 1 ) {
      dataTable[binIndx][0] += ' - ' + dataTable[i-1][0];
      dataTable[binIndx][1] = binCnt;
    }
    dataTable = dataTable.slice( 0, plotParams.numPoints );
    plotStats.endRange = dataTable.length;
    return true;
  }

  function zoomDataRange(chr,srt,end) {
    setContig(chr);
    // adjust coordinates if would put view beyond the ends of selected contig
    var siz = end - srt;
    if( srt < 1 ) srt = 1;
    end = srt + siz;
    if( end > plotStats.chromLength ) {
      end = plotStats.chromLength;
      srt = end - siz;
      if( siz < 0 ) siz = 1;
    }
    // reset flags based on viewing single contig
    plotStats.zmSrtChrom = "";
    plotStats.zmEndChrom = "";
    plotStats.zmOutChrom = false;
    plotStats.zoomInMode = false;
    plotStats.zoomChrom = setContigRange(srt+'-'+end);
    zoomData();
    setUnzoomTitle(false);
  }

  function zoomData(fromFile) {
    loadData();
    // to avoid recursion
    if( fromFile == undefined || fromFile == false ) {
      // For single contig references the whole genome view is better binned than using the coarse grained view
      // and the difference is noticable when switching from the whole genome view. Hence override this situation.
      if( plotStats.totalChroms == 1 && covFilter.pos_srt == 1 && covFilter.pos_end == plotStats.chromLength ) {
        if( plotStats.wgnNumPoints == plotParams.numPoints && wgncovFile != '' ) {
          setUnzoomTitle(false);
          unzoomToFile( wgncovFile );
          return;
        }
      }
    }
    updatePlot();
  }

  function unzoomData() {
    plotStats.zoomChrom = false;
    covFilter.pos_srt = 0;
    covFilter.pos_end = 0;
    covFilter.maxrows = plotParams.numPoints;
    loadData();
    updatePlot();
  }

  function loadError() {
    alert("An error occurred while loading from "+
      (covFilter.chrom == '' ? covFilter.inputfile : covFilter.bbcfile)+"\n"+
      covFilter.chrom+":"+covFilter.pos_srt+"-"+covFilter.pos_end+"\n("+
      covFilter.clipleft+"-"+covFilter.clipright+")");
    $('#RC-message').text('');
  }

  // load data using PHP to dataTable[] using options in covFilter{}
  function loadTSV() {
    var readChrList = (plotStats.chrList == '');
    var chrid = '';
    var noTargets = false;
    var srcf = (covFilter.chrom == '' ? covFilter.inputfile : 'lifechart/region_coverage.php3');
    dataTable = [];
    $('#RC-message').text('Loading...');
    $.ajaxSetup( {dataType:"text",async:false} );
    $.get(srcf, covFilter, function(mem) {
      var lines = mem.split("\n");
      $.each(lines, function(n,row) {
        var fields = $.trim(row).split('\t');
        if( n == 0 ) {
          fieldIds = fields;
          noTargets = fields.length <= 6;
          if( fields.length < 3 ) {
            loadError();
            return false;
          }
          if( fields[0].substr(0,5).toLowerCase() == 'error' ) alert(row);
        } else if( fields[0] != "" ) {
          // important to convert numeric fields to numbers for performance
          for( var i = 1; i < fields.length; ++i ) { fields[i] = +fields[i]; }
          if( noTargets ) fields[5] = fields[6] = 0;
          dataTable.push( fields );
          if( readChrList && chrid != fields[0] ) {
            chrid = fields[0];
            plotStats.chrList += chrid + ':';
            plotStats.chrLens[chrid] = parseInt(fields[2]);
          }
        }
      });
    }).error(loadError).success(function(){
      $('#RC-message').text('');
    });
  }

  function commify(val) {
    // expects positive integers
    var jrs = val.toString();
    return jrs.replace(/(\d)(?=(\d\d\d)+(?!\d))/g, "$1,");
  }

  function updatePlotStats() {
    plotStats.numPoints = dataTable.length;
    plotStats.numFields = fieldIds.length;
    plotStats.onTargets = !wholeGenome && fieldIds.length > 5;
    if( plotStats.numPoints == 0 ) {
      plotStats.minX = 0;
      return;
    }
    // to show the full width of last bin, maxX is the limit to start of one bin beyond those shown
    plotStats.minX = 0;
    plotStats.maxX = plotStats.numPoints;
    var chr1 = dataTable[0][DataField.contig_id];
    var str1 = dataTable[0][DataField.pos_start];
    var lastp = plotStats.numPoints - 1;
    var chrN = dataTable[lastp][DataField.contig_id];
    var endN = dataTable[lastp][DataField.pos_end];
    plotStats.multiChrom = (chr1 != chrN);
    var numRep = 0;
    var numChr = 0;
    var lastChr = "";  // required for counting when concat contigs view
    for( var i = 0; i < plotStats.numPoints; ++i ) {
      if( plotStats.binnedChroms ) {
        numChr += Math.abs(dataTable[i][DataField.pos_start]);
        numRep += dataTable[i][DataField.pos_end];
      } else {
        if( dataTable[i][DataField.contig_id] != lastChr ) {
          ++numChr;
          lastChr = dataTable[i][DataField.contig_id];
        }
        numRep += dataTable[i][DataField.pos_end] - dataTable[i][DataField.pos_start] + 1;
      }
    }
    plotStats.chromsInView = numChr;
    plotStats.targetBinSize = numRep / covFilter.maxrows;
    plotStats.targetsRepresented = Math.floor(numRep);
    // adjust contig title if binned
    if( plotStats.binnedChroms ) {
      var i = chr1.indexOf(' - ');
      if( i >= 0 ) chr1 = chr1.substr(0,i);
      i = chrN.indexOf(' - ');
      if( i >= 0 ) chrN = chrN.substr(i+3);
    }
    plotStats.xTitle = chr1;
    if( plotStats.multiChrom ) {
      plotStats.xTitle = chr1 + ' - ' + chrN;
    } else if( str1 > 1 || endN < plotStats.chromLength ) {
      plotStats.xTitle += ': ' + commify(str1) + ' - ' + commify(endN);
    }
    plotStats.xTitle += '  (' + (plotStats.multiChrom ? commify(numChr) + ' contigs, ' : '');
    plotStats.xTitle += commify(plotStats.targetsRepresented) + ' bases)';
    plotStats.binnedData = (plotStats.targetBinSize > 1.0000001);
    if( !plotStats.binnedData ) plotStats.targetBinSize = 1;

    // check for small (filtered) data sets
    plotStats.minNumPoints = plotStats.numPoints < def_minPoints ? plotStats.numPoints : def_minPoints;

    // update filters to reflect current loading
    if( chr1 == chrN ) {
      covFilter.pos_srt = str1;
      covFilter.pos_end = endN;
      $("#RC-chromRange").val(commify(str1)+'-'+commify(endN));
    } else {
      $("#RC-chromRange").val('');
    }
    if( plotParams.numPoints != plotStats.numPoints ) {
      $('#RC-numBars').text('('+plotStats.numPoints+')');
    } else {
      $('#RC-numBars').text('');
    }
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

  function absFormat(val, axis) {
    return ''+Math.abs(val.toFixed(axis.tickDecimals));
  }

  function updatePlot() {
    plotData = [];
    if( plotStats.numFields <= 1 ) {
      return;
    }
    options = {
      grid: {minBorderMargin:0, hoverable:true, clickable:true, backgroundColor:"#F8F8F8"},
      selection: {mode:plotParams.zoomMode == 2 ? "xy" : "x"},
      legend: {position:plotParams.barAxis == 0 ? "nw" : "sw"},
      series: {axis:1, bars:{show:true,align:"left"}, line:{show:false}},
      xaxis: {ticks:0, tickLength:0, axisLabel:plotStats.xTitle, axisLabelFontSizePixels:18, min:plotStats.minX, max:plotStats.maxX },
      yaxis: {tickFormatter:absFormat, axisLabelFontSizePixels:16},
      xaxes: {}, yaxes: []
    };
    var nplot = 0;
    var d1 = [];
    var d2 = [];
    var d3 = [];
    var d4 = [];
    var ymin = 0, ymax = 0;
    var pmin = 0, pmax = 0;
    var binned = plotStats.binnedData;
    var f_reads = DataField.fwd_reads;
    var r_reads = DataField.rev_reads;
    var f_ontarg = DataField.fwd_ont;
    var r_ontarg = DataField.rev_ont;
    var chrid = DataField.contig_id;
    var srt = DataField.pos_start;
    var end = DataField.pos_end;
    var chrView = (plotStats.multiChrom && plotStats.numPoints > 1);
    var onTargets = plotStats.onTargets;
    // def_tinyValue is used so zero height bars are visible/selectable at maximum zoom
    for( var i = 0; i < plotStats.numPoints; ++i ) {
      var scale = plotParams.aveBase ? 1.0/(dataTable[i][end]-dataTable[i][srt]+1) : 1;
      if( plotParams.barAxis == 0 ) {
        var reads = scale * (dataTable[i][f_reads]+dataTable[i][r_reads]);
        var ontarg = scale * (dataTable[i][f_ontarg]+dataTable[i][r_ontarg]);
        if( reads == 0 ) ncov = def_tinyValue;
        if( onTargets ) d1.push( [i,ontarg] );
        d2.push( [i,reads] );
        if( reads > ymax ) {
          pmax = ymax;
          ymax = reads;
        } else if( reads > pmax ) {
          pmax = reads;
        } 
      } else {
        var freads = dataTable[i][f_reads] * scale;
        var fontarg = dataTable[i][f_ontarg] * scale;
        var rreads = -dataTable[i][r_reads] * scale;
        var rontarg = -dataTable[i][r_ontarg] * scale;
        if( freads == 0 ) freads = def_tinyValue;
        if( rreads == 0 ) rreads = -def_tinyValue;
        if( onTargets ) d1.push( [i,fontarg] );
        if( onTargets ) d2.push( [i,rontarg] );
        d3.push( [i,freads] );
        d4.push( [i,rreads] );
        if( freads > ymax ) {
          pmax = ymax;
          ymax = freads;
        } else if( freads > pmax ) {
          pmax = freads;
        }
        if( rreads < ymin ) {
          pmin = ymin;
          ymin = rreads;
        } else if( rreads < pmin ) {
          pmin = rreads;
        }
      }
    }
    // add axis labels if in multiple contigs view and not too many labels
    if( chrView && plotStats.chromsInView <= plotStats.maxXaxisLabels ) {
      var xticks = [];
      var lastChr = '';
      var lastX = 0;
      var chrN = 0;
      //var forms = '<span style="font-size:9px;line-height:7px">';
      //var formU = '<span style="font-size:9px;padding-top:0px">';
      //var formL = '<span style="font-size:9px;padding-top:7px">';
      var formU = '<span style="font-size:9px;line-height:1px;position:relative:top:-10px">';
      var formL = '<span style="font-size:9px;line-height:1px;padding-top:-5px"><br/>';
      var formE = '</span>';
      for( var i = 0; i < plotStats.numPoints; ++i ) {
        var chr = dataTable[i][chrid];
        if( chr != lastChr ) {
          lastChr = chr;
          var label = chr;
          var dtick = (i - lastX) * 200 / plotStats.numPoints;
          lastX = i;
          // staggered labels - cludgy as do not know actual sizes (until after drawing!)
          //if( (chrN % 2) == 1 && dtick <= 2+chr.length ) label = '<br>'+chr;
          label = ((chrN % 2) == 1 && dtick <= 2+chr.length ? formL : formU) + chr + formE;
          xticks.push( [i+0.5,label] );
          ++chrN;
        }
      }
      options.xaxis.ticks = xticks;
      options.xaxis.tickLength = 4;
    }
    // check for outlier and avoid if specified by user
    if( ymax > def_outlierFactor*pmax || ymin < def_outlierFactor*pmin ) {
      // if not yet displayed display (unchecked first if not already checked)
      if( $('#RC-offScaleOutlierControl').css("display") == 'none' ) {
        $('#RC-offScaleOutlierControl').show();
      }
      if( $('#RC-offScaleOutlier').attr('checked') ) {
        if( ymax > def_outlierFactor*pmax ) {
          ymax = 1.05 * pmax;
        } else {
          ymin = 1.05 * pmin;
        }
      }
    } else {
      // if no outlier, remove option and reset to unchecked
      $('#RC-offScaleOutlierControl').hide();
      $('#RC-offScaleOutlier').attr('checked', (plotParams.offScaleOutlier = false) );
    }
    ymin = roundAxis(ymin);
    ymax = roundAxis(ymax);
    if( plotStats.zoomChrom && !plotParams.resetYScale ) {
      // always adjust zoom if max/min increase due to re-binning
      if( ymin < plotStats.minY ) plotStats.minY = ymin;
      if( ymax > plotStats.maxY ) plotStats.maxY = ymax;
      if( plotParams.zoomMode != 1 ) {
         ymin = plotStats.minY;
         ymax = plotStats.maxY;
      }
    } else {
      plotParams.resetYScale = false;
      plotStats.minY = ymin;
      plotStats.maxY = ymax;
    }

    if( plotParams.barAxis == 0 ) {
      if( onTargets ) {
        plotData.push( { label: LegendLabels.offReads, color: ColorSet.offReads, data: d2 } );
        plotData.push( { label: LegendLabels.ontReads, color: ColorSet.ontReads, data: d1 } );
      } else {
        plotData.push( { label: LegendLabels.allReads, color: ColorSet.allReads, data: d2 } );
      }
    } else {
      if( onTargets ) {
        plotData.push( { label: LegendLabels.fwdOffReads, color: ColorSet.fwdOffReads, data: d3 } );
        plotData.push( { label: LegendLabels.fwdOntReads, color: ColorSet.fwdOntReads, data: d1 } );
        plotData.push( { label: LegendLabels.revOffReads, color: ColorSet.revOffReads, data: d4 } );
        plotData.push( { label: LegendLabels.revOntReads, color: ColorSet.revOntReads, data: d2 } );
      } else {
        plotData.push( { label: LegendLabels.fwdReads, color: ColorSet.fwdOffReads, data: d3 } );
        plotData.push( { label: LegendLabels.revReads, color: ColorSet.revOffReads, data: d4 } );
      }
    }
    var ytitle = plotParams.aveBase ? "Base Read Depth" : "Total Base Reads";
    options.yaxes.push( {position:"left", axisLabel:ytitle, min:ymin, max:ymax} );
    options.legend.show = plotParams.showLegend;
    plotStats.tooltipZero = 0.01*(ymax-ymin);
    ++nplot;

    // Add 2nd yaxis plot
    if( plotParams.overPlot > 0 ) {
      pointMap = []; // necessary to map to none overlayed bins
      var aLabel, pLabel, pColor;
      var d3 = [];
      if( plotParams.overPlot == 1 ) {
        aLabel = "Reads On-Target";
        pLabel = LegendLabels.precentOntarg;
        pColor = ColorSet.precentOntarg;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          var totalReads = dataTable[i][f_reads] + dataTable[i][r_reads];
          if( totalReads <= 0 ) continue;
          var fwdReads = dataTable[i][f_ontarg] + dataTable[i][r_ontarg];
          var pcFwd = totalReads > 0 ? 100 * fwdReads / totalReads : 0;
          d3.push( [(i+0.5),pcFwd] );
          pointMap.push(i);
        }
      } else {
        aLabel = "Forward Reads";
        pLabel = LegendLabels.fwdBias;
        pColor = ColorSet.fwdBias;
        for( var i = 0; i < plotStats.numPoints; ++i ) {
          var fwdReads = dataTable[i][f_reads];
          var totalReads = fwdReads + dataTable[i][r_reads];
          if( totalReads <= 0 ) continue;
          var pcFwd = totalReads > 0 ? 100 * fwdReads / totalReads : 50;
          d3.push( [(i+0.5),pcFwd] );
          pointMap.push(i);
        }
      }
      plotData.push( {
        label: pLabel, color: pColor, data: d3, yaxis: 2, bars: {show:false}, points: {show:true}, shadowSize: 0 } );
      options.yaxes.push( {position:"right", axisLabel:aLabel, min:0, max:100, tickFormatter: percentFormat} );
      options.grid.aboveData = true;
      if( plotParams.overPlot == 2 ) {
        options.grid.markings = [ {color: pColor, linewidth: 1, y2axis: {from:50,to:50}} ];
      }
      ++nplot;
    }

    plotStats.numPlots = nplot;
    hideTooltip();
    plotObj = $.plot(placeholder, plotData, options);
    canvas = plotObj.getCanvas();
  }

  // start up customization attributes
  if( startShowOptions )
    $('#RC-controlpanel').show();
  if( startOutlierOffScale )
    $('#RC-offScaleOutlier').attr('checked', (plotParams.offScaleOutlier = true) );
  unzoomToFile(chrcovFile);
  autoHideLegend();

  // automatically change initial view to full contig if only one
  if( plotStats.totalChroms == 1 ) {
    $("#RC-unzoomToggle").click();
  } else if( plotStats.totalChroms > plotStats.maxXaxisLabels ) {
    // disable loads from whole genome view if too many contigs (how many?)
    setUnzoomTitle(false);
    wgncovFile = '';
  }

  // collapse view after EVRYTHING has been drawn in open chart (to avoid flot issues)
  if( startCollapsed ) {
    $("#RC-collapsePlot").attr("class","ui-icon ui-icon-triangle-1-s");
    $("#RC-collapsePlot").attr("title","Expand view");
    $('#RC-controlpanel').hide();
    $('.RC-shy').hide();
    $('#RC-chart').resizable('destroy');
    $('#RC-noncanvas').hide();
  }
});
