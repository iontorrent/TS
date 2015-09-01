// html for chart container and filters bar - note these are invisible and moved into position later
document.write('\
<div id="DOC-chart" class="unselectable" style="border:2px solid #666;page-break-inside:avoid;display:none">\
  <div id="DOC-titlebar" class="grid-header" style="min-height:24px;border:0">\
    <span id="DOC-collapsePlot" style="float:left" class="ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Depth of Coverage Chart</span>\
    <span class="DOC-shy flyhelp" id="DOC-plotLabel" style="padding-left:20px">Plot:</span>\
    <select class="DOC-shy txtSelect" id="DOC-graphType">\
      <option value="Full">Maximum Read Depth</option>\
      <option value="View" selected="selected">99.9% of All Reads</option>\
      <option value="Mean">Normalized Coverage</option>\
    </select>\
    <input class="DOC-shy" id="DOC-unzoomToggle" type="button" value="Zoom Out" style="margin-left:10px;width:70px">\
    <span id="DOC-toggleControls" style="float:right" class="DOC-shy ui-icon ui-icon-search" title="Show/Hide the view options panel"></span>\
    <span id="DOC-help" style="float:right;margin-left:0;margin-right:0" class="DOC-shy ui-icon ui-icon-help"></span>\
    <span id="DOC-message" class="message"></span>\
  </div>\
  <div id="DOC-noncanvas" style="background:#EEF;border-top:2px solid #666">\
   <div id="DOC-plotspace" style="padding:4px">\
    <div id="DOC-placeholder" style="width:100%"></div>\
    <div id="DOC-cursorCoords" style="color:#bbd"></div>\
   </div>\
  </div>\
</div>\
<div id="DOC-controlpanel" class="filter-panel" style="display:none">\
  <table><tr>\
    <td class="nwrap">Viewing Options:</td>\
    <td class="nwrap"><span class="flyhelp" id="DOC-autoZoomLabel">Automatic Zoom</span>:\
      <input type="checkbox" id="DOC-autoZoom" checked="checked"></td>\
    <td class="nwrap"><span class="flyhelp" id="DOC-showLegendLabel">Show Legend</span>:\
      <input type="checkbox" id="DOC-showLegend" checked="checked"></td>\
    <td class="nwrap DOC-hideFilter"><span class="flyhelp" id="DOC-include0xLabel">Include 0x Coverage</span>:\
      <input type="checkbox" id="DOC-include0x" checked="checked"></td>\
    <td class="nwrap DOC-hideFilter"><span class="flyhelp" id="DOC-binSizeLabel">Bin Size</span>:\
      <input type="text" class="numSearch" id="DOC-binSize" value=10 size=4>&nbsp;<span id="DOC-binSizeUsed"></span>\
   </tr></table>\
</div>\
<div id="DOC-tooltip" style="display:none">\
  <div><span id="DOC-tooltip-close" title="Close" class="help-box ui-icon ui-icon-close"></span></div>\
  <div id="DOC-tooltip-body"></div>\
</div>\
<div id="DOC-helptext" class="helpblock" style="display:none">\
This chart shows the distribution of targeted base coverage.<br/><br/>\
The plot area may be re-sized by dragging the borders of the chart or hidden using<br/>\
the Collapse View button in the upper right corner.<br/><br/>\
Moving the mouse over data bar or point in the graph area will show some minimal<br/>\
information about the data plotted. Clicking on the same data will produce a more<br/>\
detailed information box that remains until dismissed. (For the Normalized Coverage<br/>\
plot the pointer coordinates are shown at the bottom left instead of data point help.)<br/><br/>\
Click and drag the mouse to select a region in the plot space to zoom in. Use the<br/>\
Zoom Out button (or double-click in space in the plot area) to return to the full view.<br/><br/>\
You may change how the data is viewed using the Plot selector and controls on the<br/>\
options panel, opened using the search icon in the title bar.<br/><br/>\
Look for additional fly-over help on or near the controls provided.\
</div>\
');

$(function () {

  // check placer element exists
  if( !$('#DepthOfCoverageChart').length ) return;

  // check browser environment
  var fixIE = (typeof G_vmlCanvasManager != 'undefined');
  var useFlash = (typeof FlashCanvas != 'undefined');
  var useExCan = (fixIE && !useFlash);

  // minimum sizes for chart widget
  var def_minWidth = 625;
  var def_minHeight = 200; // includes title bar and control panel, if open

  // configure widget size and file used from placement div attributes
  var coverageFile = $("#DepthOfCoverageChart").attr("datafile");
  if( coverageFile == undefined || coverageFile == "" ) {
    //alert("ERROR on page: DepthOfCoverageChart widget requires attribute 'datafile' is set.");
    $('#DepthOfCoverageChart').hide();
    return;
  }

  var startCollapsed = $("#DepthOfCoverageChart").attr("collapse");
  startCollapsed = (startCollapsed != undefined);

  var tmp = $('#DepthOfCoverageChart').width();
  if( tmp < def_minWidth ) tmp = def_minWidth;
  $("#DOC-chart").width(tmp);
  tmp = $('#DepthOfCoverageChart').height();
  if( tmp < def_minHeight ) tmp = def_minHeight;
  $("#DOC-chart").height(tmp);
  $("#DOC-placeholder").height(tmp-36);
  $("#DepthOfCoverageChart").css('height','auto');

  $("#DOC-controlpanel").appendTo('#DOC-titlebar');
  $("#DOC-chart").appendTo('#DepthOfCoverageChart');
  $("#DOC-chart").show();
  $('#DOC-chart').css("height","auto");

  // some default values for plot display
  var disableTitleBar = false;
  var cursorCoords = $('#DOC-cursorCoords');
  var placeholder = $("#DOC-placeholder");

  var maxLoadFields = 3;  // to reduce amount fo data loaded for plot
  var dblclickUnzoomFac = 10;

  var resiz_def = {
    alsoResize: "#DOC-placeholder",
    minWidth:def_minWidth,
    minHeight:def_minHeight,
    handles:"e,s,se",
    resize:function(e,u){ updatePlot(false); }
  };
  $('#DOC-chart').resizable(resiz_def);

  $("#DOC-collapsePlot").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#DOC-plotspace').is(":visible") ) {
      $(this).attr("class","ui-icon ui-icon-triangle-1-s");
      $(this).attr("title","Expand view");
      $('#DOC-controlpanel').slideUp();
      $('.DOC-shy').fadeOut(400);
      $('#DOC-chart').resizable('destroy');
      $('#DOC-noncanvas').slideUp('slow');
      hideTooltip();
    } else {
      $(this).attr("class","ui-icon ui-icon-triangle-1-n");
      $(this).attr("title","Collapse view");
      $('.DOC-shy').fadeIn(400);
      $('#DOC-noncanvas').slideDown('slow',function(){
        $('#DOC-chart').resizable(resiz_def);
      });
    }
    $("#DOC-chart").css('height','auto');
  });

  $("#DOC-toggleControls").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#DOC-controlpanel').is(":visible") ) {
      $('#DOC-controlpanel').slideUp();
    } else {
      $('#DOC-controlpanel').slideDown();
    }
    $('#DOC-chart').css("height","auto");
    cursorCoords.appendTo('#DOC-placeholder');
  });

  $("#DOC-help").click( function() {
    var offset = $("#DOC-help").offset();
    var ypos = offset.left - $('#DOC-helptext').width();
    $("#DOC-help").removeAttr("title");
    $('#DOC-helptext').css({
      position: 'absolute', display: 'none',
      top: offset.top+16, left: ypos+8
    }).appendTo("body").slideDown();
  });

  $("#DOC-help").hover( null, function() {
    $('#DOC-helptext').fadeOut(200);
    $("#DOC-help").attr( "title", "Click for help." );
  });

  $('#DOC-placeholder').dblclick(function(e) {
    if( !plotStats.zoom || plotStats.numPlots <= 0 ) return;
    if( lastHoverBar.binNum < 0 ) updatePlot(true);
  });

  //$('#DOC-chart').noContext();

  function rightClickMenu(e) {
    alert("r-click");
  }

  // attempt to disable defalt context menu and enable r-click
  if( useFlash ) {
    // only works for flashcanvas pro!
    FlashCanvas.setOptions( {disableContextMenu : true} );
    // instead use area outside of canvas
    //$('#DOC-noncanvas').noContext();
    //$('#DOC-noncanvas').rightClick(rightClickMenu);
  } else {
    //$('#DOC-chart').noContext();
    //$('#DOC-chart').rightClick(rightClickMenu);
  }

  var plotStats = {
    defaultPlot : "",
    numFields : 0,
    numPlots : 0,
    numPoints : 0,
    binSizeDef : 1,
    zoom: false,
    minX : 0,
    maxX : 0,
    rangeX : 0,
    meanX : 1,
    cov999x : 0,
    minY : 0,
    maxY : 0,
    binSize : 1,
    barScale : 1,
    plotScale : 1,
    axisRatio : 1,
    tooltipZero : 0
  };

  var plotParams = {
    resetYScale: false,
    readAxis : 1,
    binSize : 1,
    showLegend : true,
    include0x : true,
    zoomMode : 1,
    barAxis : 0,
    barType : 1,
    plotAxis : 1,
    plotType : 1
  };

  var LegendLabels = {
    readDepth : "Base Read Depth",
    covCount : "Bases",
    cumCover : "Cumulative Bases"
  };

  function customizeChart() {
    // add fly-over help to controls here in case need to customize for chart data
    $("#DOC-plotLabel").attr( "title",
      "Select the how the depth of coverage data is plotted.\n"+
      "'Maximum Read Depth' specifies to plot coverage out to the maximum depth. This is the default "+
      "setting if the there is a significant coverage out at the largest read depths.\n"+
      "'99.9% of All Reads' specifies to plot up to 99.9% of the total reads from 0x, i.e. up to where the "+
      "Cumulative Bases curve cuts the right y-axis at 0.01%. This view is useful to avoid outlier read "+
      "depths that add would otherwise make the majority of distribution squeezed up against the y-axis.\n"+
      "This is the default plot if 99.9% of the reads fall in the first half of the full range of read depths.\n"+
      "'Normalized Coverage' uses a depth axis that is normalized (divided) by the average read depth "+
      "and plotted to twice the average coverage depth. This plot is useful for generalized analysis of the "+
      "coverage when comparing different experiments or reading off coverage at some fraction of the mean value." );
    $("#DOC-unzoomToggle").attr( "title",
      "Click this button to 'zoom out' the view to show coverage over the whole read depth range in the view "+
      "specified by the Plot selector. (Has no effect if the view was not previously zoomed-in.)" );
    $("#DOC-autoZoomLabel").attr( "title",
      "Select how the y-axis zoom works when selecting a set of read depths (x-axis data) to zoom in to view. " +
      "When checked the y-axis range will be automatically set to the largest read count "+
      "of the read depths currently in view. This mode is particularly useful to magnify the view for read depths "+
      "with relatively low representation. When unchecked the range is set to the largest read count for any "+
      "read depth, regardless of the current data in view.\n(The right-hand y-axis scale is fixed to 0% - 100%.)" );
    $("#DOC-showLegendLabel").attr( "title", "Select whether the legend is displayed over the plot area." );
    $("#DOC-binSizeLabel").attr( "title",
      "Sets the size (width) of the data bins for counting read depth. Using larger bins typically allows the "+
      "distribution of read counts for ranges of read depths to be more easily visualized or interactively "+
      "assessing finer or broader distributions of target coverage. An appropriate default value for Bin Size "+
      "is determined based on the maximum read depth and this value is reset whenever the value for 'Plot' is "+
      "changed or by attempting to set the value for Bin Size empty. Additionally the value set may be temporarily "+
      "ignored if binning size is too small relative to the amount of data in view. (The value employed will then "+
      "be shown in parentheses.) You may also set the Bin Size to 0 to see base coverage plotted as a line rather than bars." );
    $("#DOC-include0xLabel").attr( "title",
      "Uncheck this value to have the 0x coverage ignorred in the plotted data. This will cause the 0x read depth data, e.g. "+
      "the number of target bases that had no reads, to be ignorred in the display. The read depth will be binned starting "+
      "1x rather than 0x so that at Bin Size 1 no data point will be plotted or at Bin Size 10 the first bin will "+
      "represent 1x-10x read depths rather than 0x-9x read depths. This is useful where you are just interested in coverage "+
      "over target bases that were covered, especially if the number of missed targets was high and the 0x data pushes the "+
      "coverage distribution curve towards the x-axis." );
    $("#DOC-help").attr( "title", "Click for help." );
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

  var lastHoverBar = {binNum:-1, isRev:false, clickItem:null, postZoom:false, sticky:false, label:'' };

  $('#DOC-snapShot').click(function() {
    if( canvas != null ) {
      canvas2png(canvas);
    }
  });

  // Uses button names for type string
  function standardPlot(type) {
    if( plotStats.numFields <= 1 ) return;
    plotParams.barType = 1;  // bar
    plotParams.barAxis = 1;  // %
    plotParams.plotType = 1; // line
    plotParams.include0x = true;
    plotParams.showLegend = true;
    if( type == "Full" ) {
      $('.DOC-hideFilter').show();
      plotParams.readAxis = 0;
    } else if( type == "View" ) {
      plotParams.readAxis = 1;
      $('.DOC-hideFilter').show();
    } else if( type == "Mean" ) {
      $('.DOC-hideFilter').hide();
      plotParams.readAxis = 2;
      plotParams.barType = 0; // hidden
      plotParams.showLegend = false;
    }
    plotParams.binSize = autoBinSize();
    updateBinSizeUsed(plotParams.binSize);
    updateGUIPlotParams();
    updatePlotTitles();
    updatePlot(true);
  }

  function autoBinSize() {
    if( plotParams.readAxis > 1 ) return 1;
    if( plotParams.readAxis == 1 && 2 * plotStats.cov999x <= plotStats.numPoints ) {
       return roundBinSize(plotStats.cov999x);
    }
    return plotStats.binSizeDef;
  }

  function updateGUIPlotParams() {
    $('#DOC-autoZoom').attr('checked',(plotParams.zoomMode == 1));
    $('#DOC-binSize').val(plotParams.binSize);
    $('#DOC-showLegend').attr('checked',plotParams.showLegend);
    $('#DOC-include0x').attr('checked',plotParams.include0x);
  }
 
  $('#DOC-autoZoom').change(function() {
    plotParams.zoomMode = ($(this).attr("checked") == "checked") ? 1 : 0;
    updatePlot(false);
  });

  $('#DOC-binSize').change(function() {
    var val = $.trim(this.value);
    if( val == '' ) {
      this.value = plotParams.binSize = autoBinSize();  // reset to auto
    } else if( isNaN(val) || val < 0 ) {
      this.value = plotParams.binSize; // reset to current value
    } else {
      this.value = plotParams.binSize = Math.floor(val);
    }
    plotParams.resetYScale = true;
    updatePlot(false);
  });

  $('#DOC-graphType').change(function() {
    standardPlot(this.value);
  });

  $("#DOC-unzoomToggle").click(function() {
    if( plotStats.zoom ) updatePlot(true);
  });

  $('.DOC-selectParam').change(function() {
    plotParams[this.id] = parseInt(this.value);
    updatePlotTitles();
    updatePlot( this.id == "readAxis" );
  });

  $('#DOC-showLegend').change(function() {
    plotParams.showLegend = ($(this).attr("checked") == "checked");
    updatePlot(false);
  });

  $('#DOC-include0x').change(function() {
    plotParams.include0x = ($(this).attr("checked") == "checked");
    updatePlot(false);
  });

  function unzoomToFile(filename) {
    loadTSV(filename);
    updatePlotStats();
    $('#DOC-graphType').val(plotStats.defaultPlot);
    standardPlot(plotStats.defaultPlot);
  }

  function loadTSV(tsvFile) {
    dataTable = [];
    $('#DOC-message').text('Loading...');
    $.ajaxSetup( {dataType:"text",async:false} );
    $.get(tsvFile, function(mem) {
      var lines = mem.split("\n");
      var exptDepth = 0;
      $.each(lines, function(n,row) {
        var fields = $.trim(row).split('\t').slice(0,maxLoadFields);
        if( n == 0 ) {
          fieldIds = fields;
        } else if( fields[0] != "" ) {
          // important to convert numeric fields to numbers for performance
          for( var i = 0; i < fields.length; ++i ) { fields[i] = +fields[i]; }
          // fill data that appears to be missing - typically derived fields are not even read in
          if( fields[0] > exptDepth ) {
            var mfields = fields.slice();  // copy
            mfields[1] = 0; // no reads at depth
            while( exptDepth < fields[0] ) {
              mfields[0] = exptDepth++;
              dataTable.push( mfields );
            }
          }
          dataTable.push( fields );
          exptDepth = fields[0]+1;
        }
      });
    }).error(function(){
      alert("An error occurred while loading from "+tsvFile);
      $('#DOC-message').text('');
    }).success(function(){
      $('#DOC-message').text('');
    });
  }

  function updatePlotTitles() {
    fieldIds[0] = LegendLabels.readDepth;
    if( plotParams.readAxis == 2 ) fieldIds[0] = "Normalized " + fieldIds[0];
    if( plotParams.readAxis == 3 ) fieldIds[0] = "100x Normalized " + fieldIds[0];
    fieldIds[1] = LegendLabels.covCount; // + " (" + (plotParams.barAxis ? "% Bases" : "Count") + ")";
    fieldIds[2] = LegendLabels.cumCover; // + " (" + (plotParams.plotAxis ? "% Bases" : "Count") + ")";
  }

  function roundBinSize(numPoints) {
    var binSize = 5*Math.floor(0.2+numPoints/500);
    if( binSize >= 5000 ) binSize = 5000*Math.floor(0.25+0.0002*binSize);
    else if( binSize >= 1000 ) binSize = 1000*Math.floor(0.25+0.001*binSize);
    else if( binSize >= 500 ) binSize = 500*Math.floor(0.25+0.002*binSize);
    else if( binSize >= 100 ) binSize = 100*Math.floor(0.25+0.01*binSize);
    else if( binSize >= 50 ) binSize = 50*Math.floor(0.25+0.02*binSize);
    else if( binSize >= 25 ) binSize = 25*Math.floor(0.25+0.04*binSize);
    else if( binSize >= 10 ) binSize = 10*Math.floor(0.25+0.1*binSize);
    else if( binSize < 1 ) binSize = numPoints > 200 ? 2 : 1;
    return binSize;
  }

  function updatePlotStats() {
    plotStats.numPoints = 0;
    plotStats.numFields = dataTable.length > 0 ? fieldIds.length : 0;
    if( plotStats.numFields <= 1 ) return;

    // plot range anticipates sparse file (no zero coverage rows)
    plotStats.numPoints = dataTable[dataTable.length-1][0]+1;
    plotStats.minX = 0;
    plotStats.maxX = plotStats.numPoints;
    plotStats.binSizeDef = roundBinSize(plotStats.numPoints);
    updatePlotTitles();

    // mean for normalized x-axis and scaling for % y-axes
    var sumc = 0, sumd = 0, cd = 0;
    var maxBar = 0, maxPlot = 0;
    for( var i = 0; i < dataTable.length; ++i ) {
      sumc += dataTable[i][1];
      sumd += dataTable[i][0] * dataTable[i][1];
      if( dataTable[i][1] > maxBar ) maxBar = dataTable[i][1];
      if( plotStats.numFields > 2 ) {
        if( dataTable[i][2] > maxPlot ) maxPlot = dataTable[i][2];
      }
    }
    plotStats.meanX = sumc > 0 ? sumd / sumc : 1;
    plotStats.barScale = sumc > 0 ? 100/sumc : 1;
    plotStats.plotScale = maxPlot > 0 ? 100/maxPlot : 1;
    plotStats.minY = 0;
    plotStats.maxY = maxBar;

    // 99.9% coverage read depth limit for view
    plotStats.cov999x = plotStats.numPoints;
    if( plotStats.numFields > 2 ) {
      for( var i = dataTable.length-1; i >= 0; --i ) {
        if( dataTable[i][2] * plotStats.plotScale >= 0.1 ) {
          plotStats.cov999x = dataTable[i][0];
          break;
        }
      }
    }
    // avoid the solid block output
    if( plotStats.cov999x < 5 ) {
      ++plotStats.cov999x;
    }
    // set the default plot based on whether the 99.9% view is merited
    if( plotStats.cov999x * 2 < plotStats.numPoints ) {
      plotStats.defaultPlot = "View";
    } else {
      plotStats.defaultPlot = "Full";
    }
  }

  function roundAxis( maxVal ) {
    var b = Math.pow( 10, Math.round(Math.log(maxVal)/Math.LN10)-1 );
    return b * Math.floor(1+maxVal/b);
  }

  function percentFormat(val, axis) {
    return ''+val.toFixed(axis.tickDecimals)+'%';
  }

  placeholder.bind("plotselected", function (event, ranges) {
    plotStats.zoom = true;
    if( plotParams.readAxis == 2 ) {
      plotStats.minX = options.xaxis.min = ranges.xaxis.from;
      plotStats.maxX = options.xaxis.max = ranges.xaxis.to;
      options.xaxis.tickDecimals = null;
    } else {
      plotStats.minX = options.xaxis.min = Math.floor(ranges.xaxis.from);
      plotStats.maxX = options.xaxis.max = Math.ceil(ranges.xaxis.to);
      options.xaxis.tickDecimals = 0;
    }
    if( plotStats.maxX > plotStats.numPoints ) {
      // avoid round up errors 
      plotStats.maxX = options.xaxis.max = plotStats.numPoints;
    }
    if( plotParams.zoomMode == 2 ) {
      //plotStats.minY = options.yaxes[0].min = ranges.yaxis.from;
      plotStats.maxY = options.yaxes[0].max = ranges.yaxis.to;
      if( plotStats.numPlots > 1 ) {
        //options.yaxes[1].min = options.yaxes[0].min * plotStats.axisRatio;
        options.yaxes[1].max = options.yaxes[0].max * plotStats.axisRatio;
      }
    }
    updatePlot(false);
  });

  function updateBinSizeUsed(binSize) {
    plotStats.binSize = binSize;
    if( binSize != plotParams.binSize ) {
      $('#DOC-binSizeUsed').text('('+binSize+')');
    } else {
      $('#DOC-binSizeUsed').text('');
    }
  }

  var covAtLastMap = 0;
  function coverageAt(x,d) {
    // Return the coverage given data in sparse array
    // Not intended for random access - must start with a reset (x <= 0) and ask for increasing (or same) x positions
    if( x <= 0 ) {
      covAtLastMap = x = 0;
    }
    // search for next >= depth data record
    while( dataTable[covAtLastMap][0] < x ) {
      if( ++covAtLastMap >= dataTable.length ) {
        covAtLastMap = dataTable.length - 1;
        break;
      }
    }
    // the actual position is what was requested, not in the data
    if( d < 1 ) return x;
    if( d == 1 ) {
      // assumes first data row is number of reads at this depth
      return x < dataTable[covAtLastMap][0] ? 0 : dataTable[covAtLastMap][1];
    }
    // assumes all other values are cumulative
    return dataTable[covAtLastMap][d];
  }

  function coverageLast(d) {
    // Return data for last item
    return dataTable[dataTable.length-1][d];
  }

  // Creates and renders a new Plot() given the current plot parameters
  // if reset == false then plot is updated at current zoom, else rest to full view
  // side effects: set plotObj/canvas references, plotStats.(numPlots,axisRatio)
  function updatePlot(reset) {
    if( reset && plotStats.numFields > 1 )
    {
       if( plotParams.readAxis == 1 ) {
         plotStats.minX = 0;
         plotStats.maxX = plotStats.cov999x;
       } else if( plotParams.readAxis == 2 ) {
         plotStats.minX = 0;
         plotStats.maxX = 2.0;
       } else if( plotParams.readAxis == 3 ) {
         plotStats.minX = 0;
         plotStats.maxX = 200;
       } else {
         plotStats.minX = 0;
         plotStats.maxX = plotStats.numPoints;
       }
       plotStats.zoom = false;
    }
    plotData = [];
    options = {
      grid: {minBorderMargin:0, hoverable:true, clickable:true, backgroundColor:"#F8F8F8"},
      selection: {mode:plotParams.zoomMode == 2 ? "xy" : "x"},
      xaxes: {tickDecimals:0,minTickSize:1}, yaxes: [],
      xaxis: {}, yaxis: {tickFormatter:null}, legend:{}
    };
    if( plotStats.numFields > 0 ) {
      options.xaxis = { axisLabel: fieldIds[0], axisLabelFontSizePixels: 18 };
      options.yaxis = { axisLabelFontSizePixels: 16 };
    }
    if( plotParams.readAxis == 2 ) {
      if( !plotStats.zoom ) {
        options.xaxis.tickSize = 0.1;
        options.xaxis.tickDecimals = 1;
        options.yaxis.tickSize = 10;
      }
    } else {
      options.xaxis.minTickSize = 1;
      options.xaxis.tickDecimals = 0;
    }
    // plot-type dependent flags
    var skip0x = !plotParams.include0x;
    var xscale = plotParams.readAxis >= 2 ? 1.0/plotStats.meanX : 1;
    var barType = plotParams.binSize > 0 || plotParams.readAxis >= 2 ? plotParams.barType : 3;
    var binSize = plotParams.binSize > 0 && plotParams.readAxis < 2 ? plotParams.binSize : 1;
    if( plotParams.readAxis == 3 ) xscale *= 100;
    var minXrange = plotStats.minX - xscale * binSize;
    var maxXrange = plotStats.maxX + xscale * binSize;
    // this approximates to 1000 points displayed (vs. default of ~100 bins for full range)
    var minBinSize = roundBinSize( 0.1 * (maxXrange - minXrange) / xscale );
    if( binSize < minBinSize ) binSize = minBinSize;
    updateBinSizeUsed(binSize);
    var fullLYaxis = (plotParams.readAxis < 2 || plotParams.zoomMode == 0 || !plotStats.zoom);
    var nplot = 0;
    var barMax = 0, axisRatio = 1;
    // TO DO: Refactor - code is a bit of a mess since originally it plotted any number of graphs
    var numFields = 3; // plotStats.numFields to plot all fields agains the first
    for( var sn = 1; sn < numFields; ++sn ) {
      var xstep = binSize;
      var yscale = 1;
      var ytform = '%s';
      var barBin = false;
      if( sn == 1 ) {
        barBin = (xstep > 1);
        if( plotParams.barAxis > 0 ) {
          ytform = percentFormat;
          yscale = plotStats.barScale;
        }
      } else {
        if( plotParams.plotType == 0 ) continue;
        //if( plotParams.plotType == 1 ) xstep = 1;
        if( plotParams.plotAxis > 0 ) {
          ytform = percentFormat;
          yscale = plotStats.plotScale;
        }
      }
      var series = { label: fieldIds[sn], yaxis: ++nplot, data: [], shadowSize: 0 };
      var ymin = -1, ymax = 0, yval = 0;
      var i = skip0x ? 1 : 0;
      coverageAt(0,0); // reset tracker
      for( ; i < plotStats.numPoints; i += xstep ) {
        if( barBin ) {
          var sum = 0;
          var srt = barType == 2 ? i-Math.floor(xstep/2) : i;
          var lmt = srt+xstep;
          if( lmt > plotStats.numPoints ) { lmt = plotStats.numPoints; }
          for( var j = srt < 0 ? 0 : srt; j < lmt; ++j ) {
            sum += coverageAt(j,sn);
          }
          yval = yscale * sum;
        } else {
          yval = yscale * coverageAt(i,sn);
        }
        // need local max. for automatic mode
        var x = xscale * coverageAt(i,0);
        if( plotParams.zoomMode == 0 || (x > minXrange && x < maxXrange) ) {
          if( yval > ymax ) ymax = yval;
          if( yval < ymin || ymin < 0 ) ymin = yval;
          series.data.push( [ x, yval ] );
        }
      }
      if( sn == 1 ) {
         if( ymax == 0 ) ymax = 0.000001;
        barMax = ymax = roundAxis(ymax);
        if( plotStats.zoom && !plotParams.resetYScale ) {
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
        if( barType == 0 ) {
          series.lines = { show: false };
          series.label = null;
          hideBar = true;
        } else if( barType >= 3 ) {
          series.lines = { show: true };
          if( barType == 4 ) { series.points = { show:true }; }
        } else {
          series.bars = { show: true, barWidth: xstep * xscale, align: barType == 2 ? "center" : "left" };
        }
      } else {
        if( fullLYaxis ) ymax = 100;
        if( barMax > 0 ) axisRatio = ymax/barMax;
        if( nplot > 1 ) series.shadowSize = 4;
        if( plotParams.plotType & 1 ) series.lines = { show:true };
        if( plotParams.plotType & 2 ) series.points = { show: true };
        // add in the last point when skipped due to bin size
        if( i-xstep < coverageLast(0) ) {
          series.data.push( [ xscale * coverageLast(0), yscale * coverageLast(sn) ] );
        }
      }
      plotData.push( series );
      var axisPos = (nplot%2) == 1 || barType == 0;
      // always use 0 base if at full view, non-automatic zoom, or a bar plot
      if( fullLYaxis ) ymin = 0;
      options.yaxes.push( {position:axisPos ? "left" : "right", min:ymin, max:ymax, tickFormatter:ytform} );
      options.yaxes[nplot-1].axisLabel = fieldIds[sn];
      if( sn == 1 && barType == 0 ) {
        options.yaxes[0].show = false;
      }
    }
    if( barType == 0 ) --nplot;
    options.legend.show = plotParams.showLegend;
    options.xaxis.min = plotStats.minX;
    // adjust x axis to account for binning
    options.xaxis.max = (binSize == 1 || plotParams.readAxis == 2) ? plotStats.maxX : binSize * Math.floor(1+plotStats.maxX/binSize);
    plotStats.numPlots = nplot;
    plotStats.axisRatio = axisRatio;
    plotStats.tooltipZero = 0.01*(plotStats.maxY-plotStats.minY);

    hideTooltip();
    plotObj = $.plot(placeholder, plotData, options);
    canvas = plotObj.getCanvas();
  }

  placeholder.bind("plothover", function(event, pos, item) {
    // show cursor coords for normalized plots
    if( plotParams.readAxis >= 2 ) {
      if( plotObj ) plotObj.unhighlight();
      setCursor( showCursorCoords(pos.x,pos.y,pos.y2) ? 'crosshair' : 'default' );
      return;
    }
    var hoverTip = !lastHoverBar.sticky;
    if( cursorOverItem(pos,item) ) {
      setCursor('pointer');
      if( hoverTip ) showTooltip(item,pos,false);
    } else {
      setCursor('crosshair');
      if( hoverTip ) hideTooltip();
    }
  });

  placeholder.bind("plotclick", function(e,pos,item) {
    // ignore false triggering due to mouse selection for zoom
    if( lastHoverBar.postZoom ) {
      lastHoverBar.postZoom = false;
      return;
    }
    // custom: do not highlight if in normalized plot
    if( plotParams.readAxis >= 2 ) {
      if( plotObj ) plotObj.unhighlight();
      return;
    }
    if( cursorOverItem(pos,item) ) {
      showTooltip(item,pos,true);
      lastHoverBar.clickItem = item;
      if( item != null ) plotObj.highlight(item.series,item.datapoint);
    } else {
      hideTooltip();
    }
  });

  placeholder.bind("mouseleave", function() {
    setCursor('default');
  });

  function cursorOverPlot(x,y) {
    var ymin = options.yaxes[0].min;
    var ymax = options.yaxes[0].max;
    return plotStats.numPlots > 0 && x >= plotStats.minX && x < plotStats.maxX && y >= ymin && y <= ymax;
  }

  function cursorOverItem(pos,item) {
    if( pos.x >= plotStats.numPoints ) return false;
    //return item || Math.abs(pos.y) < plotStats.tooltipZero;
    return item != null;
  }

  function hideTooltip() {
    if( plotObj ) plotObj.unhighlight();
    $("#DOC-tooltip").hide();
    lastHoverBar.binNum = -1;
    lastHoverBar.clickItem = null;
    lastHoverBar.sticky = false;
    lastHoverBar.label = '';
  }

  function showTooltip(item,pos,sticky) {
    if( !item ) return;
    var binNum = item.dataIndex;
    var label = item.series.label;
    if( lastHoverBar.binNum == binNum && lastHoverBar.sticky == sticky && lastHoverBar.label == label ) return;
    hideTooltip();
    // custom: disable tooptip for normalized plots
    if( plotParams.readAxis >= 2 ) return;
    lastHoverBar.binNum = binNum;
    lastHoverBar.sticky = sticky;
    lastHoverBar.label = label;
    var bgColor = item.series.color;
    $('#DOC-tooltip-body').html( sticky ? tooltipMessage(item,binNum) : tooltipHint(item,binNum) );
    var posx = pos.pageX+12;
    var posy = pos.pageY-10;
    if( sticky ) {
      var cof = $('#DOC-chart').offset();
      var ht = $('#DOC-tooltip').height();
      var ymax = cof.top + $('#DOC-chart').height() - ht;
      posy = pos.pageY - $('#DOC-tooltip').height()/2;
      if( posy > ymax ) posy = ymax;
      if( posy < cof.top-4 ) posy = cof.top-4;
      var xmid = cof.left + $('#DOC-chart').width()/2;
      if( pos.pageX > xmid ) posx = pos.pageX - $('#DOC-tooltip').width() - 16;
    }
    $('#DOC-tooltip').css({
      position: 'absolute', left: posx, top: posy, maxWidth: 280,
      background: bgColor, padding: '3px '+(sticky ? '7px' : '4px'),
      border: (sticky ? 2 : 1)+'px solid #444',
      opacity: sticky ? 1: 0.7
    }).appendTo("body").fadeIn(sticky ? 10 : 100);
  }

  $('#DOC-tooltip-close').click( function() {
    hideTooltip();
  });

  function dataBar(id) {
    return (id === LegendLabels.covCount);
  }

  function sigfig(val) {
    val = parseFloat(val);
    var av = Math.abs(val);
    if( av == 0 ) return "0";
    if( av >= 100 ) return val.toFixed(0);
    if( av >= 10 ) return val.toFixed(2);
    if( av >= 1 ) return val.toFixed(2);
    if( av >= 0.01 ) return val.toFixed(3);
    return val.toFixed(4);
  }

  // this would need an update if user axis scaling is re-enabled
  function tooltipHint(item,bin) {
    $('#DOC-tooltip-close').hide();
    var id = item.series.label;
    if( dataBar(id) ) {
      return sigfig(item.datapoint[1])+'%';
    }
    if( id === LegendLabels.cumCover ) {
      return sigfig(item.datapoint[1])+'%';
    }
    return '?';
  }

  function tooltipMessage(item,bin) {
    var label = item.series.label;
    var axis = item.series.yaxis;
    var isPc = label.indexOf('%') > 0;
    var j = label.indexOf(' (');
    if( j > 0 ) label = label.substr(0,j);
    var br = "<br/>";

    var skip0x = !plotParams.include0x;
    var binSize = plotStats.binSize;
	var binOff = plotParams.readAxis < 2 ? skip0x + parseInt(plotStats.minX/binSize)*binSize : plotStats.minX;
    var depth;
    if( plotParams.readAxis == 2 ) {
      depth = ((binOff+bin)/plotStats.meanX).toFixed(3);
    } else if( plotParams.readAxis == 3 ) {
      depth = (100*(binOff+bin)/plotStats.meanX).toFixed(1);
    } else if( label == LegendLabels.cumCover ) {
      depth = commify(binOff + bin * binSize);
    } else {
      depth = binOff + bin * binSize;
      if( binSize > 1 ) depth = commify(depth) + (binSize == 2 ? "," : "-") + commify(depth+binSize-1);
    }
    var xtitle = "depth";
    if( binSize > 1 && label == LegendLabels.covCount ) xtitle += "s";
    var msg = label+" at "+xtitle+" "+depth+br;

    // get counts and percentage values from the y value actually plotted.
    var y = item.datapoint[1];
    var ypc = y;
    if( label == LegendLabels.covCount ) {
      if( plotParams.barAxis == 1 ) { y /= plotStats.barScale; }
      else { ypc *= plotStats.barScale };
    } else {
      if( plotParams.plotAxis == 1 ) { y /= plotStats.barScale; }
      else { ypc *= plotStats.barScale };
    }
    msg += "Number of target bases read: "+commify(y.toFixed(0))+br;
    msg += "Fraction of all target bases read: "+sigfig(ypc)+"%"+br;
    $('#DOC-tooltip-close').show();
    return msg;
  }

  function commify(val) {
    // expects positive integers
    var jrs = val.toString();
    return jrs.replace(/(\d)(?=(\d\d\d)+(?!\d))/g, "$1,");
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

  function showCursorCoords(x,y,y2) {
    if( cursorOverPlot(x,y) ) {
      y2 = y2.toFixed(2)+'%';
      var xFix = plotParams.readAxis < 2 ? 0 : 3;
      cursorCoords.html(x.toFixed(xFix)+", "+y2);
      var offset = placeholder.offset();
      cursorCoords.appendTo(placeholder);
      cursorCoords.offset( {top:offset.top+placeholder.height()-18, left:offset.left} );
      cursorCoords.show();
      return true;
    }
    cursorCoords.hide();
    return false;
  }

  // autoload - after everything is defined
  unzoomToFile(coverageFile);
  cursorCoords.appendTo(placeholder);

  // collapse view after EVRYTHING has been drawn in open chart (to avoid flot issues)
  if( startCollapsed ) {
    $("#DOC-collapsePlot").attr("class","ui-icon ui-icon-triangle-1-s");
    $("#DOC-collapsePlot").attr("title","Expand view");
    $('#DOC-controlpanel').hide();
    $('.DOC-shy').hide();
    $('#DOC-chart').resizable('destroy');
    $('#DOC-noncanvas').hide();
  }

});
