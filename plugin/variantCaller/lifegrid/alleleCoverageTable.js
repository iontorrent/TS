// html for table container and filters bar - note these are invisible and moved into position later
document.write('\
<div id="AC-tablecontent" style="display:none">\
  <div id="AC-titlebar" class="grid-header">\
    <span id="AC-collapseGrid" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Allele Coverage for bases in HotSpot Regions<sup>&dagger;</sup></span>\
    <span id="AC-toggleFilter" style="float:right" class="ui-icon ui-icon-search" title="Toggle search/filter panel"></span>\
    <span id="AC-message" class="message"></span>\
  </div>\
  <div id="AC-grid" class="grid-body"></div>\
  <div id="AC-pager" class="grid-footer"></div>\
  <p id="AC-footnote" class="grid-footnote"><sup><b>&dagger;</b></sup>Table does not include coverage for HotSpot insertion variants.</p>\
  <p id="AC-footnote-close" style="display:none" class="grid-text"><br/></p>\
</div>\
<div id="AC-filterpanel" class="filter-panel" style="display:none">\
  <table style="width:100%"><tr><td>\
    <span class="nwrap">Chrom <select id="AC-selectChrom" class="txtSelect" style="width:80px"><option value=""/></select></span>\
    <span class="nwrap">Position <input type="text" id="AC-txtSearchPosStart" class="numSearch" size=9>\
      to <input type="text" id="AC-txtSearchPosEnd" class="numSearch" size=9"></span>\
    <span class="nwrap">Target ID <input type="text" class="txtSearch" id="AC-txtSearchTargetID" size=11></span>\
    <span id="AC-filterHotSpot"><span class="nwrap">HotSpot ID <input type="text" class="txtSearch" id="AC-txtSearchHotSpotID" size=11></span></span>\
    <span class="nwrap">Cov &ge; <input type="text" id="AC-txtSearchCovMin" class="numSearch" size=7 value=""></span>\
    <span class="nwrap">% Cov (+) <input type="text" class="numSearch" id="AC-txtSearchFreqMin" size=4 value="0">\
      to <input type="text" id="AC-txtSearchFreqMax" class="numSearch" size=4 value="100"></span>\
  </td><td style="float:right;padding-right:8px">\
    <input type="button" id="AC-clearSelected" value="Clear Filters" title="Clear all current filters."><br/>\
    <input type="button" id="AC-checkSelected" class="checkOff" value="Selected"\
      title="Display only selected rows. Other filters are ignored and disabled while this filter is checked.">\
  </td></tr></table>\
</div>\
<div id="AC-mask" class="grid-mask"></div>\
<div id="AC-dialog" class="tools-dialog" style="display:none">\
  <div id="AC-dialog-title" class="title">Export Selected</div>\
  <div id="AC-dialog-content" class="content">...</div>\
  <div id="AC-dialog-buttons" class="buttons">\
    <input type="button" value="OK" id="AC-exportOK">\
    <input type="button" value="Cancel" onclick="$(\'#AC-dialog\').hide();$(\'#AC-mask\').hide();">\
  </div>\
</div>\
');

$(function () {

var disableTitleBar = false;

$("#AC-collapseGrid").click(function(e) {
  if( disableTitleBar ) return;
  if( $('#AC-grid').is(":visible") ) {
    $(this).attr("class","ui-icon ui-icon-triangle-1-s");
    $(this).attr("title","Expand view");
    $('#AC-grid').slideUp('slow',function(){
      $('#AC-titlebar').css("border","1px solid grey");
      $('#AC-toggleFilter').attr("class","");
    });
    $('#AC-filterpanel').slideUp();
    $('#AC-footnote-close').slideDown();
    $('#AC-pager').slideUp();
    $('#AC-footnote').slideUp();
  } else {
    $(this).attr("class","ui-icon ui-icon-triangle-1-n");
    $(this).attr("title","Collapse view");
    $('#AC-titlebar').css("border-bottom","0");
    $('#AC-footnote-close').slideUp();
    $('#AC-footnote').slideDown();
    $('#AC-pager').slideDown();
    $('#AC-grid').slideDown('slow',function(){
      $('#AC-toggleFilter').attr("class","ui-icon ui-icon-search");
    });
  }
});

$("#AC-toggleFilter").click(function(e) {
  if( disableTitleBar ) return;
  if( $('#AC-filterpanel').is(":visible") ) {
    $('#AC-filterpanel').slideUp();
  } else if( $('#AC-grid').is(":visible") ){
    $('#AC-filterpanel').slideDown();
  }
});

var filterSettings = {};

function resetFilterSettings() {
  filterSettings = {
    searchSelected: false,
    searchStringChrom: "",
    searchStringPosStart: Number(0),
    searchStringPosEnd: Number(0),
    searchStringTargetID: "",
    searchStringHotSpotID: "",
    searchStringFreqMin: Number(0),
    searchStringFreqMax: Number(100),
    searchStringCovMin: Number(0)
  }
}

function updateFilterSettings() {
  updateSelectedFilter(false);
  $("#AC-selectChrom").attr('value',filterSettings['searchStringChrom']);
  $("#AC-txtSearchPosStart").attr('value',filterSettings['searchStringPosStart'] ? "" : filterSettings['searchStringPosStart']);
  $("#AC-txtSearchPosEnd").attr('value',filterSettings['searchStringPosEnd'] ? "" : filterSettings['searchStringPosEnd']);
  $("#AC-txtSearchTargetID").attr('value',filterSettings['searchStringTargetID']);
  $("#AC-txtSearchHotSpotID").attr('value',filterSettings['searchStringHotSpotID']);
  $("#AC-txtSearchFreqMin").attr('value',filterSettings['searchStringFreqMin']);
  $("#AC-txtSearchFreqMax").attr('value',filterSettings['searchStringFreqMax']);
  $("#AC-txtSearchCovMin").attr('value',filterSettings['searchStringCovMin'] ? "" : filterSettings['searchStringCovMin']);
}

function updateSelectedFilter(turnOn) {
  filterSettings['searchSelected'] = turnOn;
  $('#AC-checkSelected').attr('class', turnOn ? 'checkOn' : 'checkOff');
  $('.txtSearch').attr('disabled',turnOn);
  $('.numSearch').attr('disabled',turnOn);
  checkboxSelector.setFilterSelected(turnOn);
}

function myFilter(item,args) {
  // for selected only filtering ignore all other filters
  if( args.searchSelected ) return item["check"];
  if( args.searchStringChrom != "" && item["chrom"] != args.searchStringChrom ) return false;
  if( strNoMatch( item["targetid"].toUpperCase(), args.searchStringTargetID ) ) return false;
  if( rangeNoMatch( item["position"], args.searchStringPosStart, args.searchStringPosEnd ) ) return false;
  if( rangeLess( item["coverage"], args.searchStringCovMin ) ) return false;
  if( rangeNoMatch( item["bias"], args.searchStringFreqMin, args.searchStringFreqMax ) ) return false;
  if( item["hotspotid"] != undefined && strNoMatch( item["hotspotid"].toUpperCase(), args.searchStringHotSpotID ) ) return false;
  return true;
}

function exportTools() {
  var items = dataView.getItems();
  var numSelected = 0;
  for( var i = 0; i < items.length; ++i ) {
    if( items[i]['check'] ) ++numSelected;
  }
  var $content = $('#AC-dialog-content');
  $content.html('Rows selected: '+numSelected+'<br/>');
  if( numSelected == 0 ) {
    $content.append('<p>You must first select rows of the table data to export.</p>');
    $('#AC-exportOK').hide();
  } else {
    // extra code here preempts additional export options
    $content.append('<p>\
      <input type="radio" name="exportTool" id="AC-ext1" value="table" checked="checked"/>\
        <label for="AC-ext1">Download table file of selected rows.</label><p/>');
    $('#AC-exportOK').show();
  }
  // open dialog over masked out table
  var pos = $('#AC-pager').offset();
  var x = pos.left+22;
  var y = pos.top-$('#AC-dialog').height()+3;
  $('#AC-dialog').css({ left:x, top:y });
  pos = $('#AC-tablecontent').offset();
  var hgt = $('#AC-tablecontent').height()-$('#AC-footnote').height()-9;
  var wid = $('#AC-tablecontent').width()+2;
  $('#AC-mask').css({ left:pos.left, top:pos.top, width:wid, height:hgt });
  $('#AC-mask').show();
  $('#AC-dialog').show();
}

var dataFile = $("#alleleCoverageTable").attr("fileurl");

$('#AC-exportOK').click(function(e) {
  $('#AC-dialog').hide();
  var items = dataView.getItems();
  var checkList = [];
  for( var i = 0; i < items.length; ++i ) {
    if( items[i]['check'] ) {
      checkList.push(items[i]['id']);
    }
  }
  var rows = checkList.sort(function(a,b){return a-b;})+",";
  $('#AC-mask').hide();
  var op = $("input[@name=exportTool]:checked").val();
  if( op == "table" ) {
    window.open("subtable.php3?dataFile="+dataFile+"&rows="+rows);
  }
});

function ChromIGV(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "N/A" }
  var pos = grid.getData().getItem(row)['chrom'] + ":" + value;
  var locpath = window.location.pathname.substring(0,window.location.pathname.lastIndexOf('/'));
  var igvURL = window.location.protocol + "//" + window.location.host + locpath + "/igv.php3";
  var href = "http://www.broadinstitute.org/igv/projects/current/igv.php?locus="+pos+"&sessionURL="+igvURL;
  return "<a href='"+href+"'>"+value+"</a>";
}

var columns = [];
var checkboxSelector = new Slick.CheckboxSelectColumn();
columns.push(checkboxSelector.getColumnDefinition());
columns.push({
  id: "chrom", name: "Chrom", field: "chrom", width: 54, minWidth: 40, maxWidth: 100, sortable: true,
  toolTip: "The chromosome (or contig) name in the reference genome." });
columns.push({
  id: "position", name: "Position", field: "position", width: 65, minWidth: 38, maxWidth: 80, sortable: true, formatter: ChromIGV,
  toolTip: "The one-based position in the reference genome. Click the link to open the position in IGV and view all reads covering the position." });
columns.push({
  id: "targetid", name: "Target ID", field: "targetid", width: 72, minWidth: 40, maxWidth: 200, sortable: true,
  toolTip: "Name of the target region containing the HotSpot variation site." });
columns.push({
  id: "reference", name: "Ref", field: "reference", width: 36, minWidth: 28, maxWidth: 200,
  toolTip: "The reference base." });
columns.push({
  id: "coverage", name: "Cov", field: "coverage", width: 55, minWidth: 40, maxWidth: 74, sortable: true,
  toolTip: "The total reads covering the position, including deletions." });
columns.push({
  id: "cov_a", name: "A Reads", field: "cov_a", width: 55, minWidth: 26, maxWidth: 74,
  toolTip: "Number of reads calling A." });
columns.push({
  id: "cov_c", name: "C Reads", field: "cov_c", width: 55, minWidth: 26, maxWidth: 74,
  toolTip: "Number of reads calling C." });
columns.push({
  id: "cov_g", name: "G Reads", field: "cov_g", width: 55, minWidth: 26, maxWidth: 74,
  toolTip: "Number of reads calling G." });
columns.push({
  id: "cov_t", name: "T Reads", field: "cov_t", width: 55, minWidth: 26, maxWidth: 74,
  toolTip: "Number of reads calling T." });
columns.push({
  id: "cov_d", name: "Deletions", field: "cov_d", width: 65, minWidth: 40, maxWidth: 74, sortable: true,
  toolTip: "Number of reads calling deletion at this base location." });
columns.push({
  id: "cov_f", name: "+Cov", field: "cov_f", width: 55, minWidth: 34, maxWidth: 74,
  toolTip: "Number of forward reads aligned over the reference base that did not produce a base deletion call." });
columns.push({
  id: "cov_r", name: "-Cov", field: "cov_r", width: 55, minWidth: 34, maxWidth: 74,
  toolTip: "Number of reverse reads aligned over the reference base that did not produce a base deletion call." });
columns.push({
  id: "bias", name: "% +Cov", field: "bias", width: 59, minWidth: 40, maxWidth: 74, sortable: true, formatter: formatPercent,
  toolTip: "The proportion of forward reads to all reads at this position that did not produce a base deletion call." });

// set up assumind there is no hotspot field - defined when file is loaded
$("#AC-filterHotSpot").hide();
$("#alleleCoverageTable").css('width','781');

// define the grid and attach head/foot of the table
var options = {
  editable: true,
  autoEdit: false,
  enableCellNavigation: true,
  multiColumnSort: true
};
var dataView = new Slick.Data.DataView({inlineFilters: true});
var grid = new Slick.Grid("#AC-grid", dataView, columns, options);
grid.setSelectionModel( new Slick.RowSelectionModel({selectActiveRow: false}) );
grid.registerPlugin(checkboxSelector);

var pager = new Slick.Controls.Pager(dataView, grid, exportTools, $("#AC-pager"));
var columnpicker = new Slick.Controls.ColumnPicker(columns, grid, options);

// move the hidden panels to their positioning targets and display contents
$("#AC-tablecontent").appendTo('#alleleCoverageTable');
$("#AC-tablecontent").show();
$("#AC-filterpanel").appendTo('#AC-titlebar');

// multi-column sort method: uses data type but with original mapping for chromosome
var chrMap = [];

grid.onSort.subscribe(function(e,args) {
  var cols = args.sortCols;
  dataView.sort(function (dataRow1, dataRow2) {
    for( var i = 0, l = cols.length; i < l; i++ ) {
      var field = cols[i].sortCol.field;
      var value1 = dataRow1[field];
      var value2 = dataRow2[field];
      if( value1 == value2 ) continue;
      if( field === 'chrom' ) {
        value1 = chrMap[value1];
        value2 = chrMap[value2];
      }
      var sign = cols[i].sortAsc ? 1 : -1;
      return (value1 > value2) ? sign : -sign;
    }
    return 0;
  });
});

// wire up model events to drive the grid
dataView.onRowCountChanged.subscribe(function (e, args) {
  grid.updateRowCount();
  grid.render();
});

dataView.onRowsChanged.subscribe(function (e, args) {
  grid.invalidateRows(args.rows);
  grid.render();
  checkboxSelector.checkAllSelected();
});

// --- filter panel methods
$("#AC-checkSelected").click(function(e) {
  var turnOn = ($(this).attr('class') === 'checkOff');
  updateSelectedFilter(turnOn);
  updateFilter();
  dataView.setPagingOptions({pageNum: 0});
});

$("#AC-clearSelected").click(function(e) {
  resetFilterSettings();  
  updateFilterSettings();
  updateFilter();
});

$("#AC-selectChrom").change(function(e) {
  filterSettings['searchStringChrom'] = this.value;
  updateFilter();
});

$("#AC-txtSearchPosStart").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  this.value = this.value.replace(/\D/g,"");
  filterSettings['searchStringPosStart'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#AC-txtSearchPosEnd").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  this.value = this.value.replace(/\D/g,"");
  filterSettings['searchStringPosEnd'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#AC-txtSearchTargetID").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  filterSettings['searchStringTargetID'] = this.value.toUpperCase();
  updateFilter();
});

$("#AC-txtSearchFreqMin").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = 0; }
  this.value = forceStringFloat( this.value );
  filterSettings['searchStringFreqMin'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#AC-txtSearchFreqMax").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = 100; }
  this.value = forceStringFloat( this.value );
  filterSettings['searchStringFreqMax'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#AC-txtSearchCovMin").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  this.value = this.value.replace(/\D/g,"");
  filterSettings['searchStringCovMin'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#AC-txtSearchHotSpotID").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  filterSettings['searchStringHotSpotID'] = this.value.toUpperCase();
  updateFilter();
});

function updateFilter() {
  dataView.setFilterArgs(filterSettings);
  dataView.refresh();
}
checkboxSelector.setUpdateFilter(updateFilter);
resetFilterSettings();  
updateFilterSettings();

// set to default to 0 rows, including header
$("#AC-grid").css('height','27px');
$("#AC-grid").resizable({
  alsoResize: "#alleleCoverageTable",
  minWidth:300,
  handles:"e,s,se",
  stop:function(e,u) {
    $("#alleleCoverageTable").css('height','auto');
  }
});
grid.resizeCanvas();

// initialize the model after all the events have been hooked up
var data = []; // defined by file load later
dataView.beginUpdate();
dataView.setItems(data);
dataView.setFilterArgs(filterSettings);
dataView.setFilter(myFilter);
dataView.endUpdate();
dataView.syncGridSelection(grid, true);

// define function to load the table data and add to onload call list
// - dataView, grid, columns, data and chrMap[] all defined above
function loadtable() {
  var errorTrace = -1;
  var loadUpdate = 10000;
  var firstPartialLoad = true;
  var haveHotSpots = false;
  var chrNum = 0;
  var numRecords = 0;
  var initialRowDisplay = 10;

  function onLoadPartial() {
    if( firstPartialLoad ) {
      firstPartialLoad = false;
      var numDataRows = (numRecords < initialRowDisplay) ? numRecords : initialRowDisplay;
      $("#AC-grid").css('height',(numDataRows*25+27)+'px');
      // add HotSpot ID column if data available show filter
      if( haveHotSpots ) {
        columns.splice(4,0,{
          id: "hotspotid", name: "HotSpot ID", field: "hotspotid", width: 74, minWidth: 40, maxWidth: 200, sortable: true,
          toolTip: "Name of the HotSpot variant (site)." });
        grid.setColumns(columns);
        $("#alleleCoverageTable").css('width','855');
        $("#AC-filterHotSpot").show();
      }
    }
    dataView.setItems(data);
    grid.resizeCanvas();
    grid.render();
  }

  function onLoadSuccess() {
    onLoadPartial();
    $('#AC-message').html('');
  }

  function onLoadError() {
    if( errorTrace <= 1 ) {
      disableTitleBar = true;
      $('#AC-pager').hide();
      $('#AC-grid').hide();
      $('#AC-titlebar').css("border","1px solid grey");
      $('#AC-collapseGrid').attr('class','ui-icon ui-icon-alert');
      $('#AC-collapseGrid').attr("title","Failed to load data.");
      $('#AC-toggleFilter').attr('class','ui-icon ui-icon-alert');
      $('#AC-toggleFilter').attr("title","Failed to load data.");
      $('.grid-footnote').html('');
    }
    if( errorTrace < 0 ) {
      alert("Could open Allele Coverage table data file\n'"+dataFile+"'.");
    } else {
      alert("An error occurred loading Allele Coverage data from file\n'"+dataFile+"' at line "+errorTrace);
    }
    $('#AC-message').append('<span style="color:red;font-style:normal">ERROR</span>');
  }

  $('#AC-message').html('Loading...');
  if( dataFile == null || dataFile == undefined || dataFile == "" ) {
    return onLoadError();
  }

  $.get(dataFile, function(mem) {
    var lines = mem.split("\n");
    $.each(lines, function(n,row) {
      errorTrace = n;
      var fields = $.trim(row).split('\t');
      var chr = fields[0];
      if( chr == '' ) return true; // continue
      if( n == 0 ) {
        if( fields[3] == 'HotSpot ID' ) {
          haveHotSpots = true;
        }
      } else { 
        data[numRecords] = haveHotSpots ? {
          id : Number(numRecords),
          check : false,
          chrom : chr,
          position : Number(fields[1]),
          targetid : fields[2],
          hotspotid : fields[3],
          reference : fields[4],
          coverage : Number(fields[5]),
          cov_a : Number(fields[6]),
          cov_c : Number(fields[7]),
          cov_g : Number(fields[8]),
          cov_t : Number(fields[9]),
          cov_f : Number(fields[10]),
          cov_r : Number(fields[11]),
          cov_d : Number(fields[12]),
          bias : Number(100*fields[10])/(Number(fields[10])+Number(fields[11]))
        } : {
          id : Number(numRecords),
          check : false,
          chrom : chr,
          position : Number(fields[1]),
          targetid : fields[2],
          reference : fields[3],
          coverage : Number(fields[4]),
          cov_a : Number(fields[5]),
          cov_c : Number(fields[6]),
          cov_g : Number(fields[7]),
          cov_t : Number(fields[8]),
          cov_f : Number(fields[9]),
          cov_r : Number(fields[10]),
          cov_d : Number(fields[11]),
          bias : Number(100*fields[10])/(Number(fields[10])+Number(fields[11]))
        };
        // record unique identifies and order of chromosomes from source
        if( selectAppendUnique('#AC-selectChrom',chr,chr) ) { chrMap[chr] = chrNum++; }
        ++numRecords;
        if( loadUpdate > 0 && numRecords % loadUpdate == 0 ) onLoadPartial();
      }
    });
  }).success(onLoadSuccess).error(onLoadError);
}
postPageLoadMethods.push({callback: loadtable, priority: 30});
  
});
