// html for table container and filters bar - note these are invisible and moved into position later
document.write('\
<div id="VC-tablecontent" style="display:none">\
  <div id="VC-titlebar" class="grid-header">\
    <span id="VC-collapseGrid" style="float:left" class="ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Variant Calls</span>\
    <span id="VC-toggleFilter" style="float:right" class="ui-icon ui-icon-search" title="Toggle search/filter panel"></span>\
    <span id="VC-message" class="message"></span>\
  </div>\
  <div id="VC-grid" class="grid-body"></div>\
  <div id="VC-pager" class="grid-footer"></div>\
  <p class="grid-text"/>\
</div>\
<div id="VC-filterpanel" class="filter-panel" style="display:none">\
  <table style="width:100%"><tr><td>\
    <span class="nwrap">Chrom <select id="VC-selectChrom" class="txtSelect" style="width:80px"><option value=""/></select></span>\
    <span class="nwrap">Position <input type="text" id="VC-txtSearchPosStart" class="numSearch" size=9>\
      to <input type="text" id="VC-txtSearchPosEnd" class="numSearch" size=9"></span>\
    <span class="nwrap">Gene Sym <input type="text" class="txtSearch" id="VC-txtSearchGeneSym" size=10></span>\
    <span class="nwrap">Target ID <input type="text" class="txtSearch" id="VC-txtSearchTargetID" size=11></span>\
    <span id="VC-filterHotSpot"><span class="nwrap">HotSpot ID <input type="text" class="txtSearch" id="VC-txtSearchHotSpotID" size=11></span></span>\
    <span class="nwrap">Type <select id="VC-selectVarType" class="txtSelect" style="width:55px"><option value=""/></select></span>\
    <span class="nwrap">Zygosity <select id="VC-selectPloidy" class="txtSelect" style="width:55px"><option value=""/></select></span>\
    <span class="nwrap">Var Freq <input type="text" class="numSearch" id="VC-txtSearchFreqMin" size=4 value="0">\
      to <input type="text" id="VC-txtSearchFreqMax" class="numSearch" size=4 value="100"></span>\
    <span class="nwrap">Cov &ge; <input type="text" id="VC-txtSearchCovMin" class="numSearch" size=7 value=""></span>\
  </td><td style="float:right;padding-right:8px">\
    <input type="button" id="VC-clearSelected" value="Clear Filters" title="Clear all current filters."><br/>\
    <input type="button" id="VC-checkSelected" class="checkOff" value="Selected"\
      title="Display only selected rows. Other filters are ignored and disabled while this filter is checked.">\
  </td></tr></table>\
</div>\
<div id="VC-mask" class="grid-mask"></div>\
<div id="VC-dialog" class="tools-dialog" style="display:none">\
  <div id="VC-dialog-title" class="title">Export Selected</div>\
  <div id="VC-dialog-content" class="content">...</div>\
  <div id="VC-dialog-buttons" class="buttons">\
    <input type="button" value="OK" id="VC-exportOK">\
    <input type="button" value="Cancel" onclick="$(\'#VC-dialog\').hide();$(\'#VC-mask\').hide();">\
  </div>\
</div>\
');

$(function () {

var disableTitleBar = false;

$("#VC-collapseGrid").click(function(e) {
  if( disableTitleBar ) return;
  if( $('#VC-grid').is(":visible") ) {
    $(this).attr("class","ui-icon ui-icon-triangle-1-s");
    $(this).attr("title","Expand view");
    $('#VC-filterpanel').slideUp();
    $('#VC-pager').slideUp();
    $('#VC-grid').slideUp('slow',function(){
      $('#VC-titlebar').css("border","1px solid grey");
      $('#VC-toggleFilter').attr("class","");
    });
  } else {
    $(this).attr("class","ui-icon ui-icon-triangle-1-n");
    $(this).attr("title","Collapse view");
    $('#VC-titlebar').css("border-bottom","0");
    $('#VC-pager').slideDown();
    $('#VC-grid').slideDown('slow',function(){
      $('#VC-toggleFilter').attr("class","ui-icon ui-icon-search");
    });
  }
});

$("#VC-toggleFilter").click(function(e) {
  if( disableTitleBar ) return;
  if( $('#VC-filterpanel').is(":visible") ) {
    $('#VC-filterpanel').slideUp();
  } else if( $('#VC-grid').is(":visible") ){
    $('#VC-filterpanel').slideDown();
  }
});

var filterSettings = {};

function resetFilterSettings() {
  filterSettings = {
    searchSelected: false,
    searchStringChrom: "",
    searchStringPosStart: Number(0),
    searchStringPosEnd: Number(0),
    searchStringGeneSym: "",
    searchStringTargetID: "",
    searchStringHotSpotID: "",
    searchStringVarType: "",
    searchStringPloidy: "",
    searchStringFreqMin: Number(0),
    searchStringFreqMax: Number(100),
    searchStringCovMin: Number(0)
  }
}

function updateFilterSettings() {
  updateSelectedFilter(false);
  $("#VC-selectChrom").attr('value',filterSettings['searchStringChrom']);
  $("#VC-txtSearchPosStart").attr('value',filterSettings['searchStringPosStart'] ? "" : filterSettings['searchStringPosStart']);
  $("#VC-txtSearchPosEnd").attr('value',filterSettings['searchStringPosEnd'] ? "" : filterSettings['searchStringPosEnd']);
  $("#VC-txtSearchGeneSym").attr('value',filterSettings['searchStringGeneSym']);
  $("#VC-txtSearchTargetID").attr('value',filterSettings['searchStringTargetID']);
  $("#VC-txtSearchHotSpotID").attr('value',filterSettings['searchStringHotSpotID']);
  $("#VC-selectVarType").attr('value',filterSettings['searchStringVarType']);
  $("#VC-selectPloidy").attr('value',filterSettings['searchStringPloidy']);
  $("#VC-txtSearchFreqMin").attr('value',filterSettings['searchStringFreqMin']);
  $("#VC-txtSearchFreqMax").attr('value',filterSettings['searchStringFreqMax']);
  $("#VC-txtSearchCovMin").attr('value',filterSettings['searchStringCovMin'] ? "" : filterSettings['searchStringCovMin']);
}

function updateSelectedFilter(turnOn) {
  filterSettings['searchSelected'] = turnOn;
  $('#VC-checkSelected').attr('class', turnOn ? 'checkOn' : 'checkOff');
  $('.txtSearch').attr('disabled',turnOn);
  $('.numSearch').attr('disabled',turnOn);
  checkboxSelector.setFilterSelected(turnOn);
}

function myFilter(item,args) {
  // for selected only filtering ignore all other filters
  if( args.searchSelected ) return item["check"];
  if( args.searchStringChrom != "" && item["chrom"] != args.searchStringChrom ) return false;
  if( strNoMatch( item["genesym"].toUpperCase(), args.searchStringGeneSym ) ) return false;
  if( strNoMatch( item["targetid"].toUpperCase(), args.searchStringTargetID ) ) return false;
  if( rangeNoMatch( item["position"], args.searchStringPosStart, args.searchStringPosEnd ) ) return false;
  if( args.searchStringVarType != "" && item["vartype"] != args.searchStringVarType ) return false;
  if( args.searchStringPloidy != "" && item["ploidy"] != args.searchStringPloidy ) return false;
  if( rangeNoMatch( item["varfreq"], args.searchStringFreqMin, args.searchStringFreqMax ) ) return false;
  if( rangeLess( item["coverage"], args.searchStringCovMin ) ) return false;
  if( item["hotspotid"] != undefined && strNoMatch( item["hotspotid"].toUpperCase(), args.searchStringHotSpotID ) ) return false;
  return true;
}

function exportTools() {
  // could use data[] here directly
  var items = dataView.getItems();
  var numSelected = 0;
  for( var i = 0; i < items.length; ++i ) {
    if( items[i]['check'] ) ++numSelected;
  }
  var $content = $('#VC-dialog-content');
  $content.html('Rows selected: '+numSelected+'<br/>');
  if( numSelected == 0 ) {
    $content.append('<p>You must first select rows of the table data to export.</p>');
    $('#VC-exportOK').hide();
  } else {
    $content.append('<p>\
      <input type="radio" name="exportTool" id="VC-ext1" value="table" checked="checked"/>\
        <label for="VC-ext1">Download table file of selected rows.</label><br/>\
      <input type="radio" name="exportTool" id="VC-ext2" value="taqman"/>\
        <label for="VC-ext2">Submit variants for TaqMan assay design.</label></p>' );
    $('#VC-exportOK').show();
  }
  // open dialog over masked out table
  var pos = $('#VC-pager').offset();
  var x = pos.left+22;
  var y = pos.top-$('#VC-dialog').height()+3;
  $('#VC-dialog').css({ left:x, top:y });
  pos = $('#VC-tablecontent').offset();
  var hgt = $('#VC-tablecontent').height()+7; // extra for borders?
  var wid = $('#VC-tablecontent').width()+2;
  $('#VC-mask').css({ left:pos.left, top:pos.top, width:wid, height:hgt });
  $('#VC-mask').show();
  $('#VC-dialog').show();
}

var dataFile = $("#variantCallsTable").attr("fileurl");

$('#VC-exportOK').click(function(e) {
  $('#VC-dialog').hide();
  // use ID's and resort to original order for original input file order matching
  var items = dataView.getItems();
  var checkList = [];
  for( var i = 0; i < items.length; ++i ) {
    if( items[i]['check'] ) {
      checkList.push(items[i]['id']);
    }
  }
  var rows = checkList.sort(function(a,b){return a-b;})+",";
  $('#VC-mask').hide();
  var op = $("input[@name=exportTool]:checked").val();
  if( op == "table" ) {
    window.open("subtable.php3?dataFile="+dataFile+"&rows="+rows);
  } else if( op == "taqman" ) {
    window.open("taqman.php3?dataFile="+dataFile+"&rows="+rows);
  }
});

function ChromIGV(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "N/A" }
  var pos = grid.getData().getItem(row)['chrom'] + ":" + value;
  var locpath = window.location.pathname.substring(0,window.location.pathname.lastIndexOf('/'));
  var igvURL = window.location.protocol + "//" + window.location.host + locpath + "/igv.php3";
  // link to Broad IGV
  //var href = "http://www.broadinstitute.org/igv/projects/current/igv.php?locus="+pos+"&sessionURL="+igvURL;
  // link to internal IGV
  var launchURL = window.location.protocol + "//" + window.location.host + ":8080/IgvServlet/igv";
  var href = launchURL + "?locus="+pos+"&sessionURL="+igvURL;
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
  toolTip: "The one-based position in the reference genome. Click the link to open the variant in IGV and view all reads covering the variant." });
columns.push({
  id: "genesym", name: "Gene Sym", field: "genesym", width: 72, minWidth: 40, maxWidth: 200, sortable: true,
  toolTip: "Gene symbol for the gene where the variant is located. This value is not available (N/A) if no target regions were defined." });
columns.push({
  id: "targetid", name: "Target ID", field: "targetid", width: 72, minWidth: 40, maxWidth: 200, sortable: true,
  toolTip: "Name of the target region where the variant is located. This value is not available (N/A) if no target regions were defined." });
columns.push({
  id: "vartype", name: "Type", field: "vartype", width: 46, minWidth: 40, maxWidth: 46, sortable: true,
  toolTip: "Type of variantion detected (SNP/INDEL)." });
columns.push({
  id: "ploidy", name: "Zygosity", field: "ploidy", width: 54, minWidth: 40, maxWidth: 46, sortable: true,
  toolTip: "Assigned zygosity of the variation: Homozygous (Hom), Heterozygous (Het) or No Call (NC)." });
columns.push({
  id: "reference", name: "Ref", field: "reference", width: 36, minWidth: 28,
  toolTip: "The reference base(s)." });
columns.push({
  id: "variant", name: "Variant", field: "variant", width: 44, minWidth: 38,
  toolTip: "Variant allele base(s)." });
columns.push({
  id: "varfreq", name: "Var Freq", field: "varfreq", width: 68, minWidth: 50, maxWidth: 68, sortable: true, formatter: formatPercent,
  toolTip: "Frequency of the variant allele." });
columns.push({
  id: "p_value", name: "P-value", field: "p_value", width: 60, minWidth: 40, maxWidth: 64, sortable: true, formatter: formatScientific,
  toolTip: "Estimated probability that the variant could be produced by chance." });
columns.push({
  id: "coverage", name: "Cov", field: "coverage", width: 50, minWidth: 40, maxWidth: 64, sortable: true,
  toolTip: "The total number of reads covering the reference base position." });
columns.push({
  id: "refcoverage", name: "Ref Cov", field: "refcoverage", width: 60, minWidth: 50, maxWidth: 64, sortable: true,
  toolTip: "The number of reads covering the reference allele." });
columns.push({
  id: "varcoverage", name: "Var Cov", field: "varcoverage", width: 61, minWidth: 50, maxWidth: 64, sortable: true,
  toolTip: "The number of reads covering the variant allele." });

// set up assumind there is no hotspot field - defined when file is loaded
$("#VC-filterHotSpot").hide();
$("#variantCallsTable").css('width','787px');

// define the grid and attach head/foot of the table
var options = {
  editable: true,
  autoEdit: false,
  enableCellNavigation: true,
  multiColumnSort: true
};
var dataView = new Slick.Data.DataView({inlineFilters: true});
var grid = new Slick.Grid("#VC-grid", dataView, columns, options);
grid.setSelectionModel( new Slick.RowSelectionModel({selectActiveRow: false}) );
grid.registerPlugin(checkboxSelector);

var pager = new Slick.Controls.Pager(dataView, grid, exportTools, $("#VC-pager"));
var columnpicker = new Slick.Controls.ColumnPicker(columns, grid, options);

$("#VC-tablecontent").appendTo('#variantCallsTable');
$("#VC-tablecontent").show();
$("#VC-filterpanel").appendTo('#VC-titlebar');

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
$("#VC-checkSelected").click(function(e) {
  var turnOn = ($(this).attr('class') === 'checkOff');
  updateSelectedFilter(turnOn);
  updateFilter();
  dataView.setPagingOptions({pageNum: 0});
});

$("#VC-clearSelected").click(function(e) {
  resetFilterSettings();  
  updateFilterSettings();
  updateFilter();
});

$("#VC-selectChrom").change(function(e) {
  filterSettings['searchStringChrom'] = this.value;
  updateFilter();
});

$("#VC-txtSearchPosStart").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  this.value = this.value.replace(/\D/g,"");
  filterSettings['searchStringPosStart'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#VC-txtSearchPosEnd").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  this.value = this.value.replace(/\D/g,"");
  filterSettings['searchStringPosEnd'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#VC-txtSearchGeneSym").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  filterSettings['searchStringGeneSym'] = this.value.toUpperCase();
  updateFilter();
});

$("#VC-txtSearchTargetID").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  filterSettings['searchStringTargetID'] = this.value.toUpperCase();
  updateFilter();
});

$("#VC-selectVarType").change(function(e) {
  filterSettings['searchStringVarType'] = this.value;
  updateFilter();
});

$("#VC-selectPloidy").change(function(e) {
  filterSettings['searchStringPloidy'] = this.value;
  updateFilter();
});

$("#VC-txtSearchFreqMin").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = 0; }
  this.value = forceStringFloat( this.value );
  filterSettings['searchStringFreqMin'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#VC-txtSearchFreqMax").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = 100; }
  this.value = forceStringFloat( this.value );
  filterSettings['searchStringFreqMax'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#VC-txtSearchCovMin").keyup(function(e) {
  Slick.GlobalEditorLock.cancelCurrentEdit();
  if( e.which == 27 ) { this.value = ""; }
  this.value = this.value.replace(/\D/g,"");
  filterSettings['searchStringCovMin'] = Number( this.value == "" ? 0 : this.value );
  updateFilter();
});

$("#VC-txtSearchHotSpotID").keyup(function(e) {
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
$("#VC-grid").css('height','27px');
$("#VC-grid").resizable({
  alsoResize: "#variantCallsTable",
  minWidth:300,
  handles:"e,s,se",
  stop:function(e,u) {
    $("#variantCallsTable").css('height','auto');
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
      $("#VC-grid").css('height',(numDataRows*25+27)+'px');
      // add HotSpot ID column if data available show filter
      if( haveHotSpots ) {
        columns.push({
          id: "hotspotid", name: "HotSpot ID", field: "hotspotid", width: 74, minWidth: 40, maxWidth: 200, sortable: true,
          toolTip: "HotSpot ID for one or more starting locations matching the identified variant." });
        grid.setColumns(columns);
        $("#variantCallsTable").css('width','861');
        $("#VC-filterHotSpot").show();
      }
    }
    dataView.setItems(data);
    grid.resizeCanvas();
    grid.render();
  }

  function onLoadSuccess() {
    onLoadPartial();
    $('#VC-message').html('');
  }

  function onLoadError() {
    if( errorTrace <= 1 ) {
      disableTitleBar = true;
      $('#VC-pager').hide();
      $('#VC-grid').hide();
      $('#VC-titlebar').css("border","1px solid grey");
      $('#VC-collapseGrid').attr('class','ui-icon ui-icon-alert');
      $('#VC-collapseGrid').attr("title","Failed to load data.");
      $('#VC-toggleFilter').attr('class','ui-icon ui-icon-alert');
      $('#VC-toggleFilter').attr("title","Failed to load data.");
    }
    if( errorTrace < 0 ) {
      alert("Could open Variant Calls table data file\n'"+dataFile+"'.");
    } else {
      alert("An error occurred loading Variant Calls data from file\n'"+dataFile+"' at line "+errorTrace);
    }
    $('#VC-message').append('<span style="color:red;font-style:normal">ERROR</span>');
  }
  
  $('#VC-message').html('Loading...');
  if( dataFile == null || dataFile == undefined || dataFile == "" ) {
    return onLoadError();
  }

  $.get(dataFile, function(mem) {
    var lines = mem.split("\n");
    $.each(lines, function(n,row) {
      errorTrace = n;
      var fields = $.trim(row).split('\t');
      var chr = fields[0];
      if( n > 0 && chr != '' ) {
        data[numRecords] = {
          id : Number(numRecords),
          check : false,
          chrom : chr,
          position : Number(fields[1]),
          genesym : fields[2],
          targetid : fields[3],
          vartype : fields[4],
          ploidy : fields[5],
          reference : fields[6],
          variant : fields[7],
          varfreq : Number(fields[8]),
          p_value : Number(fields[9]),
          coverage : Number(fields[10]),
          refcoverage : Number(fields[11]),
          varcoverage : Number(fields[12])
        };
        if( fields[13] != null && fields[13] != undefined ) {
          data[numRecords]['hotspotid'] = fields[13];
          haveHotSpots = true;
        }
        // record unique identifies and order of chromosomes from source
        if( selectAppendUnique('#VC-selectChrom',chr,chr) ) { chrMap[chr] = chrNum++; }
        selectAppendUnique('#VC-selectVarType',fields[4],fields[4]);
        selectAppendUnique('#VC-selectPloidy',fields[5],fields[5]);
        ++numRecords;
        if( loadUpdate > 0 && numRecords % loadUpdate == 0 ) onLoadPartial();
      }
    });
  }).success(onLoadSuccess).error(onLoadError);
}
postPageLoadMethods.push({callback: loadtable, priority: 40});

});
