// html for table container - note these are invisible and moved into position later
document.write('\
<div id="VCS-tablecontent" style="display:none">\
  <div id="VCS-titlebar" class="grid-header">\
    <span id="VCS-collapseGrid" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Variant Calls Summary</span>\
    <span id="VCS-message" class="message"></span>\
  </div>\
  <div id="VCS-grid" class="grid-body"></div>\
  <div id="VCS-pager" class="grid-footer"></div>\
  <p class="grid-text"/>\
</div>\
');

$(function () {

var disableTitleBar = false;

$("#VCS-collapseGrid").click(function(e) {
  if( disableTitleBar ) return;
  if( $('#VCS-grid').is(":visible") ) {
    $(this).attr("class","ui-icon ui-icon-triangle-1-s");
    $(this).attr("title","Expand view");
    $('#VCS-pager').slideUp();
    $('#VCS-grid').slideUp('slow',function(){
      $('#VCS-titlebar').css("border","1px solid grey");
    });
  } else {
    $(this).attr("class","ui-icon ui-icon-triangle-1-n");
    $(this).attr("title","Collapse view");
    $('#VCS-titlebar').css("border-bottom","0");
    $('#VCS-pager').slideDown();
    $('#VCS-grid').slideDown();
  }
});

var columns = [];
columns.push({
  id: "chrom", name: "Chrom", field: "chrom", width: 72, minWidth: 40, maxWidth: 80,
  toolTip: "The chromosome (or contig) name in the reference genome." });
columns.push({
  id: "variants", name: "Variants", field: "variants", width: 72, minWidth: 38, maxWidth: 80,
  toolTip: "The total number of variants called in (the target regions of) the reference." });
columns.push({
  id: "hetsnps", name: "Het SNPs", field: "hetsnps", width: 72, minWidth: 54, maxWidth: 80,
  toolTip: "The total number of heterozygous SNPs called in (the target regions of) the reference." });
columns.push({
  id: "homsnps", name: "Hom SNPs", field: "homsnps", width: 72, minWidth: 54, maxWidth: 80,
  toolTip: "The total number of homozygous SNPs called in (the target regions of) the reference." });
columns.push({
  id: "hetindels", name: "Het INDELs", field: "hetindels", width: 72, minWidth: 54, maxWidth: 80,
  toolTip: "The total number of heterozygous INDELs called in (the target regions of) the reference." });
columns.push({
  id: "homindels", name: "Hom INDELs", field: "homindels", width: 72, minWidth: 54, maxWidth: 80,
  toolTip: "The total number of homozygous INDELs called in (the target regions of) the reference." });

// set up assumind there is no hotspot field - defined when file is loaded
$("#variantCallsSummaryTable").css('width','449px');

// define the grid and attach head/foot of the table
var options = {
  editable: false,
  autoEdit: false,
  enableCellNavigation: true,
  multiColumnSort: false
};
var dataView = new Slick.Data.DataView({inlineFilters: true});
var grid = new Slick.Grid("#VCS-grid", dataView, columns, options);
grid.setSelectionModel( new Slick.RowSelectionModel({selectActiveRow: false}) );

var pager = new Slick.Controls.Pager(dataView, grid, null, $("#VCS-pager"));

$("#VCS-tablecontent").appendTo('#variantCallsSummaryTable');
$("#VCS-tablecontent").show();
$("#VCS-filterpanel").appendTo('#VCS-titlebar');

// wire up model events to drive the grid
dataView.onRowCountChanged.subscribe(function (e, args) {
  grid.updateRowCount();
  grid.render();
});

dataView.onRowsChanged.subscribe(function (e, args) {
  grid.invalidateRows(args.rows);
  grid.render();
});

// set to default to 0 rows, including header
$("#VCS-grid").css('height','27px');
$("#VCS-grid").resizable({
  alsoResize: "#variantCallsSummaryTable",
  minWidth:300,
  handles:"e,s,se",
  stop:function(e,u) {
    $("#variantCallsSummaryTable").css('height','auto');
  }
});
grid.resizeCanvas();

// initialize the model after all the events have been hooked up
var data = []; // defined by file load later
dataView.beginUpdate();
dataView.setItems(data);
dataView.endUpdate();
dataView.syncGridSelection(grid, true);

// define function to load the table data and add to onload call list
// - dataView, grid, columns, data and chrMap[] all defined above
var dataFile = $("#variantCallsSummaryTable").attr("fileurl");

function loadtable() {
  var errorTrace = -1;
  var loadUpdate = 10000;
  var firstPartialLoad = true;
  var haveHotSpots = false;
  var numRecords = 0;
  var initialRowDisplay = 10;

  function onLoadPartial() {
    if( firstPartialLoad ) {
      firstPartialLoad = false;
      var numDataRows = (numRecords < initialRowDisplay) ? numRecords : initialRowDisplay;
      $("#VCS-grid").css('height',(numDataRows*25+27)+'px');
      // add HotSpot ID column if data available show filter
      if( haveHotSpots ) {
        columns.push({
          id: "hotspots", name: "HotSpots", field: "hotspots", width: 72, minWidth: 54, maxWidth: 80,
          tooolTip: "The total number of variants identified with one or more HotSpots."});
        grid.setColumns(columns);
        $("#variantCallsSummaryTable").css('width','521');
      }
    }
    dataView.setItems(data);
    grid.resizeCanvas();
    grid.render();
  }

  function onLoadSuccess() {
    onLoadPartial();
    $('#VCS-message').html('');
  }

  function onLoadError() {
    if( errorTrace <= 1 ) {
      disableTitleBar = true;
      $('#VCS-pager').hide();
      $('#VCS-grid').hide();
      $('#VCS-titlebar').css("border","1px solid grey");
      $('#VCS-collapseGrid').attr('class','ui-icon ui-icon-alert');
      $('#VCS-collapseGrid').attr("title","Failed to load data.");
    }
    if( errorTrace < 0 ) {
      alert("Could open Variant Calls Summary table data file\n'"+dataFile+"'.");
    } else {
      alert("An error occurred loading Variant Calls Summary data from file\n'"+dataFile+"' at line "+errorTrace);
    }
    $('#VCS-message').append('<span style="color:red;font-style:normal">ERROR</span>');
  }
  
  $('#VCS-message').html('Loading...');
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
          chrom : chr,
          variants : Number(fields[1]),
          hetsnps : Number(fields[2]),
          homsnps : Number(fields[3]),
          hetindels : Number(fields[4]),
          homindels : Number(fields[5])
        };
        if( fields[6] != null && fields[6] != undefined ) {
          data[numRecords]['hotspots'] = Number(fields[6]);
          haveHotSpots = true;
        }
        ++numRecords;
        if( loadUpdate > 0 && numRecords % loadUpdate == 0 ) onLoadPartial();
      }
    });
  }).success(onLoadSuccess).error(onLoadError);
}
postPageLoadMethods.push({callback: loadtable, priority: 10});

});
