// html for table container - note these are invisible and moved into position later
document.write('\
<div id="FL-tablecontent" style="display:none">\
  <div id="FL-titlebar" class="grid-header">\
    <span id="FL-collapseGrid" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">File Links</span>\
    <span id="FL-message" class="message"></span>\
  </div>\
  <div id="FL-grid" class="grid-body"></div>\
  <p class="grid-text"/>\
</div>\
');

$(function () {

var disableTitleBar = false;

$("#FL-collapseGrid").click(function(e) {
  if( disableTitleBar ) return;
  if( $('#FL-grid').is(":visible") ) {
    $(this).attr("class","ui-icon ui-icon-triangle-1-s");
    $(this).attr("title","Expand view");
    $('#FL-grid').slideUp('slow',function(){
      $('#FL-titlebar').css("border","1px solid grey");
    });
  } else {
    $(this).attr("class","ui-icon ui-icon-triangle-1-n");
    $(this).attr("title","Collapse view");
    $('#FL-titlebar').css("border-bottom","0");
    $('#FL-grid').slideDown();
  }
});

function filelink(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "" }
  var link = grid.getData().getItem(row)['link'];
  return "<span style='font-size:11pt;padding: 4px'><a href='"+link+"'>"+value+"</a></span>";
}

var columns = [{id: "text", name: "", field: "text", width: 480, resizable : false, formatter: filelink}];

// set up assumind there is no hotspot field - defined when file is loaded
$("#fileLinksTable").css('width','480');

// define the grid and attach head/foot of the table
var options = {
  autoEdit: false,
  autoHeight: true,
  enableColumnReorder: false
};
var dataView = new Slick.Data.DataView({inlineFilters: true});
var grid = new Slick.Grid("#FL-grid", dataView, columns, options);
grid.setSelectionModel( new Slick.RowSelectionModel({selectActiveRow: false}) );

$("#FL-tablecontent").appendTo('#fileLinksTable');
$("#FL-tablecontent").show();
$("#FL-filterpanel").appendTo('#FL-titlebar');

// wire up model events to drive the grid
dataView.onRowCountChanged.subscribe(function (e, args) {
  grid.updateRowCount();
  grid.render();
});

dataView.onRowsChanged.subscribe(function (e, args) {
  grid.invalidateRows(args.rows);
  grid.render();
});

$("#FL-grid .slick-header-columns").css("display","none");
$("#VC-grid").css('height','0px');
grid.resizeCanvas();

// initialize the model after all the events have been hooked up
var data = []; // defined by file load later
dataView.beginUpdate();
dataView.setItems(data);
dataView.endUpdate();
dataView.syncGridSelection(grid, true);

// define function to load the table data and add to onload call list
// - dataView, grid, columns, data and chrMap[] all defined above
var dataFile = $("#fileLinksTable").attr("fileurl");

function loadtable() {
  var errorTrace = -1;
  var numRecords = 0;
  var initialRowDisplay = 10;

  function onLoadSuccess() {
    $("#FL-grid").css('height',(numRecords*25+2)+'px');
    dataView.setItems(data);
    grid.resizeCanvas();
    grid.render();
    $('#FL-message').html('');
  }

  function onLoadError() {
    if( errorTrace <= 1 ) {
      disableTitleBar = true;
      $('#FL-grid').hide();
      $('#FL-titlebar').css("border","1px solid grey");
      $('#FL-collapseGrid').attr('class','ui-icon ui-icon-alert');
      $('#FL-collapseGrid').attr("title","Failed to load data.");
    }
    if( errorTrace < 0 ) {
      alert("Could open file links data file\n'"+dataFile+"'.");
    } else {
      alert("An error occurred loading file links data from file\n'"+dataFile+"' at line "+errorTrace);
    }
    $('#FL-message').append('<span style="color:red;font-style:normal">ERROR</span>');
  }
  
  $('#FL-message').html('Loading...');
  if( dataFile == null || dataFile == undefined || dataFile == "" ) {
    return onLoadError();
  }

  $.get(dataFile, function(mem) {
    var lines = mem.split("\n");
    $.each(lines, function(n,row) {
      errorTrace = n;
      var fields = $.trim(row).split('\t');
      var text = fields[0];
      if( n > 0 && text != '' ) {
        data[numRecords] = {
          id : Number(numRecords),
          text : text,
          link : fields[1]
        };
        ++numRecords;
      }
    });
  }).success(onLoadSuccess).error(onLoadError);
}
postPageLoadMethods.push({callback: loadtable, priority: 20});

});
