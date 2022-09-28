// html for table container and filters bar - note these are invisible and moved into position later

document.write('\
<div id="GBU-tablecontent" class="unselectable" style="display:none">\
  <div id="GBU-titlebar" class="grid-header">\
    <span id="GBU-collapseGrid" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Gene Base Uniformity (GBU)</span>\
    <span id="GBU-message" class="message"></span>\
  </div>\
  <div id="GBU-Grid" class="grid-body selectable" style="height:200px"></div>\
</div>\
');

$(function () {

var disableTitleBar = false;
	
  // check placer element exists
  if( !$('#GBUTable').length ) return;

  $("#GBU-collapseGrid").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#GBU-Grid').is(":visible") ) {
      $(this).attr("class","ui-icon ui-icon-triangle-1-s");
      $(this).attr("title","Expand view");
      $('#GBU-Grid').slideUp('slow',function(){
        $('#GBU-titlebar').css("border","1px solid grey");
      });
    } else {
      $(this).attr("class","ui-icon ui-icon-triangle-1-n");
      $(this).attr("title","Collapse view");
      $('#GBU-titlebar').css("border-bottom","0");
      $('#GBU-Grid').slideDown();
    }
  });
	
  var grid;
  
  var columns = [];
  columns.push({
  id: "gene", name: "Genes", field: "gene", width: 120, minWidth: 100, maxWidth: 130, sortable: true,
  toolTip: "Gene Name" });
  
  columns.push({
  id: "mincov", name: "MinCov", field: "mincov", width: 120, minWidth: 100, maxWidth: 130, sortable: true,
  toolTip: "Minimum Coverage" });
  
  columns.push({
  id: "maxcov", name: "MaxCov", field: "maxcov", width: 120, minWidth: 100, maxWidth: 130, sortable: true,
  toolTip: "Maximum Coverage" });
  
  columns.push({
  id: "gbu", name: "GBU", field: "gbu", width: 120, minWidth: 100, maxWidth: 130, sortable: true,
  toolTip: "Gene Base Uniformity (GBU)" });

  var options = {
    enableCellNavigation: true,
    enableColumnReorder: false,
    multiColumnSort: true
  };

  var data = [];
  
  function loadCSV(dataFile) {
    $.ajaxSetup( {dataType:"text",async:false} );
    $.get(dataFile, function(mem) {
      var lines = mem.replace(/^(?=\n)$|^\s*|\s*$|\n\n+/gm,"").split("\n");
      $.each(lines, function(n,row) {
        if (n > 0) {
          var fields = $.trim(row).split(',');
          data[n-1] = {
         	gene:fields[0],
               	mincov:Math.round(fields[1]),
        	maxcov:Math.round(fields[2]),
        	gbu:fields[3]
         }
       }
     });
    }).error(function(){
      alert("An error occurred while loading from "+dataFile);
      $('#GBU-message').text('');
    }).success(function(){
      $('#GBU-message').text('');
    });
  }

  var dataFile = $("#GBUTable").attr("gbuurl");
  loadCSV(dataFile);
  
  $("#GBU-tablecontent").appendTo('#GBUTable');
  $("#GBU-tablecontent").show();
  grid = new Slick.Grid("#GBU-Grid", data, columns, options);

  grid.onSort.subscribe(function (e, args) {
      var cols = args.sortCols;

      data.sort(function (dataRow1, dataRow2) {
        for (var i = 0, l = cols.length; i < l; i++) {
          var field = cols[i].sortCol.field;
          var sign = cols[i].sortAsc ? 1 : -1;
          var value1 = dataRow1[field], value2 = dataRow2[field];
          var result = (value1 == value2 ? 0 : (value1 > value2 ? 1 : -1)) * sign;
          if (result != 0) {
            return result;
          }
        }
        return 0;
      });
      grid.invalidate();
      grid.render();
    });
})
