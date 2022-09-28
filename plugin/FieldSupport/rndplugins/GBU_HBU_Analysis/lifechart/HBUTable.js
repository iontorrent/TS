// html for table container and filters bar - note these are invisible and moved into position later

document.write('\
<div id="HBU-tablecontent" class="unselectable" style="display:none">\
  <div id="HBU-titlebar" class="grid-header">\
    <span id="HBU-collapseGrid" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Hotspot Base Uniformity (HBU)</span>\
    <span id="HBU-message" class="message"></span>\
  </div>\
  <div id="HBU-Grid" class="grid-body selectable" style="height:200px"></div>\
</div>\
');

$(function () {

var disableTitleBar = false;

  // check placer element exists
  if( !$('#HBUTable').length ) return;

  $("#HBU-collapseGrid").click(function(e) {
    if( disableTitleBar ) return;
    if( $('#HBU-Grid').is(":visible") ) {
      $(this).attr("class","ui-icon ui-icon-triangle-1-s");
      $(this).attr("title","Expand view");
      $('#HBU-Grid').slideUp('slow',function(){
        $('#HBU-titlebar').css("border","1px solid grey");
      });
    } else {
      $(this).attr("class","ui-icon ui-icon-triangle-1-n");
      $(this).attr("title","Collapse view");
      $('#HBU-titlebar').css("border-bottom","0");
      $('#HBU-Grid').slideDown();
    }
  });
	
  var grid;
  
  var columns = [];
  columns.push({
  id: "gene", name: "Genes", field: "gene", width: 80, minWidth: 70, maxWidth: 90, sortable: true,
  toolTip: "Gene Name" });
  
  columns.push({
  id: "mincov", name: "MinCov", field: "mincov", width: 80, minWidth: 70, maxWidth: 90, sortable: true,
  toolTip: "Minimum Coverage" });
  
  columns.push({
  id: "maxcov", name: "MaxCov", field: "maxcov", width: 80, minWidth: 70, maxWidth: 90, sortable: true,
  toolTip: "Maximum Coverage" });
  
  columns.push({
  id: "hbu", name: "HBU", field: "hbu", width: 80, minWidth: 70, maxWidth: 90, sortable: true,
  toolTip: "Hotspot Base Uniformity (HBU)" });

  columns.push({
  id: "cov1x", name: "Cov1X", field: "cov1x", width: 80, minWidth: 70, maxWidth: 90, sortable: true,
  toolTip: "The number of bases coveraged by at least one read." });
  
  columns.push({
  id: "hlen", name: "HLen", field: "hlen", width: 80, minWidth: 70, maxWidth: 90, sortable: true,
  toolTip: "The number of bases of the hotspot" });

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
        	hbu:fields[3],
        	cov1x:Math.round(fields[4]),
        	hlen:Math.round(fields[8])
         }
       }
     });
    }).error(function(){
      alert("An error occurred while loading from "+dataFile);
      $('#HBU-message').text('');
    }).success(function(){
      $('#HBU-message').text('');
    });
  }

  var dataFile = $("#HBUTable").attr("hbuurl");
  loadCSV(dataFile);
  
  $("#HBU-tablecontent").appendTo('#HBUTable');
  $("#HBU-tablecontent").show();
  grid = new Slick.Grid("#HBU-Grid", data, columns, options);

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
