// html for table container - note these are invisible and moved into position later
document.write('\
<div id="FL-tablecontent" class="unselectable" style="display:none">\
  <div id="FL-titlebar" class="grid-header">\
    <span id="FL-collapseGrid" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">File Links</span>\
    <span id="FL-message" class="message"></span>\
  </div>\
  <div id="FL-grid" class="linkstable"></div>\
</div>\
');

$(function () {

var disableTitleBar = false;

  // check placer element exists
  if( !$('#FileLinksTable').length ) return;

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

  var dataTable = [];
  var fieldsIds;

  function loadTSV(tsvFile) {
    dataTable = [];
    $('#FL-message').text('Loading...');
    $.ajaxSetup( {dataType:"text",async:false} );
    $.get(tsvFile, function(mem) {
      var lines = mem.split("\n");
      $.each(lines, function(n,row) {
        var fields = $.trim(row).split('\t');
        if( n == 0 ) {
          fieldIds = fields;
        } else if( fields[0] != "" ) {
          dataTable.push( fields );
        }
      });
    }).error(function(){
      alert("An error occurred while loading from "+tsvFile);
      $('#FL-message').text('');
    }).success(function(){
      $('#FL-message').text('');
    });
  }

  function formatData(text,link,help,fmat,row) {
    if( text == null || text == '' ) return '';
    var fmatTitle = text.replace( /\.$/, '' );
    if( text.substring(0,4).toLowerCase() != "link" ) {
      text = "Download the " + text.charAt(0).toLowerCase() + text.substring(1);
    }
    if( help == undefined || help == '' ) help = text;
    help = help.replace( /#/g, '\n' );
    if( link == 'IGV' ) {
      link = window.location.protocol + "//" + window.location.host + ":8080/IgvServlet/igv";
    }
    var cnt = "<div class='linkstable-row' row='"+(row&1 ? 'odd' : 'even')+"'><a class='flyhelp' href='"+link+"' title='"+help+"'>"+text+"</a>";
    if( fmat != undefined && fmat != "" ) {
      fmat = '<span style="text-decoration:underline">' + fmatTitle + '</span>#' + fmat;
      fmat = fmat.replace( /#/g, '<br/>' );
      var helpElem = "FL-help-"+row;
      cnt += '<span id="'+helpElem+'" style="float:right;margin-top:4px;margin-right:-6px" class="FL-HELP ui-icon ui-icon-help" title="Click for file format description."></span>';
      $('<div id="'+helpElem+'-txt">'+fmat+'</div>').hide().addClass('helpblock').appendTo('body');
    }
    return cnt+'</div>';
  }

  $('.FL-HELP').live("click", function(e) {
    var span = '#'+e.target.id;
    var htxt = span + '-txt';
    var offset = $(span).offset();
    var xpos = offset.left - $(htxt).width() - 8;
    var ypos = offset.top - $(htxt).height()/2;
    $(span).removeAttr("title");
    $(htxt).css({ position: 'absolute', left: xpos, top: ypos }).appendTo("body").slideDown();
  });

  $('.FL-HELP').live("hover", null, function(e) {
    var span = '#'+e.target.id;
    var htxt = span + '-txt';
    $(span).attr( "title", "Click for file format description." );
    $(htxt).fadeOut(200);
  });

  function displayTable() {
    if( dataTable.length <= 0 ) return;
    var text = '';
    for( var i = 0; i < dataTable.length; ++i ) {
      text += formatData(dataTable[i][0],dataTable[i][1],dataTable[i][2],dataTable[i][3],i);
    }
    $('#FL-grid').html(text);
  }

  var dataFile = $("#FileLinksTable").attr("fileurl");
  if( dataFile == undefined || dataFile == "" ) {
    // in tradition of HTML, an empty div results in no complaints
    //alert("ERROR on page: FileLinksTable widget requires attribute 'fileurl' is set.");
    $('#FileLinksTable').hide();
    return;
  }
  $("#FL-tablecontent").appendTo('#FileLinksTable');
  $("#FL-tablecontent").show();
  loadTSV(dataFile);
  displayTable();
  
});
