// html for Diffential Expression file creation dialog
document.write('\
<div id="ASRNA-mask" class="grid-mask"></div>\
<div id="ASRNA-dialog" class="tools-dialog" style="width:auto;display:none">\
  <div id="ASRNA-dialog-title" class="title">Export Differential Expression Table</div>\
  <div id="ASRNA-dialog-content" class="content">\
    <h3>Specify two barcodes to compare:</h3>\
    <table><th>\
      <td><table>\
        <tr><td>Control</td></tr>\
        <tr><td><select class="txtSelect" id="ASRNA-barcode1"></select></td></tr>\
      </table></td>\
      <td><table>\
        <tr><td>Experiment</td></tr>\
        <tr><td><select class="txtSelect" id="ASRNA-barcode2"></select></td></tr>\
      </table></td>\
    </th></table>\
    <h6 style="max-width:490px">\
    Differential expression for each target will represented as the log2 of the ratio\
    of RPM reads of the experiment barcode to the control barcode. RPM reads ratios are\
    calculated assuming 10 reads for targets with less than 10 reads.\
    </h6>\
  </div>\
  <div id="ASRNA-dialog-buttons" class="buttons">\
    <input type="button" value="Download" style="width:auto" id="ASRNA-dialogOK">\
    <input type="button" value="Cancel" style="width:auto" id="ASRNA-dialogCancel" onclick="$(\'#ASRNA-dialog\').hide();$(\'#ASRNA-mask\').hide();">\
  </div>\
</div>\
');

$(function() {
  var bclist = $("#ASRNA-DLDE-dialog").attr("bclist");
  var barcodes = [];
  var bc2hide = "";
  if( bclist !== undefined ) barcodes = bclist.split(',');
  if( barcodes.length > 1 ) {
    var ddl1 = $("#ASRNA-barcode1");
    var ddl2 = $("#ASRNA-barcode2");
    for( i = 0; i < barcodes.length; ++i ) {
      ddl1.append("<option value='"+barcodes[i]+"'>"+barcodes[i]+"</option>");
      ddl2.append("<option value='"+barcodes[i]+"'>"+barcodes[i]+"</option>");
    }
    // first item selected should not be also in second list
    ddl2.val(barcodes[1]);
    bc2hide = barcodes[0];
    $("#ASRNA-barcode2 option[value='"+bc2hide+"']").hide();
  } else {
    $('#ASRNA-dialog-content').html("You must have at least 2 barcodes to generate<br/>a diferential expression matrix.");
    $('#ASRNA-dialogOK').hide();
    $('#ASRNA-dialogCancel').val("Close");
  }
  var inputData = $("#ASRNA-DLDE-dialog").attr("data");
  if( inputData === undefined || inputData === '' ) {
    $('#ASRNA-dialog-content').html("Error: Missing 'data' attribute from div tag!");
    $('#ASRNA-dialogOK').hide();
    $('#ASRNA-dialogCancel').val("Close");
  }

  $('#ASRNA-DLDE-dialog').click(function() {
    $('#ASRNA-mask').css({ position:'fixed' });
    $('#ASRNA-mask').show();
    var pos = $(this).offset();
    $('#ASRNA-dialog').css({ left:pos.left-5, top:pos.top });
    $('#ASRNA-dialog').show();
  });

  $('#ASRNA-barcode1').change(function() {
    var b1sel = $(this).val();
    var b2sel = $("#ASRNA-barcode2").val();
    $("#ASRNA-barcode2 option[value='"+bc2hide+"']").show();
    if( b1sel == b2sel ) $("#ASRNA-barcode2").val(bc2hide);
    bc2hide = b1sel;
    $("#ASRNA-barcode2 option[value='"+bc2hide+"']").hide();
  });

  $('#ASRNA-dialogOK').click(function(e) {
    $('#ASRNA-dialog').hide();
    $('#ASRNA-mask').hide();
    bc1 = $("#ASRNA-barcode1").val();
    bc2 = $("#ASRNA-barcode2").val();
    bcc = $("#ASRNA-barcode1 option").length;
    window.location = 'lifechart/pair_de_table.php3?data='+inputData+'&bc1='+bc1+'&bc2='+bc2+'&bccount='+bcc;
  });

});

