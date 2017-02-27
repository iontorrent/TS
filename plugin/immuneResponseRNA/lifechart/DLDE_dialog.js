// html for Diffential Expression file creation dialog
document.write('\
<div id="ASRNA-mask" class="grid-mask"></div>\
<div id="ASRNA-dialog" class="tools-dialog" style="width:auto;display:none">\
  <div id="ASRNA-dialog-title" class="title">Export Differential Expression Table</div>\
  <div id="ASRNA-dialog-content" class="content">\
    <h3>Specify two barcodes to compare:</h3>\
    <table>\
      <td><table>\
        <tr><th>Control</th></tr>\
        <tr><td><select style="height:22px;border: 1px solid lightblue;" id="ASRNA-barcode1"></select></td></tr>\
      </table></td>\
      <td><table>\
        <tr><th>Experiment</th></tr>\
        <tr><td><select style="height:22px;border: 1px solid lightblue;" id="ASRNA-barcode2"></select></td></tr>\
      </table></td>\
    </table>\
    <table>\
      <td><table>\
        <tr><td><span class="help"\
          title="The read detection threshold is used to determine how differential expression is handled for amplicons with very low read counts; implementing the strategy specified below. A low value, e.g. 1, will give the maximum sensitivity but may facilitate large inaccuracy in differential expression ratios. A value of 0 is not allowed as this leads to infinite log2 values (or errors) in the results. Amplicon reads are summed for multiple barcodes in the control or experiment group before this threshold is applied. The total barcode reads employed for the Reads Per Million (RPM) reads normalization does not consider the detection threshold.">\
          For targets with read counts below detection threshold:</span>&nbsp;&nbsp;\
          <input type="text" style="width:40px;height:12px;margin-top:8px" id="ASRNA-threshold" value=10>\
        </td></tr>\
        <tr><td style="padding-left:20px">\
          <input type="radio" name="ASRNA-thresAction" value="threshold" checked style="padding-right:50px">&nbsp; <span class="help"\
            title="If the total read counts for a given amplicon is below the read detection threshold then the read count is set to this threshold. If both control and experiment read counts are below the threshold then the RPM reads ratio is set to 1.">Apply the threshold as the minimum read count</span>\
          <td/></tr>\
        <tr><td style="padding-left:20px">\
          <input type="radio" name="ASRNA-thresAction" value="exclude">&nbsp; <span class="help"\
            title="If the total read counts for a given amplicon is below the read detection threshold for either the control or experiment then data for this amplicon is excluded from the differential expression table.">Exclude targets from the diffential expression table</span>\
        </td></tr>\
      </table></td>\
    </table>\
    <h6 style="max-width:490px">\
    Differential expression for each target will represented as the log2 of the ratio\
    of RPM reads of the experiment barcode to the control barcode.\
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
    var x = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
    var y = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
    var pos = $(this).offset();
    var a = $('#ASRNA-dialog').width();
    var b = $('#ASRNA-dialog').height();
    $('#ASRNA-dialog').css({ left:(x-a)/2, top:(y-b)/2 });
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
    var strVal = $('#ASRNA-threshold').val();
    var val = parseInt(strVal);
    if( val <= 0 || val != parseFloat(strVal) ) {
      alert("Read counts detecton threshold\nmust be a positive integer value.");
      return;
    }
    $('#ASRNA-dialog').hide();
    $('#ASRNA-mask').hide();
    bc1 = $("#ASRNA-barcode1").val();
    bc2 = $("#ASRNA-barcode2").val();
    bcc = $("#ASRNA-barcode1 option").length;
    threshold = $("#ASRNA-threshold").val();
    thresAction = $('input[name=ASRNA-thresAction]:checked').val();
    window.location = 'lifechart/pair_de_table.php3?data='+inputData+'&threshold='+threshold+'&thresAction='+thresAction+'&bc1='+bc1+'&bc2='+bc2+'&bccount='+bcc;
  });

});

