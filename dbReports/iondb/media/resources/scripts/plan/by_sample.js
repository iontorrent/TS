/**
    This file contains JQuery event attachments
    for the Plan By Sample's Barcoding Chevron
*/

var SAMPLESETITEMS = SAMPLESETITEMS || {};

$(document).ready(function () {
    
    // on sample change: update table fields that come from SampleSet
    //TODO: may need to also load Gender, RelationRole, SetID
    $(".irSampleName").change(function(){
    	
        var $row = $(this).parent().parent();
        var ITEM = SAMPLESETITEMS[$(this).val()];
    	
    	/*
        var preSelectedBarcode = ITEM['barcodeId'];
        if (preSelectedBarcode) {
        	console.log("by_sample.js preSelectedBarcode=", preSelectedBarcode);
        }
		*/
		
        var userSelectedBarcode = $row.find('select[name=barcode]').find(":selected").data("id_str");
        //console.log("by_samaple userSelectedBarcode=", userSelectedBarcode);
        
        $row.find('input[name=sampleExternalId]').val(ITEM['externalId']);
        $row.find('input[name=sampleDescription]').val(ITEM['description']);

        $row.find('select[name=ircancerType]').val(ITEM['ircancerType']);
        $row.find('input[name=ircellularityPct]').val(ITEM['ircellularityPct']);
        $row.find('input[name=irbiopsyDays]').val(ITEM['irbiopsyDays']);
        $row.find('input[name=ircoupleID]').val(ITEM['ircoupleId']);
        $row.find('input[name=irembryoID]').val(ITEM['irembryoId']);
        
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();

        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
    	//console.log("at by_sample - irSampleName.change() isSameSampleForDual=", isSameSampleForDual, "; ITEM=", ITEM);
    	
        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True")) {
        	var currentRow = $(this).closest('tr');

        	if (currentRow.index() == 0 && ($('#numRows').val() > 1) && (isSameSampleForDual)) {
        		var row1 = $('#row1');
        	
        		var value = $row.find('input[name=sampleExternalId]').val();
        		row1.children().find('input[name=sampleExternalId]').val(value);   
            	
        		value = $row.find('input[name=sampleDescription]').val();
        		row1.children().find('input[name=sampleDescription]').val(value);    
        		
                value = $row.find('select[name=ircancerType]').val();
                row1.children().find('select[name=ircancerType]').val(value);
                
                value = $row.find('input[name=ircellularityPct]').val();                
                row1.children().find('input[name=ircellularityPct]').val(value);

                value = $row.find('input[name=irbiopsyDays]').val();
                row1.children().find('input[name=irbiopsyDays]').val(value);
                value = $row.find('input[name=ircoupleID]').val();
                row1.children().find('input[name=ircoupleID]').val(value);
                value = $row.find('input[name=irembryoID]').val();  
                row1.children().find('input[name=irembryoID]').val(value);
                
                //if user changes sample selection, duplicate the barcode selection as well to avoid incorrect carryover of barcode selection
                // for the previous sample
                if (userSelectedBarcode) {
                	value = $row.find('select[name=barcode]').val();
                	console.log("by_sample going to set row1 barcode selection. value=", value);
                	
           	 		row1.find('select[name=barcode]').val(value);
                }
                
        	}

        }
    });

});
