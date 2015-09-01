/**
 Update hidden samplesTable input whenever any samples/barcodes/IR params change
 */
updateSamplesTable = function () {
    var barcoded = $('input[id=chk_barcoded]').is(':checked');
    var table = [];

    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

    $('#chipsets').find('tbody tr').each(function () {
        var row = {};

        $(this).find(':input').each(function () {
            if (this.name == 'barcode') {
                if (barcoded) row['barcodeId'] = $(this).find(':selected').data('id_str');
            } else {
                row[this.name] = this.value ? this.value : '';
                //console.log("page_plan_sample_table.updateSamplesTable name=", this.name, "; row[name]=", row[this.name]);
            }
        });

        // add IR pass-through fields, properties of IR workflow
        if (USERINPUT.is_ir_connected) {
            var irWorkflow = $(this).find('select[name=irWorkflow]').val();
            row['ApplicationType'] = USERINPUT.workflow_to_application_type_map[irWorkflow];
            row['irRelationshipType'] = USERINPUT.sample_group_to_workflow_map[irWorkflow]["relationshipType"];
        }
        
        table.push(row);

    });
    if (table.length > 0) {
        //console.log("page_plan_sample_table.updateSamplesTable table=", JSON.stringify(table));
        $('#samplesTable').val(JSON.stringify(table));
    }
}

var isEven = function(aNumber){
    return (aNumber % 2 == 0) ? true : false;
};
 
// create new table row by copying first row
createRow = function (i) {
    //console.log("createRow() i=", i);
    
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    
    var previousRowIndex = i - 1;
    if (isDualNucleotideType == "True" && isEven(i)) {
        previousRowIndex = i - 2;
    }
    if (previousRowIndex < 0) {
        previousRowIndex = 0;
    }

    var $row = $('#row' + previousRowIndex).clone(true, true).attr('id', 'row' + i);

    $row.children().find('select[name=barcode]').val(i);
    
    if (isDualNucleotideType != "True") {
        //empty values for the new rows
        $row.find("input").each(function () {
            $(this).val('');
        });

        //auto-assign row number value
        $row.children().eq(0).find('input').attr('value', i + 1);
        
        var $row0_reference = $('#row0').find('select[name=reference]').val();
        var $row0_target = $('#row0').find('select[name=targetRegionBedFile]').val();
        var $row0_hotSpot = $('#row0').find('select[name=hotSpotRegionBedFile]').val();
        //console.log("createRow() $row0_reference=", $row0_reference, "; $row0_target=", $row0_target, "; $row0_hotSpot=", $row0_hotSpot);

        $row.find('select[name=reference]').val($row0_reference);
        $row.find('select[name=targetRegionBedFile]').val($row0_target);
        $row.find('select[name=hotSpotRegionBedFile]').val($row0_hotSpot);

        //auto-assign sample name
        $row.find('.irSampleName').val("Sample " + (i + 1));
    }
    else {
        //auto-assign row number value
        $row.children().eq(0).find('input').attr('value', i + 1);

        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
        initForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, i, $row, $('#row' + previousRowIndex), true); 
    
    }

    return $row;
}

function initForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, rowIndex, row, previousRow, isCreateRow) {
    //(1) for even row, set nucleotide type to DNA; set to RNA if odd row
    //(2) for even row, increment sample name; set to previous row's sample name if odd row
    //(3) auto-populate reference, target region and hotSpot from Reference chevron for the DNA sample
    //(4) disable and disallow user input for RNA's target region and hotSpot

    var isCreate = $('input[id=isCreate]').val();
    var autoSampleName = rowIndex + 1;
    
    if (isEven(rowIndex)) {
        row.children().find('select[name=nucleotideType]').val("DNA");  
        if (isSameSampleForDual) {
            autoSampleName = rowIndex / 2 + 1;
        }
        row.find('.irSampleName').val("Sample " + autoSampleName);
       
        row.children().find('input[name=sampleExternalId]').val('');
        row.children().find('input[name=sampleDescription]').val('');

        row.removeClass("rna");
        row.addClass("dna");
        initSampleRefInfoForDualNucleotideType(row);

        setIRValuesForDualNucleotideType(rowIndex, row, previousRow, isCreateRow);        
    }
    else {
        row.children().find('select[name=nucleotideType]').val("RNA");
        if (isSameSampleForDual) {
            row.children().find('input[name=sampleName]').val(previousRow.find('.irSampleName').val());
            
            var value = previousRow.children().find('input[name=sampleExternalId]').val();            
            row.children().find('input[name=sampleExternalId]').val(value);
            
            value = previousRow.children().find('input[name=sampleDescription]').val();
            row.children().find('input[name=sampleDescription]').val(value);
        }
        else {
            row.find('.irSampleName').val("Sample " + autoSampleName);
        }

        row.removeClass("dna");
        row.addClass("rna");
        initSampleRefInfoForDualNucleotideType(row);
        
        setIRValuesForDualNucleotideType(rowIndex, row, previousRow, isCreateRow);

        if (isSameSampleForDual) {
            disableRNARowForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, rowIndex, row);
        }
    } 
}


function initSampleRefInfoForDualNucleotideType(currentRow) {
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
    var isSameRefInfoPerSample = $('input[id=isSameRefInfoPerSample]').is(":checked");
    
    if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True")) {
        if (currentRow.hasClass("dna")) {
            var selectedReference = $('select[name=default_reference]').val();
            var selectedTargetBedFile = $('select[name=default_targetBedFile]').val();
            var selectedHotSpotBedFile = $('select[name=default_hotSpotBedFile]').val();
            
            currentRow.children().find('select[name=reference]').val(selectedReference);
            currentRow.children().find('select[name=targetRegionBedFile]').val(selectedTargetBedFile);
            currentRow.children().find('select[name=hotSpotRegionBedFile]').val(selectedHotSpotBedFile);  
        }
        else if (currentRow.hasClass("rna")) {
            var mixedTypeRNA_selectedReference = $('select[name=mixedTypeRNA_reference]').val();
            var mixedTypeRNA_selectedTargetBedFile = $('select[name=mixedTypeRNA_targetBedFile]').val();
            
            //console.log("rowIndex=", rowIndex, "; mixedTypeRNA_selectedReference=", mixedTypeRNA_selectedReference, "; mixedTypeRNA_selectedTargetBedFile=", mixedTypeRNA_selectedTargetBedFile);
            
            currentRow.children().find('select[name=reference]').val(mixedTypeRNA_selectedReference);
            currentRow.children().find('select[name=targetRegionBedFile]').val(mixedTypeRNA_selectedTargetBedFile);
                    
            //RNA sample has no hotSpot selection
            currentRow.children().find('select[name=hotSpotRegionBedFile]').val("");
        }
    }
}

function setIRValuesForDualNucleotideType(rowIndex, row, previousRow, isCreateRow) {
    if (!isEven(rowIndex)) {
        var value = previousRow.children().find('select[name=ircancerType]').val();
        row.children().find('select[name=ircancerType]').val(value);

        value = previousRow.children().find('input[name=ircellularityPct]').val();
        row.children().find('input[name=ircellularityPct]').val(value);

        value = previousRow.children().find('select[name=irWorkflow]').val();
        row.children().find('select[name=irWorkflow]').val(value);

        value = previousRow.children().find('select[name=irRelationRole]').val();
        if (value) {
            //the 2nd row may not have the selected value in the drop down yet
            var isExist = false;
            row.children().find('select[name=irRelationRole]').each(function () {
                if (this.value == value) {
                    isExist = true;
                }
            });
            console.log("row selected relation=", value, "; isExist=", isExist);

            if (isExist == false) {
                var options = previousRow.children().find('select[name=irRelationRole] option').clone();

                row.children().find('select[name=irRelationRole]').empty();
                row.children().find('select[name=irRelationRole]').append(options);
            }
            row.children().find('select[name=irRelationRole]').val(value);
        }

        value = previousRow.children().find('select[name=irGender]').val();
        row.children().find('select[name=irGender]').val(value);

        value = previousRow.children().find('input[name=irSetID]').val();
        row.children().find('input[name=irSetID]').val(value);
    }

    if (isCreateRow) {
        row.children().find('input[name=irSetID]').val("");
    }
    
    var selectedWorkflow = previousRow.children().find('select[name=irWorkflow]').val();
    updateIRWorkflowSelectionForDualNucleotideType(previousRow, row, selectedWorkflow);
}

function disableRNARowsForDualNucleotideType(isDualNucleotideType, isSameSampleForDual) {
    if (isDualNucleotideType != "True") {
        return;
    }
    var numRowValue = $('#numRows').val();
    var rowCount = parseInt(numRowValue);
     
    var isCreate = $('input[id=isCreate]').val();
    
    for (var i = 0; i < rowCount; i++) {
        var row = $('#row' + i);

        if (!isEven(i)) {
            if (isSameSampleForDual) {
                var previousRowIndex = i - 1;
                var previousRow = $('#row' + previousRowIndex);
                initForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, i, row, previousRow, false); 
            }
            else {
                disableRNARowForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, i, row);
            }
        }
        else {
            disableIRCells(false, i, row);
        }
    }
}

function disableRNARowForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, rowIndex, row) {
    if (isSameSampleForDual) {
        row.children().find('input[name=sampleName]').attr("disabled", true);
        row.children().find('input[name=sampleExternalId]').attr("disabled", true);
        row.children().find('input[name=sampleDescription]').attr("disabled", true);
                
        row.children().find('select[name=hotSpotRegionBedFile]').attr("disabled", true);

        var isToDisable = true;
        if (isEven(rowIndex)) {
            isToDisable = false;
            row.children().find('select[name=ircancerType]').removeAttr("disabled");
            row.children().find('input[name=ircellularityPct]').removeAttr("disabled");
        }
        else {
            row.children().find('select[name=ircancerType]').attr("disabled", true);
            row.children().find('input[name=ircellularityPct]').attr("disabled", true);
        
        }
        disableIRCells(isToDisable, rowIndex, row);   
    }   
    else {
        row.children().find('input[name=sampleName]').removeAttr("disabled");
        row.children().find('input[name=sampleExternalId]').removeAttr("disabled");
        row.children().find('input[name=sampleDescription]').removeAttr("disabled");

        row.children().find('select[name=ircancerType]').removeAttr("disabled");
        row.children().find('input[name=ircellularityPct]').removeAttr("disabled");
 
        disableIRCells(false, rowIndex, row);
    }
}


function disableIRCells(isToDisable, rowIndex, row) {
    if (isToDisable) {
        //row.children().find('select[name=ircancerType]').attr("disabled", true);
        //row.children().find('input[name=ircellularityPct]').attr("disabled", true);

        row.children().find('select[name=irWorkflow]').attr("disabled", "disabled");
        row.children().find('select[name=irRelationRole]').attr("disabled", true);  
        row.children().find('select[name=irGender]').attr("disabled", true);  
        row.children().find('input[name=irSetID]').attr("disabled", true);  
    }
    else {
        //row.children().find('select[name=ircancerType]').removeAttr("disabled");
        //row.children().find('input[name=ircellularityPct]').removeAttr("disabled");

        row.children().find('select[name=irWorkflow]').removeAttr("disabled");
        row.children().find('select[name=irRelationRole]').removeAttr("disabled");
        row.children().find('select[name=irGender]').removeAttr("disabled");
        row.children().find('input[name=irSetID]').removeAttr("disabled");
    }
}

function updateIRWorkflowSelectionForDualNucleotideType(currentRow, nextRow, selectedWorkflow) {
    if (isEven(currentRow.index())) {
        nextRow.children().find('select[name=irWorkflow]').val(selectedWorkflow);

        //clear the 2nd row's relation selected value
        nextRow.children().find('select[name=irRelationRole]').val("");
        //if the 1st row only has 1 value, it will be auto-select and change won't be triggered
        relationValue = currentRow.children().find('select[name=irRelationRole]').val();
        
        //console.log("updateIRWorkflowSelectionForDualNucleotideType() selectedWorkflow=", selectedWorkflow, "; relationValue=", relationValue);

        if (selectedWorkflow && !relationValue) {
            //select the first non-blank value if that is the only non-blank choice
            var count = currentRow.children().find('select[name=irRelationRole] option').length;
            if (count == 2) {
                currentRow.children().find('select[name=irRelationRole] :nth-child(2)').prop('selected', true); // To select via index
                relationValue = currentRow.children().find('select[name=irRelationRole]').val();
            }
            //console.log("updateIRWorkflowSelectionForDualNucleotideType() AGAIN!! relationValue=", relationValue);
        }
        
        if (relationValue) {
            //the 2nd row may not have the selected value in the drop down
            var isExist = false;
            nextRow.children().find('select[name=irRelationRole]').each(function () {
                if (this.value == relationValue) {
                    isExist = true;
                }
            });
            if (isExist == false) {
                var options = currentRow.children().find('select[name=irRelationRole] option').clone();

                nextRow.children().find('select[name=irRelationRole]').empty();
                nextRow.children().find('select[name=irRelationRole]').append(options);
            }
            nextRow.children().find('select[name=irRelationRole]').val(relationValue);
        }

        var irSetID = currentRow.children().find('input[name=irSetID]').val();
        nextRow.children().find('input[name=irSetID]').val(irSetID);
    }
}

//20150205TODO - refactor to take advantage of the generalized code in this file
function initForBasicDualNucleotideType() {
    //(1) auto-set number of barcode count to 2;
    //(2) auto-set nucleotide type to DNA on row 1 and RNA on row 2;
    //(3) disable sample name, sample id and description ui widgets on row 2
    //(4) auto-populate reference, target region and hotSpot from Reference chevron for the DNA sample
    //(5) disable and disallow user input for RNA's target region and hotSpot
    
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    //console.log("typeof isDualNucleotideType=", Object.prototype.toString.call(isDualNucleotideType));
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

    var isCreate = $('input[id=isCreate]').val();

    if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True")) {
        disableRNARowsForDualNucleotideType(isDualNucleotideType, isSameSampleForDual);             
        var numRowValue = $('#numRows').val();
        //console.log("current numRowValue=", numRowValue);
        //console.log("typeof numRowValue=", Object.prototype.toString.call(numRowValue));

        if (isCreate != "True") {
            setRowsMetaDataForDualNucleotideType(isDualNucleotideType, isSameSampleForDual);             
            return;
        }
        if (parseInt(numRowValue) > 1) {
            setRowsMetaDataForDualNucleotideType(isDualNucleotideType, isSameSampleForDual);             
            return;
        }
        
        $('#numRows').val("2");
        $('#numRows').change();

        var row0 = $('#row0');
        var row1 = $('#row1');
    
        row0.children().find('select[name=nucleotideType]').val("DNA");
        row0.removeClass("rna");
        row0.addClass("dna");
        
        row1.children().find('select[name=nucleotideType]').val("RNA");
        row1.removeClass("dna");
        row1.addClass("rna");
        
        //do not clear selection or data loss during edit!! row1.children().find('select[name=targetRegionBedFile]').val("");
        row1.children().find('select[name=hotSpotRegionBedFile]').val("");

        //20140227-WIP row1.children().find('select[name=reference]').attr("disabled",  true);
        //row1.children().find('select[name=targetRegionBedFile]').attr("disabled", true);
        row1.children().find('select[name=hotSpotRegionBedFile]').attr("disabled", true);

    //now that we can have default RNA ref info, the following is no longer applicable
    /*
        //applicable regardless if isSameSampleForDual
        if (isCreate == "True") {
            console.log("at page_plan_sample_table.initForBasicDualNucleotideType - isCreate!!!");
            row1.children().find('select[name=reference]').val("");
        }
    */

        var isPlanBySample = $('input[id=isPlanBySample]').val();

        if (isSameSampleForDual) {
            if (isPlanBySample == "True") {
                var value = row0.children().find('select[name=sampleName]').val();

                row1.children().find('select[name=sampleName]').val(value);
                row1.children().find('select[name=sampleName]').attr("disabled", true);
            }
            else {
                var value = row0.children().find('input[name=sampleName]').val();

                row1.children().find('input[name=sampleName]').val(value);
                row1.children().find('input[name=sampleName]').attr("disabled", true);
            }

            var value = row0.children().find('input[name=sampleExternalId]').val();
            row1.children().find('input[name=sampleExternalId]').val(value);

            value = row0.children().find('input[name=sampleDescription]').val();
            row1.children().find('input[name=sampleDescription]').val(value);

            value = row0.children().find('select[name=ircancerType]').val();
            row1.children().find('select[name=ircancerType]').val(value);

            value = row0.children().find('input[name=ircellularityPct]').val();
            row1.children().find('input[name=ircellularityPct]').val(value);

            value = row0.children().find('select[name=irWorkflow]').val();
            row1.children().find('select[name=irWorkflow]').val(value);

            value = row0.children().find('select[name=irRelationRole]').val();

            if (value) {

                //the 2nd row may not have the selected value in the drop down yet
                var isExist = false;
                row1.children().find('select[name=irRelationRole]').each(function () {
                    if (this.value == value) {
                        isExist = true;
                    }
                });
                console.log("first row selected relation=", value, "; isExist=", isExist);

                if (isExist == false) {
                    var options = row0.children().find('select[name=irRelationRole] option').clone();

                    row1.children().find('select[name=irRelationRole]').empty();
                    row1.children().find('select[name=irRelationRole]').append(options);
                }
                row1.children().find('select[name=irRelationRole]').val(value);
            }

            value = row0.children().find('select[name=irGender]').val();
            row1.children().find('select[name=irGender]').val(value);

            value = row0.children().find('input[name=irSetID]').val();
            row1.children().find('input[name=irSetID]').val(value);
            disableRNARowForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, 1, row1);
        }
        else {
            if (isPlanBySample == "True") {
                row1.children().find('select[name=sampleName]').removeAttr("disabled");
            }
            else {
                row1.children().find('input[name=sampleName]').removeAttr("disabled");
            }
            disableRNARowForDualNucleotideType(isDualNucleotideType, isSameSampleForDual, 1, row1);
        }

        //var currentValue = row1.children().find('select[name=reference]').val();

        //selected plan reference is for DNA sample
        //if (IS_CREATE == "True" && currentValue == SELECTED_PLAN_REFERENCE) {
//        if (IS_CREATE == "True") {
//        	console.log("Going to clear row1 reference value. currentValue=", currentValue, "; SELECTED_PLAN_REFERENCE=", SELECTED_PLAN_REFERENCE);
//            row1.children().find('select[name=reference]').val("");        	
//        }
    }
}


function setRowsMetaDataForDualNucleotideType(isDualNucleotideType, isSameSampleForDual) {
    if (isDualNucleotideType != "True") {
        return;
    }
    var numRowValue = $('#numRows').val();
    var rowCount = parseInt(numRowValue);
   
    for (var i = 0; i < rowCount; i++) {
        var row = $('#row' + i);
        
        var value = row.children().find('select[name=nucleotideType]').val(); 
        if (value.toLowerCase() == "dna") {
            row.removeClass("rna");
            row.addClass("dna")        
        }  
        else if (value.toLowerCase() == "rna") {
            row.removeClass("dna");
            row.addClass("rna")        
        }
    }
}

/**
 *   Auto-hide summary panel if needed
 */
prepareSampleIRConfiguration = function () {
    var selectedIRAccount = USERINPUT.account_name ? USERINPUT.account_name : "None";

    if (selectedIRAccount != "None" && USERINPUT.is_ir_connected) {
        $("#sidebar").hide();
        $("#mainContent").removeClass("span8");
        $("#mainContent").addClass("span12");
        $("#showSummary").show();
    }
}


function updateSampleReferenceColumnsWithDefaults(defaultReference, defaultTargetBedFile, defaultHotSpotBedFile, mixedTypeRNA_reference, mixedTypeRNA_targetBedFile) {    
    updateSamplesForReference(defaultReference, false);
    updateSamplesForTargetRegion(defaultTargetBedFile, false);
    updateSamplesForHotSpot(defaultHotSpotBedFile, false);

    updateMixedTypeRNASamplesForReference(mixedTypeRNA_reference, false);
    updateMixedTypeRNASamplesForTargetRegion(mixedTypeRNA_targetBedFile, false);
        
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();

    if (isDualNucleotideType == "True"){
        $(".hideable_referenceBySample_hotSpot").each(function (index, value) {

            //here, table header is being iterated as well    
            //process RNA sample row    
            var row = $("#row" + index);
            if (row && row.hasClass("rna")) {
                row.children().find('select[name=hotSpotRegionBedFile]').val("");
            }
        });
    }
    
    //console.log("done with updateSampleReferenceColumnsWithDefaults - GOING to updateSamplesTable NOW");
    updateSamplesTable();
}


function updateSamplesForReference(defaultReference, isToUpdateSamplesTableNow) {
	var isReferenceSupported = $('input[id=isReferenceSupported]').val();
	if (isReferenceSupported != "True") {
		return;
	}
	
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();

    //20141002-WIP-TODO - needs to filter OCP BED file selection!! 
    if (isDualNucleotideType == "True"){    
        $(".hideable_referenceBySample_ref").each(function (index, value) {      
            //console.log("updateSamplesForReference...index=", index, "; value=", value );

            //here, table header is being iterated as well  
            //process DNA sample row
            var row = $("#row" + index);
            if (row && row.hasClass("dna")) {
                row.children().find('select[name=reference]').val(defaultReference).change();
            }
        });    
    }
    else {
        $("select[name=reference]").each(function(){
            $(this).val(defaultReference).change();
        });    
    }
        
	if (isToUpdateSamplesTableNow == true) {
        updateSamplesTable();
    }
}
   

function updateMixedTypeRNASamplesForReference(planReference, isToUpdateSamplesTableNow) {
	var isReferenceSupported = $('input[id=isReferenceSupported]').val();
	if (isReferenceSupported != "True" ) {
		return;
	}
	
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();

    if (isDualNucleotideType == "True"){
        $(".hideable_referenceBySample_ref").each(function (index, value) {     
            //console.log("updateMixedTypeRNASamplesForReference...index=", index, "; value=", value );

            //here, table header is being iterated as well
            //process RNA sample row  
            var row = $("#row" + index);
            if (row && row.hasClass("rna")) {
                row.children().find('select[name=reference]').val(planReference).change();
            }            
        });    
    }

	if (isToUpdateSamplesTableNow == true) {
        updateSamplesTable();
    }
}


function updateSamplesForTargetRegion(defaultTargetBedFile, isToUpdateSamplesTableNow) { 
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();

    if (isDualNucleotideType == "True"){
        $(".hideable_referenceBySample_targetRegion").each(function (index, value) {   
            //console.log("updateSamplesForTargetRegion...index=", index, "; value=", value );

            //here, table header is being iterated as well    
            //process DNA sample row
            var row = $("#row" + index);
            if (row && row.hasClass("dna")) {
                row.children().find('select[name=targetRegionBedFile]').val(defaultTargetBedFile);
            }                        
        });           
    }
    else {
        $("select[name=targetRegionBedFile]").each(function(){
            $(this).val(defaultTargetBedFile);
        });        
    }
    
	if (isToUpdateSamplesTableNow == true) {
        updateSamplesTable();
    }
}
   

function updateMixedTypeRNASamplesForTargetRegion(planTargetBedFile, isToUpdateSamplesTableNow) {    
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();

    if (isDualNucleotideType == "True"){
        $(".hideable_referenceBySample_targetRegion").each(function (index, value) {      
            //console.log("updateMixedTypeRNASamplesForTargetRegion...index=", index, "; value=", value );

            //here, table header is being iterated as well   
            //process RNA sample row     
            var row = $("#row" + index);
            if (row && row.hasClass("rna")) {
                row.children().find('select[name=targetRegionBedFile]').val(planTargetBedFile);
            }                        
        });     
    }

	if (isToUpdateSamplesTableNow == true) {
        updateSamplesTable();
    }
}


function updateSamplesForHotSpot(defaultHotSpot, isToUpdateSamplesTableNow) {
    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
    
    if (isDualNucleotideType == "True"){
        $(".hideable_referenceBySample_hotSpot").each(function (index, value) {      
            //console.log("updateSamplesForHotSpot...index=", index, "; value=", value );

            //here, table header is being iterated as well  
            //process DNA sample row  
            var row = $("#row" + index);
            if (row && row.hasClass("dna")) {
                row.children().find('select[name=hotSpotRegionBedFile]').val(defaultHotSpot);
            }                      
        });      
    }
    else {
        $("select[name=hotSpotRegionBedFile]").each(function(){
            $(this).val(defaultHotSpot);
        });
    }
    
	if (isToUpdateSamplesTableNow == true) {
        updateSamplesTable();
    }
}



showSampleReferenceColumns = function(isReferenceSupported, isTargetRegionBEDFileSupported, isHotspotRegionBEDFileSupported, isTargetRegionBEDFileBySampleSupported, isHotSpotBEDFileBySampleSupported) {
    if (isReferenceSupported == "True" ) {
        $(".hideable_referenceBySample_ref").each(function (index, value) {            
            //console.log("going to SHOW hideable_referenceBySample_ref...index=", index, "; value=", value, "; attr(id)=", $(this).attr("id"), "this.id=", $(this).id );
            $(this).show();
        });
    }
    
    if (isTargetRegionBEDFileSupported == "True" && isTargetRegionBEDFileBySampleSupported == "True") {
        $(".hideable_referenceBySample_targetRegion").each(function (index, value) {            
            //console.log("going to SHOW hideable_referenceBySample_targetRegionf...index=", index, "; value=", value, "; attr(id)=", $(this).attr("id"), "this.id=", $(this).id );
            $(this).show();
        });    
    }
    
    if (isHotspotRegionBEDFileSupported == "True" && isHotSpotBEDFileBySampleSupported == "True") {
        $(".hideable_referenceBySample_hotSpot").each(function (index, value) {            
            //console.log("going to SHOW hideable_referenceBySample_hotSpot_ref...index=", index, "; value=", value, "; attr(id)=", $(this).attr("id"), "this.id=", $(this).id );
            $(this).show();
        });    
    }    
}


toggleSampleReferenceColumnEnablements = function(isToDisable, isReferenceSupported, isTargetRegionBEDFileSupported, isHotspotRegionBEDFileSupported, isTargetRegionBEDFileBySampleSupported, isHotSpotBEDFileBySampleSupported) {
	if (isReferenceSupported == "True") {
		$(".hideable_referenceBySample_ref").each(function (index, value) {
			//console.log("going to enable/disable hideable_referenceBySample_ref...index=", index, "; value=", value, "; attr(id)=", $(this).attr("id"), "this.id=", $(this).id );
			if (isToDisable) {
				$(this).find('select').attr("disabled", true);
			}
			else {
				$(this).find('select').removeAttr("disabled");
			}
		});
	}
	
    if (isTargetRegionBEDFileSupported == "True" && isTargetRegionBEDFileBySampleSupported == "True") {
        $(".hideable_referenceBySample_targetRegion").each(function (index, value) {            
            //console.log("going to enable/disable hideable_referenceBySample_ref...index=", index, "; value=", value, "; attr(id)=", $(this).attr("id"), "this.id=", $(this).id );
            if (isToDisable) {
               $(this).find('select').attr("disabled", true);
            }
            else {
                $(this).find('select').removeAttr("disabled");
            }
        });    
    }
    
    if (isHotspotRegionBEDFileSupported == "True" && isHotSpotBEDFileBySampleSupported == "True") {
        $(".hideable_referenceBySample_hotSpot").each(function (index, value) {
            //console.log("going to enable/disable hideable_referenceBySample_hotSpot_ref...index=", index, "; value=", value, "; attr(id)=", $(this).attr("id"), "this.id=", $(this).id );
            if (isToDisable) {
               $(this).find('select').attr("disabled", true);
            }
            else {               
                $(this).find('select').removeAttr("disabled");

                //need to keep DNA+Fusion's RNA sample's hot spot disabled at all time        
                var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();   
                //here, table header is being iterated as well       
                //if (isDualNucleotideType == "True" && isEven(index)){
                if (isDualNucleotideType == "True"){
                    $('#row'+ index).children().find('select[name=hotSpotRegionBedFile]').attr("disabled", true);
                } 
            }
        });    
    }    
}       


$(document).ready(function () {

    function parse_chip_barcode(_chip_barcode){
            var chipID = _chip_barcode;
            if (_chip_barcode.substring(0,2) == "21")
            {
                 var sub_barcode = _chip_barcode.substring(2);
                 var reverse_subbarcode = sub_barcode.split("").reverse().join("");
                 var n1 = reverse_subbarcode.search("19");
                 var n2 = reverse_subbarcode.search("142");
                 if ((n1 != -1) && ((n2 == -1) || (n1 < n2)))
                 {
                     chipID = sub_barcode.substring(0,sub_barcode.length-n1-("91").length);
                 }
                 else if ((n2 != -1) && ((n1 == -1) || (n2 < n1)))
                 {
                     chipID = sub_barcode.substring(0,sub_barcode.length-n2-("241").length);
                 }
            }
            return chipID;
    }
    $("#chipBarcodeLabel").on('change', function(){
        $(this).val(parse_chip_barcode($(this).val()));
    });
    $('input[name="chipBarcode"]').on('change', function(){
        $(this).val(parse_chip_barcode($(this).val()));
    });

    $("#barcodeSampleTubeLabel").on('keyup', function () {
        $("#tubeerror").html('');
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            $("#tubeerror").html('Error, Sample tube label should contain only numbers, letters, spaces, and the following: . - _');
        }
        //call the check max length function that's in validation.js
        if (!is_valid_length($(this).val(), 512)) {
            $("#tubeerror").html('Error, Sample tube label length should be 512 characters maximum');
        }
    });

    $(".irSampleName").on('keyup', function () {
        var $td = $(this).parent();
        var $h;
        if ($td.find('h4').length > 0) {
            $h = $td.find('h4');
            $h.remove();
        }
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Error, Sample name should contain only numbers, letters, spaces, and the following: . - _');
            $td.append($h);
        }
        //call the check max length function that's in validation.js
        if (!is_valid_length($(this).val(), 127)) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Error, Sample name length should be 127 characters maximum');
            $td.append($h);
        }

        if (!is_valid_leading_chars($(this).val())) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Sample name cannot begin with (.) or (_) or (-)');
            $td.append($h);
        }

    });

    /**
     Click event handler to fill the sample names
     */
    $('#fillSampleNames').click(function () {
        $('input[name="sampleName"]').each(function (index, value) {
            var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();

            if (isDualNucleotideType == "True") {
                if (index == 0) {
                    $(this).val('barcode_' + (index + 1));
                }
                else {
                    var previousIndex = index - 1;
                    $(this).val('barcode_' + (index + 1));
                }
            }
            else {
                $(this).val('Sample ' + (index + 1));
            }
        });
        updateSamplesTable();
    });

    /**
     Click event handler to clear the sample names
     */
    $('#clearSampleNames').click(function () {
        $('input[name="sampleName"]').each(function (index, value) {
            $(this).val('');
        });
        updateSamplesTable();
    });
 
    $(".ircoupleId").on('keyup', function () {
        var $td = $(this).parent();
        var $h;
        if ($td.find('h4').length > 0) {
            $h = $td.find('h4');
            $h.remove();
        }
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Error, Couple ID should contain only numbers, letters, spaces, and the following: . - _');
            $td.append($h);
        }
        //call the check max length function that's in validation.js
        if (!is_valid_length($(this).val(), 127)) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Error, Couple ID length should be 127 characters maximum');
            $td.append($h);
        }
    }); 

    $(".irembryoId").on('keyup', function () {
        var $td = $(this).parent();
        var $h;
        if ($td.find('h4').length > 0) {
            $h = $td.find('h4');
            $h.remove();
        }
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Error, Embryo ID should contain only numbers, letters, spaces, and the following: . - _');
            $td.append($h);
        }
        //call the check max length function that's in validation.js
        if (!is_valid_length($(this).val(), 127)) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Error, Embryo ID length should be 127 characters maximum');
            $td.append($h);
        }
    }); 
    
    /**
     *   Click event handler to show/hide the sample reference columns
     */
    $("[id^=showHideReferenceBySample]").click(function () {    
    	var isReferenceSupported = $('input[id=isReferenceSupported]').val();
        var isTargetBEDFileSupported = $('input[id=isTargetBEDFileSupported]').val();
        var isHotSpotBEDFileSupported = $('input[id=isHotSpotBEDFileSupported]').val();
        var isTargetBEDFileBySampleSupported = $('input[id=isTargetBEDFileBySampleSupported]').val();
        var isHotSpotBEDFileBySampleSupported = $('input[id=isHotSpotBEDFileBySampleSupported]').val();

        if (isReferenceSupported == "True" ) {
            $(".hideable_referenceBySample_ref").each(function (index, value) {
                $(this).toggle();
            });
        }

        if (isTargetBEDFileSupported == "True" && isTargetBEDFileBySampleSupported == "True") {
            $(".hideable_referenceBySample_targetRegion").each(function (index, value) {            
                $(this).toggle();
            });    
        }
        
        if (isHotSpotBEDFileSupported == "True" && isHotSpotBEDFileBySampleSupported == "True") {
            $(".hideable_referenceBySample_hotSpot").each(function (index, value) {            
                //console.log("going to toggle hideable_referenceBySample_hotSpot_ref...index=", index, "; value=", value, "; attr(id)=", $(this).attr("id"), "this.id=", $(this).id );
                $(this).toggle();
            });    
        }    
    });

    /**
     *   Click event handler to show/hide the sample annotation columns
     */
    $("#showHideSampleAnnotation").click(function () {
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var selectedIRAccount = USERINPUT.account_name ? USERINPUT.account_name : "None";

        if (isDualNucleotideType == "True") {
            $(".hideable_sampleAnnotation_nt").toggle();
        }

		var isOnco_Pgs = $("input[name=isOnco_Pgs]:checked").val()
		if (isOnco_Pgs == 'Pgs') {
            $(".hideable_sampleAnnotation_pgs").toggle();
        }

        if (isOnco_Pgs == 'Oncology') {
            $(".hideable_sampleAnnotation_onco").toggle();
        }
    });


    $(".ircellularityPct").on('keyup', function () {
        var $td = $(this).parent();

        var $h;
        if ($td.find('h4').length > 0) {
            $h = $td.find('h4');
            $h.remove();
        }

        //call the test function identified in validation.js file
        if (!is_valid_percentage($(this).val())) {
            $h = $("<h4></h4>", {'style': 'color:red;'});
            $h.text('Cellularity % value should be between 0 to 100.');
            $td.append($h);
        }
    });

    /**
     *   Disallow non-integer user-input
     */
    $(".integer").keydown(function (event) {
        /* restrict user input for integer fields */
        if (event.shiftKey)
            if (event.keyCode != 9)
                event.preventDefault();
        if (event.keyCode == 46 || event.keyCode == 8 || event.keyCode == 9) {
        }
        else {
            if (event.keyCode < 95) {
                if (event.keyCode < 48 || event.keyCode > 57) {
                    event.preventDefault();
                }
            } else {
                if (event.keyCode < 96 || event.keyCode > 105) {
                    event.preventDefault();
                }
            }
        }
    });


    /**
     Click event handler for the refresh button which clears the IR fields,
     and re-calls the API to re-populate the IR fields
     */
    $(".refresh-uploader-information").on('click', function () {
        $("#loading").show();

        if (USERINPUT.irWorkflowSelects !== undefined) {
            USERINPUT.irGenderSelects.empty();
            USERINPUT.irWorkflowSelects.empty();
            USERINPUT.irRelationRoleSelects.empty();
        }

        load_and_set_ir_fields();
        return false;
    });

    //making the refresh icon have a hand when the mouse hovers over it
    $(".refresh-uploader-information").css('cursor', 'pointer');


    /**
     Barcoded and Non-barcoded tables create/update/show
     */
    $("input[name=is_barcoded]").click(function () {
        if (this.id == 'chk_barcoded') {
            $('.barcoded').show();
            $('.nonbarcoded').hide();
            $('#barcodeSet').prop('selectedIndex', 1);
            $('#barcodeSet').change();
            $('#numRowsLabel').text("Number of barcodes");
        } else {
            $('.barcoded').hide();
            $('.nonbarcoded').show();
            $('#barcodeSet').val('');
            $('#numRowsLabel').text("Number of chips");
        }
        $('#numRows').change();
    });


    $('#barcodeSet').change(function () {
        var barcodes = BARCODES[this.value];
        if (barcodes) { 
            $('input[id=chk_barcoded]').attr("checked", true);
            $('input[id=chk_not_barcoded]').attr("checked", false);

             // show the barcode-related columns
            $(".barcoded").show();
            //hide the non-barcoded related columns
            $(".nonbarcoded").hide();           
                
        var num_barcodes = barcodes.length;

        // if more existing rows than number of barcodes need to update numRows and table
        if ($('#chipsets').find('tr').length - 1 > num_barcodes) {
            $('#numRows').val(num_barcodes);
            $('#numRows').change();
        }

        // update all barcode selection dropdowns
        $('[name="barcode"]').each(function (n) {
            var $select = $(this);
            var num_options = $select.find('option').length;
            // replace or add options
            for (var i = 0; i < num_barcodes; i++) {
                var $option;
                if (num_options > i) {
                    // update existing option
                    $option = $select.find('option').eq(i);
                    $option.data("id_str", barcodes[i]["id_str"]);
                    $option.text(barcodes[i]["id_str"] + "(" + barcodes[i]["sequence"] + ")");
                } else {
                    // add new option
                    $option = $("<option></option>", {
                        "value": i,
                        "text": barcodes[i]["id_str"] + "(" + barcodes[i]["sequence"] + ")",
                        "data-id_str": barcodes[i]["id_str"]
                    });
                    $select.append($option);
                }
            }
            $select.val(n)
            // if new set has less barcodes than old - remove extra options
            if (num_options > num_barcodes) {
                $select.find("option:gt(" + (num_barcodes - 1) + ")").remove();
            }
        });
        }
        else {        
            $('input[id=chk_barcoded]').attr("checked", false);
            $('input[id=chk_not_barcoded]').attr("checked", true);
    
            // hide the barcode-related columns
            $(".barcoded").hide();
            //show the non-barcoded related columns
            $(".nonbarcoded").show();
            
            updateSamplesTable();
        }
        
        $('#chipsets').change();
    });


    $('#numRows').change(function () {
        var $table = $('#chipsets');
        var nrows = $table.find('tr').length - 1;

        if ($('input[id=chk_barcoded]').is(':checked')) {
            // limit to number of barcodes in set
            var num_barcodes = BARCODES[$('#barcodeSet').val()].length;
            //20150212-TEMP until anything goes-OCP is live - for OCP sample pairs need to be even number
            var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
            var numRowsValue = parseInt(this.value);

            //this.value = (this.value > num_barcodes) ? num_barcodes : this.value;
            this.value = (numRowsValue > num_barcodes) ? num_barcodes : numRowsValue;
        }

        if (this.value > nrows) {
            for (var i = nrows; i < this.value; i++) {
                var row = createRow(i);
                row.appendTo($table.find('tbody'));
                row.find('select[name=irWorkflow]').change();
            }
        } else if (this.value < nrows) {
            $table.find("tbody tr:gt(" + (this.value - 1) + ")").remove();
        }
        $table.change();
    });


    $('input[name=sampleName]').on('keyup', function () {

        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        //console.log("typeof isDualNucleotideType=", Object.prototype.toString.call(isDualNucleotideType));
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('input[name=sampleName]').val(value);
            }
        }
    });


    $('select[name=sampleName]').change(function () {

        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('select[name=sampleName]').val(value);
            }
        }
    });


    $('input[name=sampleExternalId]').on('keyup', function () {  //.change(function(){
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('input[name=sampleExternalId]').val(value);
            }
        }
    });


    $('input[name=sampleExternalId]').change(function () {
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('input[name=sampleExternalId]').val(value);
            }
        }
    });


    $('input[name=sampleDescription]').on('keyup', function () {  //.change(function(){
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('input[name=sampleDescription]').val(value);
            }
        }
    });


    $('input[name=sampleDescription]').change(function () {
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('input[name=sampleDescription]').val(value);
            }
        }
    });

    $('select[name=nucleotideType]').change(function () {
        var currentRow = $(this).closest('tr');                
        var value = $(this).val().toLowerCase();

        if (value == "dna") {
         currentRow.removeClass("rna");
         currentRow.addClass("dna");
        }
        if (value == "rna") {
         currentRow.removeClass("dna");
         currentRow.addClass("rna");
        }
        initSampleRefInfoForDualNucleotideType(currentRow);
    });
    
    $('select[name=ircancerType]').change(function () {
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            //console.log("ircancerType current row index=", currentRow.index());            
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('select[name=ircancerType]').val(value);
            }
        }
    });

    $('input[name=ircellularityPct]').on('keyup', function () {
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('input[name=ircellularityPct]').val(value);
            }
        }
    });


    $('select[name=irWorkflow]').live('change', function (e) {

        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            //console.log("### irWorkflow change - currentRow.index=", currentRow.index());
            var nextRowIndex = currentRow.index() + 1;
            var nextRow = $('#row' + nextRowIndex);
            var value = $(this).val();
            
            updateIRWorkflowSelectionForDualNucleotideType(currentRow, nextRow, value);
        }
    });


    $('select[name=irRelationRole]').live('change', function (e) {

        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');           
            //console.log("### irRelationRole change - currentRow.index=", currentRow.index());
            
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();

                //the 2nd row may not have the selected value in the drop down
                var isExist = false;
                nextRow.children().find('select[name=irRelationRole]').each(function () {
                    if (this.value == value) {
                        isExist = true;
                    }
                });

                if (isExist == false) {
                    var options = currentRow.children().find('select[name=irRelationRole] option').clone();

                    nextRow.children().find('select[name=irRelationRole]').empty();
                    nextRow.children().find('select[name=irRelationRole]').append(options);
                }

                nextRow.children().find('select[name=irRelationRole]').val(value);
            }
        }
    });


    $('select[name=irGender]').live('change', function (e) {

        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('select[name=irGender]').val(value);
            }
        }
    });


    $('input[name=irSetID]').live('change', function (e) {

        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
        var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");
        //console.log("isSameSampleForDual=", isSameSampleForDual);

        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual)) {
            var currentRow = $(this).closest('tr');
            if (isEven(currentRow.index())) {
                var nextRowIndex = currentRow.index() + 1;
                var nextRow = $('#row' + nextRowIndex);

                var value = $(this).val();
                nextRow.children().find('input[name=irSetID]').val(value);
            }
        }
    });

    /**
     Checkbox for same sample in a DNA + RNA plan is clicked
     */
    $("input[name=isOncoSameSample]").click(function () {
        var isSameSampleForDual = $(this).is(":checked");
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        
        disableRNARowsForDualNucleotideType(isDualNucleotideType, isSameSampleForDual);
        updateSamplesTable();
    });



    /**
     Checkbox for isOnco and isPgs is clicked
     */
    $("input[name=isOnco_Pgs]").click(function () {
        var isOnco_Pgs = $("input[name=isOnco_Pgs]:checked").val();

        //var selectedIRAccount = USERINPUT.account_name ? USERINPUT.account_name : "None";
        //if (selectedIRAccount != "None" && USERINPUT.is_ir_connected) {}

        if (isOnco_Pgs == 'Oncology') {
            $(".hideable_sampleAnnotation_onco").show();
         }
        else {
            $(".hideable_sampleAnnotation_onco").hide();        	
        }

        if (isOnco_Pgs == 'Pgs') {
            $(".hideable_sampleAnnotation_pgs").show();
         }
        else {
            $(".hideable_sampleAnnotation_pgs").hide();
        }
    });

    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

    if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True")) {
        initForBasicDualNucleotideType();
    }
    else {
        $("#numRows").removeAttr("disabled");
        
        //20150203TODO-do we still need this?
        if ($('#numRows').val() > 1) {
            var row1 = $('#row1');
            row1.children().find('input[name=sampleName]').removeAttr("disabled");
            row1.children().find('input[name=sampleExternalId]').removeAttr("disabled");
            row1.children().find('input[name=sampleDescription]').removeAttr("disabled");
            row1.children().find('select[name=targetRegionBedFile]').removeAttr("disabled");
            row1.children().find('select[name=hotSpotRegionBedFile]').removeAttr("disabled");

            row1.children().find('select[name=ircancerType]').removeAttr("disabled");
            row1.children().find('select[name=ircellularityPct]').removeAttr("disabled");

            row1.children().find('select[name=irWorkflow]').removeAttr("disabled");
            row1.children().find('select[name=irRelationRole]').removeAttr("disabled");

            row1.children().find('select[name=irGender]').removeAttr("disabled");
            row1.children().find('input[name=irSetID]').removeAttr("disabled");
        }
    }

    //Enable a fill down functionality on the sample/barcode grid.

    //Always enable on these fields if present.
    var fillDownElements = [
        //Select Elements
        {selector: ".ircancerType", action: "copy"},
        {selector: "select[name='irWorkflow']", action: "copy"},
        {selector: ".irRelationRole", action: "copy"},
        {selector: ".irGender", action: "copy"},

        //Input Elements - Still Works!
        {selector: ".ircellularityPct", action: "copy"},
        {selector: ".irSetID", action: "increment"},
        {selector: ".irbiopsyDays", action: "copy"},
        {selector: ".ircoupleId", action: "copy"},
        {selector: ".irembryoId", action: "copy"},
    ];

    //Only enable certain fields if not OCP planning.
    if ($('input[id=isDualNucleotideTypeBySample]').val() != "True") {
        fillDownElements = fillDownElements.concat([
            {selector: ".reference", action: "copy"},
            {selector: "select[name='targetRegionBedFile']", action: "copy"},
            {selector: "select[name='hotSpotRegionBedFile']", action: "copy"}
        ]);
    }

    var tableContainer = $("#chipsets").css("position", "relative");

    $.each(fillDownElements, function (_, options) {
        var elementSelector = options.selector;
        var action = options.action;

        var leaderSelectElementSelector = elementSelector + ":first";
        var followerSelectElementsSelector = elementSelector + ":not(:first)";
        var fillDownButton = $("<div class='btn btn-primary btn-mini'><i class='icon-circle-arrow-down icon-white'></i></div>")
            .css("border-radius", 20)
            .css("border-top-right-radius", 0)
            .css("border-bottom-right-radius", 0)
            .css("position", "absolute")
            .css("line-height", "1px")
            .css("left", -1000) //FireFox bug stops me from hiding the element.
            .appendTo("#chipsets");
        fillDownButton.tooltip({title: "Copy value to all rows."});
        fillDownButton.mousedown(function () {
            var val = $(leaderSelectElementSelector).val();
            tableContainer.find(followerSelectElementsSelector).each(function (_, selectElement) {
                if (action == "increment" && !isNaN(val)) {
                    if($("#isOncoSameSample:visible").length && $("#isOncoSameSample").attr('checked')){
                        val = parseInt(val);
                    } else {
                        val = parseInt(val) + 1;
                    }
                }
                $(selectElement).val(val);
                $(selectElement).keydown();
                $(selectElement).keyup();
                $(selectElement).change();
            });
        });

        tableContainer.on('focus', leaderSelectElementSelector,
            function () {
                fillDownButton.css("top", $(leaderSelectElementSelector).position().top + 6);
                fillDownButton.css("left", $(leaderSelectElementSelector).position().left - 27);
            }
        );

        tableContainer.on('blur', leaderSelectElementSelector,
            function () {
                fillDownButton.css("left", -1000); //FireFox bug stops me from hiding the element.
            }
        );
    });

    //Make the reference select box filter the bed file selection drop-downs.

    $("select[name=reference]").each(function(){
        //Create a hidden select to keep the initial list intact for cloning. You can't hide select options with css.
        var targetBedSelect = $(this).closest("tr").find("select[name=targetRegionBedFile]");
        var hotspotBedSelect = $(this).closest("tr").find("select[name=hotSpotRegionBedFile]");

        var hiddenTargetBedSelect = targetBedSelect.clone().appendTo($(this).closest("tr"));
        hiddenTargetBedSelect.css("display", "none").attr("name", null).addClass("hiddenTargetBedSelect");

        var hiddenHotspotBedSelect = hotspotBedSelect.clone().appendTo($(this).closest("tr"));
        hiddenHotspotBedSelect.css("display", "none").attr("name", null).addClass("hiddenHotspotBedSelect");

    }).change(function()  {
        //When the reference changes, copy the children into this select box.
        var reference = $(this).find("option:selected").data('reference');
        //console.log("page_plan_sample_table.js - reference change...reference=", reference, "; tr=", $(this).closest("tr"));
        var targetBedSelect = $(this).closest("tr").find("select[name=targetRegionBedFile]");
        var hotspotBedSelect = $(this).closest("tr").find("select[name=hotSpotRegionBedFile]");
        var hiddenTargetBedSelect = $(this).closest("tr").find(".hiddenTargetBedSelect").first();
        var hiddenHotspotBedSelect = $(this).closest("tr").find(".hiddenHotspotBedSelect").first();
        //changing the mixedTypeRNA ref selection should not clear the DNA sample's BED file selection
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
        targetBedSelect.children().remove();
        targetBedSelect.append(hiddenTargetBedSelect.children().clone().map(function(){
            if($(this).hasClass(reference) || $(this).attr("value") == ""){
                return this;
            } else {
                return null;
            }
        }));
        hotspotBedSelect.children().remove();
        hotspotBedSelect.append(hiddenHotspotBedSelect.children().clone().map(function(){
            if($(this).hasClass(reference) || $(this).attr("value") == ""){
                return this;
            } else {
                return null;
            }
        }));
    }).change();


    //Barcoded samples' reference selection lists can be programmatically updated and there can be many barcoded samples.
    //Instead of updating the samples table at change event, update when user click on the selection 
    $('select[name=reference]').bind('click', function (e) {
        var currentRow = $(this).closest('tr');
            
        //When the reference changes, copy the children into this select box.
        var reference = $(this).find("option:selected").data('reference');
        //console.log("page_plan_sample_table.js - reference BIND click...currentRow.index=", currentRow.index(), " GOING to call updateSamplesTable");        
        updateSamplesTable();
    });    
    

    $('select[name=targetRegionBedFile]').bind('click', function (e) {
        var currentRow = $(this).closest('tr');
            
        //When the reference changes, copy the children into this select box.
        var targetRegionBedFile = $(this).find("option:selected").data('targetRegionBedFile');
        //console.log("page_plan_sample_table.js - targetRegionBedFile BIND click...currentRow.index=", currentRow.index(), "; targetRegionBedFile=", targetRegionBedFile);
        updateSamplesTable();
    });
            

    $('select[name=hotSpotRegionBedFile]').bind('click', function (e) {
        var currentRow = $(this).closest('tr');
            
        //When the reference changes, copy the children into this select box.
        var hotSpotRegionBedFile = $(this).find("option:selected").data('hotSpotRegionBedFile');
        //console.log("page_plan_sample_table.js - hotSpotRegionBedFile BIND click...currentRow.index=", currentRow.index(), "; hotSpotRegionBedFile=", hotSpotRegionBedFile);
        updateSamplesTable();
    });

    $('#saveTable').click(function(e){
        e.preventDefault();
        updateSamplesTable();
        var form = $('<form id="saveTable_form" method="POST" action="' + $(this).attr("href") + '">');
        $("#samplesTable").clone().appendTo(form);
        $(document.body).append(form); // need for IE9
        form.submit();
        form.remove();
    });
    
    $('#modal_load_samples_table .csv_load.btn').click(function(e){
        // load selected csv file and update samples table
        e.preventDefault();
        $('#modal_error').empty().hide();

        var filename = $('#modal_load_samples_table :file').val();
        if (!filename){
            $('#modal_error').empty().append('Please select a CSV file').show();
            return false;
        }

        var url = $(this).attr("href");
        var form = $('#step_form').attr('action', url);
        form.append(
            $('<input>', { name: 'irSelected', val: USERINPUT.is_ir_connected ? "1":"0" })
        );
        form.ajaxSubmit({
            dataType : "json",
            url: url,
            async: false,
            beforeSend: function() {
                console.log('submitting', form)
            },
            success: function(data) {
                console.log('success', data)
                var table = data.samplesTable;
                $('#numRows').val(table.length).change();
                if ($('#isSameRefInfoPerSample').is(':checked') && !data.same_ref_and_bedfiles){
                    $('#isSameRefInfoPerSample').prop('checked', false);
                    handle_isSampleRefInfoPerBarcodeSample(false, false);
                }
                $('#chipsets tbody tr').each(function (i) {
                    var $row = $(this);
                    $.each(data.ordered_keys, function(){
                        var name = this;
                        var elem = $row.find("[name="+name+"]");
                        if (name=="barcodeId") {
                            $row.find('[name=barcode] option[data-id_str='+table[i][name]+']').prop('selected', true);
                        } else {
                            elem.val(table[i][name]);
                        }
                        // populate onchange select options
                        if (name=="reference" || name=="irWorkflow") elem.change();
                    })
                });
                updateSamplesTable();

                $('#modal_load_samples_table').modal('hide');
                $('.modal-backdrop').remove();
            },
            error: function(data){
                $('#modal_error').empty().append(data.responseText).show();
            }
        });
        return false;
    });
});
