
    
    /**
    Show/hide barcoded sample table ref info columns based on if isSameRefInfoPerSample is checked
    */
    /*
    handle_isSampleRefInfoPerBarcodeSample = function (isChecked, isInit) {
        //(1) if isChecked is true and plan is barcoded, 
        //(1.1) cascade default values to each reference and BED table cell
        //(1.2) hide the table columns
        //(1.3) disable the table columns
        //
        //(2) if isChecked is false and plan is barcoded, 
        //(2.1) show the table columns
        //(2.2) enable the table columns
                                
        var isPlan = $('input[id=isPlan]').val();
        var selectedBarcodeKit = $('#barcodeSet').val();
        var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
   
        if (isPlan == "True") {
            var selectedReference = $('select[name=default_reference]').val();
            var selectedTargetBedFile = $('select[name=default_targetBedFile]').val();
            var selectedHotSpotBedFile = $('select[name=default_hotSpotBedFile]').val();
            
            var mixedTypeRNA_selectedReference = $('select[name=mixedTypeRNA_reference]').val();
            var mixedTypeRNA_selectedTargetBedFile = $('select[name=mixedTypeRNA_targetBedFile]').val();

			//if we're initializing the page for an edit operation, do not alter the original sample selections
            var isCreate = $('input[id=isCreate]').val();
              
            //beware of hotSpot column which may not be shown for an application
            var isReferenceSupported = $('input[id=isReferenceSupported]').val();
            var isTargetRegionBEDFileSupported = $('input[id=isTargetRegionBEDFileSupported]').val();
            var isHotspotRegionBEDFileSupported = $('input[id=isHotspotRegionBEDFileSupported]').val();
            var isTargetRegionBEDFileBySampleSupported = $('input[id=isTargetRegionBEDFileBySampleSupported]').val();
            var isHotSpotBEDFileBySampleSupported = $('input[id=isHotSpotBEDFileBySampleSupported]').val();
 
 			//beware of switching chevrons after the barcodedSamples' selections have been set
                
            if (isChecked) {
                toggleSampleReferenceColumnEnablements(isChecked, isReferenceSupported, isTargetRegionBEDFileSupported, isHotspotRegionBEDFileSupported, isTargetRegionBEDFileBySampleSupported, isHotSpotBEDFileBySampleSupported);
                    
                //when editing or copying a plan, if it is not a barcoded plan, we need to populate sample ref info with defaults
                if (isCreate == "True" || isInit == false || (isCreate == "False" && selectedBarcodeKit == "") ) {
                    updateSampleReferenceColumnsWithDefaults(selectedReference, selectedTargetBedFile, selectedHotSpotBedFile, mixedTypeRNA_selectedReference, mixedTypeRNA_selectedTargetBedFile);
                }

                //don't overload this UI control to be multi-purpose; only hide the sample ref columns at init time if checkbox is checked
                if (isInit == true) {
                    $(".hideable_referenceBySample").each(function (index, value) {
                        //console.log("going to HIDE...index=", index, "; value=", value);
                        
                        $(this).hide();
                    });
                }
            }
            else {                
                showSampleReferenceColumns(isReferenceSupported, isTargetRegionBEDFileSupported, isHotspotRegionBEDFileSupported, isTargetRegionBEDFileBySampleSupported, isHotSpotBEDFileBySampleSupported);

                toggleSampleReferenceColumnEnablements(isChecked, isReferenceSupported, isTargetRegionBEDFileSupported, isHotspotRegionBEDFileSupported, isTargetRegionBEDFileBySampleSupported, isHotSpotBEDFileBySampleSupported);

            }
        }
        else {
            //we should not be here since the checkbox should not be shown
            console.log("NO-OP for handle_isSampleRefInfoPerBarcodeSample!!");
        }
          
    }
    */
