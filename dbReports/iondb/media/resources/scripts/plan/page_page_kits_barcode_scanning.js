//This script is used to detect the 2D Barcode scanning of Kits and
// selects the corresponding kits automatically in the kits Chevron


function update_kit_selection(kitInfoObject){
    isCompatible = false;
    kitType = kitInfoObject["kitType"];
    value = kitInfoObject["name"];
    kitDisplayedName = kitInfoObject["description"];

    fieldObj = {
        "SequencingKit" : "sequenceKit",
        "TemplatingKit" : "templateKit",
        "LibraryKit"    : "libraryKitType",
        "SamplePrepKit" : "samplePreparationKit",
        "ControlSequenceKit" : "controlsequence",
        "IonChefPrepKit" : "templateKit"
    }

    if (kitType == "IonChefPrepKit") {
        //change the templatingKit type selection to ionChef
        $('input[value="IonChef"]').prop("checked",true).trigger('change');
    }
    else if (kitType == "TemplatingKit") {
        $('input[value="OneTouch"]').prop("checked",true).trigger('change');
    }

    $('#' + fieldObj[kitType] + ' option').each(function(){
        if (this.value == value) {
            isCompatible = true;
            $('#'+ fieldObj[kitType]).val(value).trigger('change');
            return false;
        }
    });

    //send the error message if kit is not compatible.
    if (!isCompatible){
        errMsg = "<div class='span10 alertScannerError alert alert-danger'>Scanned barcode not compatible with the selected chip/instrument type.";
        errMsg = errMsg + "<p>" + kitType + ":" + kitDisplayedName + "</p></div>";
        $("#scannerValidationError").html(errMsg);
        $("#scannerValidationError").show().delay(12000).fadeOut();
    }
}

$(document).scannerDetection({
    timeBeforeScanTest: 200, // wait for the next character for upto 200ms
    avgTimeByChar: 100, // it's not a barcode if a character takes longer than 100ms
    onComplete: function (kitBarcode_scanner, qty) {
        var parsedBarcode = "";
        // sometimes, the scanner is detecting some junk chars during scanning, make sure no invalid chars
         kitBarcode_scanner = kitBarcode_scanner.replace(/[^A-Za-z 0-9 \.,\?""!@#\$%\^&\*\(\)-_=\+;:<>\/\\\|\}\{\[\]`~]*/g, '');
        if ((kitBarcode_scanner.substring(0, 2) == "91") && (kitBarcode_scanner.indexOf("]") > 0)) {
            console.log("The scanned barcode does comply with the expected format : ", kitBarcode_scanner);
            parsedBarcode = kitBarcode_scanner.substring(2, kitBarcode_scanner.indexOf("]"));
        }
        else if ((kitBarcode_scanner.substring(0, 2) == "91") && (kitBarcode_scanner.indexOf("100") > 0)) {
            //For some reason, "]" is not getting detected in the mac browser
            console.log("The scanned barcode does comply with the expected format : ", kitBarcode_scanner);
            parsedBarcode = kitBarcode_scanner.substring(2, kitBarcode_scanner.indexOf("100"));
        }
        else {
            errMsg = "<div class='span10 alertScannerError alert alert-danger'>The scanned barcode does not comply with the expected format. Please check.";
            console.log("The scanned barcode does not comply with the expected format : ", kitBarcode_scanner);
            $("#scannerValidationError").html(errMsg);
            $("#scannerValidationError").show().delay(12000).fadeOut();
        }
        //Barcode info detected, proceed to find which kit is scanned
        if (parsedBarcode) {
            var url = "/rundb/api/v1/kitpart/" + '?&barcode__icontains=' + parsedBarcode + '&format=json';
            $.ajax({
                type: "GET",
                url: url,
                contentType: "application/json; charset=utf-8",
                dataType: "json",
                success: function (data) {
                    if (data.objects.length > 0) {
                        kitPartObj = data.objects[0];
                        kitInfoURL = kitPartObj["kit"];
                        $.ajax({
                            type: "GET",
                            url: kitInfoURL,
                            timeout: 100,
                            success: function (kitInfoObject) {
                                console.log("Barcode Scanned : ", kitInfoObject["description"]);
                                update_kit_selection(kitInfoObject);
                            }
                        });
                    }
                },
                error: function(xhr, textStatus, error){
                    console.log(xhr.statusText);
                    console.log(textStatus);
                    console.log(error);
                }
            });
        }
    }
});