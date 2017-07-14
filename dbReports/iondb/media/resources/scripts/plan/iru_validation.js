var USERINPUT = USERINPUT || {};


function show_apprise($form, message) {
	/*we no longer allow user to igore IRU validation errors
	apprise(message, {
		    'confirm': false, 		// Ok and Cancel buttons
		    'verify': true, 	// Yes and No buttons
		    'input': false, 		// Text input (can be true or string for default text)
		    'animate': true, 	// Groovy animation (can true or number, default is 400)
		    'textOk': 'Ok', 	// Ok button default text
		    'textCancel': 'Cancel', // Cancel button default text
		    'textYes': 'Ignore Errors', 	// Yes button default text
		    'textNo': 'Fix Errors', 	// No button default text
		    'position': 'center'// position center (y-axis) any other option will default to 100 top
		}, function (r) {
				if (r) {$form.unbind("submit");$form.submit();}
			}
	);
	*/
	apprise(message);
}

function show_errors($form, error_messages){
    var $div = $("#error");
    $div.removeClass('alert alert-error').empty();

    if (error_messages.length > 0){
        var err_str = error_messages.length > 1 ? + error_messages.length + " Errors" : "Error";
        err_str += " found by Ion Reporter validation, please see highlighted boxes in the table below";
        $div.addClass('alert alert-error');
        $div.append("<h4>"+ err_str +"<a class='pull-right' href='#'><i class='showall icon-minus'></i></a></h4>");
        $div.append("<div id='all_iru_errors'><ul><li>" + error_messages.join('</li><li>') + '</li></ul></div>');
        $div.find(".showall").on('click', function () {
            $(this).toggleClass('icon-minus icon-plus');
            $("#all_iru_errors").toggle();
        });
        if (error_messages.length > 3) $div.find(".showall").click();
    }

    $('#grid').data('kendoGrid').refresh(); // refresh grid to display validation errors
    
    $.unblockUI();
    $("html, body").animate({ scrollTop: 0 }, "slow");
}

function call_validation_api($form, accountId, userInputInfoDict) {
    console.log("iru_validation.call_validation_api() userInputInfoDict=", userInputInfoDict);

    var ts_fieldnames = {
        "sample": "sampleName",
        "Workflow": "irWorkflow",
        "Relation": "irRelationRole",
        "Gender": "irGender",
        "CancerType": "ircancerType",
        "CellularityPct": "ircellularityPct",
        "setid": "irSetID",
        "NucleotideType": "nucleotideType",
        "ControlType": "controlType",
        "Reference" : "reference",
        "TargetRegionBedFile" : "targetRegionBedFile",
        "HotSpotRegionBedFile" : "hotSpotRegionBedFile"
    }

    var url = "/rundb/api/v1/plugin/IonReporterUploader/extend/wValidateUserInput/";
    var message = "Internal error at IonReporterUploader during sample/Ion Reporter configuration validation";

    $.ajax({
        type: 'POST',
        url: url + "?format=json&id="+accountId,
        data: JSON.stringify(userInputInfoDict),
        dataType: "json",
        async: true,
        success: function(data){
            var results = data;
            var error_messages = [];
            var warning_messages = [];

            if (results) {
            $.each(results["validationResults"], function(k, v){
                if (typeof v["errors"] != 'undefined' && v["errors"].length > 0) {
                    $.each(v["errors"], function(i){
                        error_messages.push("IonReporter:ERROR:Row" + v["row"] + ":" + v["errors"][i]);
                    });
                }
                if (typeof v["warnings"] != 'undefined' && v["warnings"].length > 0) {
                    $.each(v["warnings"], function(i){
                        error_messages.push("IonReporter:WARNING:Row" + v["row"] + ":" + v["warnings"][i]);
                        //warning_messages.push("IonReporter:WARNING:Row" + v["row"] + ":" + v["warnings"][i]);
                    });
                }
                // currently same row error is displayed for all highlightableFields
                // TODO: IRU needs to return error per field
                for (name in ts_fieldnames){
                    var rowError = v["errors"] ? v["errors"].join('<br>') : "";
                    var err = "";
                    if (typeof v["highlightableFields"] != 'undefined' && v["highlightableFields"].length > 0) {
                        if (v["highlightableFields"].indexOf(name) > -1 )
                            err = rowError;
                    }
                    updateSamplesTableValidationErrors(parseInt(v["row"])-1, ts_fieldnames[name], "", err);
                }
            });
                if (typeof data["status"] != 'undefined' && typeof data["error"] != 'undefined') {
                    console.log("iru_validation - data[status]=", data["status"]);
                    console.log("ion_validation - data[error]=", data["error"]);
                    if (data["status"] === "false") {
                        message += ".   ";
                        message += data["error"];
                        error_messages.push(message);
                    }
                }
            } else {
                error_messages.push(message);
            }
            
            if (error_messages.length > 0){
                if (results && results["advices"]) {
                    show_apprise($form, results["advices"]["onTooManyErrors"]);
                }
                show_errors($form, error_messages);
            } else {
                $form.unbind("submit");
                $form.submit();
                return;
            }
        },
        error: function(data){
            show_errors($form, [message, data.status +' '+ data.statusText]);
        }
    });
}

function get_user_input_info_from_ui() {
    var userInputInfo = [];
    var samplesTable = JSON.parse($('#samplesTable').val());

    for (i = 0; i < samplesTable.length; i++) {
        var row = samplesTable[i]
        if ( row["sampleName"].length > 0){
            var dict = {};
            dict["sample"] = row["sampleName"];
            dict["row"] = (i+1).toString();
            dict["Workflow"] = row["irWorkflow"];
            dict["tag_isFactoryProvidedWorkflow"] = dict["tag_isFactoryProvidedWorkflow"];
            dict["Gender"] = row["irGender"];
            dict["nucleotideType"] = row["nucleotideType"];
            dict["Relation"] = row["irRelationshipType"];
            dict["RelationRole"] = row["irRelationRole"];
            dict["cancerType"] = row["ircancerType"];
            dict["cellularityPct"] = row["ircellularityPct"];
            dict["biopsyDays"] = row["irbiopsyDays"];
            dict["coupleID"] = row["ircoupleID"];
            dict["embryoID"] = row["irembryoID"];
            dict["setid"] = row["irSetID"];
            dict["controlType"] = row["controlType"];
            dict["reference"] = row["reference"];
            dict["targetRegionBedFile"] = row["targetRegionBedFile"];
            dict["hotSpotRegionBedFile"] = row["hotSpotRegionBedFile"];

            userInputInfo.push(dict);
        }
    }
    return userInputInfo;
}

$(document).ready(function(){

    $("form").submit(function(e){
        updateSamplesTable();

        if ($("input[name=irDown]").val() == "1" || (USERINPUT.account_id == "0") || (USERINPUT.account_id == "-1") )
            return true;

        if (!USERINPUT.validate) {
            return true;
        } else {
            USERINPUT.validate = USERINPUT.is_by_sample;
        }

        $.blockUI();

        var $form = $(this);
        var error_messages = [];
        var rows_dict = get_user_input_info_from_ui();
        
        if (rows_dict.length == 0) {
            error_messages.push("You must enter at least one sample name");
        } else {
            $.each(rows_dict, function(i, dict){
                var irworkflow = dict["Workflow"];
                if ( (dict["RelationRole"].length == 0) && irworkflow && irworkflow != 'Upload Only'){
                    error_messages.push("Relation on row " + (i+1) + " cannot be blank");
                    updateSamplesTableValidationErrors(i, "irRelationRole", "", "Relation cannot be blank");
                }
            });
        }

        if (error_messages.length == 0){
            var userInput = {
                "accountId" : USERINPUT.account_id,
                "accountName" : USERINPUT.account_name,
                "isVariantCallerSelected":USERINPUT.is_variantCaller_enabled,
                "isVariantCallerConfigured": USERINPUT.is_variantCaller_configured,
                "userInputInfo" : rows_dict
            };
            call_validation_api($form, USERINPUT.account_id, userInput);
        } else {
            show_errors($form, error_messages);
        }

        return false;
    });
});
