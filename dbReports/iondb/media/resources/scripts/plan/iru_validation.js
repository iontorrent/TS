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

function call_validation_api($form, $div, postData, accountId) {
	var myData = JSON.stringify(postData["userInput"]);
	console.log("iru_validation.call_validation_api() myData=", myData);
	
	var url = "/rundb/api/v1/plugin/IonReporterUploader/extend/wValidateUserInput/";
	$.ajax({
  			type: 'POST',
  			url: url + "?format=json&id="+accountId,
  			data: JSON.stringify(postData["userInput"]),
  			success: function(data){
				var results = data;
				var error_messages = [];
				var warning_messages = [];
				$.each(results["validationResults"], function(k, v){
					if (typeof v["errors"] != 'undefined' && v["errors"].length > 0) {
						$.each(v["errors"], function(i){
							error_messages.push("IonReporter:ERROR:Row" + v["row"] + ":" + v["errors"][i]);	
						});
					}
					if (typeof v["warnings"] != 'undefined' && v["warnings"].length > 0) {
						$.each(v["warnings"], function(i){
							warning_messages.push("IonReporter:WARNING:Row" + v["row"] + ":" + v["warnings"][i]);	
						});
					}
				});
	
				if (error_messages.length == 0 && warning_messages.length == 0) {
					$form.unbind("submit");
                	$form.submit();
                	return;
				}
				else if (warning_messages.length > 0 || error_messages.length > 0) {

					$.each(error_messages, function(i){
						var message = error_messages[i];
						var $div_error = $("<div></div>", {"style" : "color:red;font-weight:bold;margin-bottom:20px;"});
						$div_error.html(message+"<br/>");
						$div.append($div_error);
					});
					$.each(warning_messages, function(i){
						var message = warning_messages[i];
						var $div_error = $("<div></div>", {"style" : "color:red;font-weight:bold;"});
						$div_error.html(message+"<br/>");
						$div.append($div_error);
					});
				}
				$.unblockUI();
				show_apprise($form, results["advices"]["onTooManyErrors"]);
				$("html, body").animate({ scrollTop: 0 }, "slow");
  			},
  			dataType: "json",
  			async: true
	});
}

function validate_user_input_from_iru($form, accountId, accountName, userInputInfoDict) {

	var $div = $("#error");
	$div.html('');

	var url = "/rundb/api/v1/plugin/IonReporterUploader/extend/validateUserInput/";
	var data = {"userInput" : userInputInfoDict};

	console.log("iru_validation.validate_user_input_from_iru() userInputInfoDict=", userInputInfoDict);
	
	if (data["userInput"]["userInputInfo"].length == 0) {
		$div.html("You must enter at least one sample name");
		$.unblockUI();
		$("html, body").animate({ scrollTop: 0 }, "slow");
		return;
	} else {
		$div.html('');
	}
	call_validation_api($form, $div, data, accountId);
}

function get_user_input_info_from_ui(accountId, accountName) {
	var userInputInfo = {"userInputInfo" : [], "accountId" : accountId, "accountName" : accountName};
	var $sampleNames = $(".irSampleName");

	for (i = 0; i < $sampleNames.length; i++) {
		var self = $($sampleNames[i]);

		if (self.val().length > 0) {
			var $tr = self.parent().parent();
			
//			$tr.each(function(index) {
//				var currentCell = this;
//				console.log("...currentCell=", currentCell);
//			});
					
			var dict = {};
			dict["sample"] = self.val();
			dict["row"] = (i+1).toString();
			dict["Workflow"] = $tr.find(".irWorkflow").val();
			dict["Gender"] = $tr.find(".irGender").val();			
			dict["nucleotideType"] = $tr.find(".nucleotideType").val();
			dict["Relation"] = $tr.find(".irRelationshipType").val();
			dict["RelationRole"] = $tr.find(".irRelationRole").val();
			dict["cancerType"] = $tr.find(".ircancerType").val();
			dict["cellularityPct"] =$tr.find(".ircellularityPct").val();
			dict["setid"] = $tr.find(".irSetID").val();
			userInputInfo["userInputInfo"].push(dict);
		}
	}

	return userInputInfo;
}

$(document).ready(function(){

	$("form").submit(function(e){
		var $div = $("#error");
		$div.html('');
        
        if ($("input[name=irDown]").val() == "1")
            return true;

        if (USERINPUT.is_by_template) {
        	if ($(this).attr('action') != USERINPUT.by_template_url) {return true;}
        } else {
        	if ($(this).attr('action') == USERINPUT.by_sample_url) {return true;}
        }
        
        $.blockUI();

        if ((USERINPUT.account_id != "0") && (USERINPUT.account_id != "-1")) {
        	var $rows = $("#chipsets tbody tr");;
        	var badRelation = false;   
        	var counter = 1; 
        	$.each($rows, function(i){
            	var $tr = $(this);            	
            	if ($tr.find(".irSampleName").val().length > 0 && ($tr.find(".irRelationRole").val() == null)) {            		
                	badRelation = true;
                	return false;            		
            	}
            	if ($tr.find(".irSampleName").val().length > 0 && $tr.find(".irRelationRole").val().length == 0 && $tr.find(".irWorkflow").val() != 'Upload Only') {
                	badRelation = true;
                	return false;
            	}
            	counter++;
        	});
        
        	if (badRelation){$div.text("Relation on row " + counter + " cannot be blank");$.unblockUI();return false;}
        	else {$div.text("");}

        	validate_user_input_from_iru($(this), USERINPUT.account_id, USERINPUT.account_name, get_user_input_info_from_ui(USERINPUT.account_id, USERINPUT.account_name));
        	return false;
   		} else {
   			return true;
   		}
        return false;
    	
     });
});
