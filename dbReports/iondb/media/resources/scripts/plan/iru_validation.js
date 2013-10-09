function call_validation_api(postData, accountId) {
	var url = "/rundb/api/v1/plugin/IonReporterUploader/extend/wValidateUserInput/";
	var result = {};
	$.ajax({
  			type: 'POST',
  			url: url + "?format=json&id="+accountId,
  			data: JSON.stringify(postData["userInput"]),
  			success: function(data){
				result = data;
  			},
  			dataType: "json",
  			async:false
	});
	
	return result;
}

function get_user_input_info_from_ui(accountId, accountName) {
	var userInputInfo = {"userInputInfo" : [], "accountId" : accountId, "accountName" : accountName};
	var is_barcoded = $("input[name=sampleName1]").length == 0;
	
	var $sampleNames;
	if (is_barcoded) {
		$sampleNames = $("input").filter(function(){return this.name.match(/barcodeSampleName\d/)});
	} else {
		$sampleNames = $("input").filter(function(){return this.name.match(/sampleName\d/)});
	}

	for (i =0; i < $sampleNames.length; i++) {
		self = $($sampleNames[i]);

		if (self.val().length > 0) {
			var $tr = self.parent().parent();
			var dict = {};
			dict["sample"] = self.val();
			dict["row"] = (i+1).toString();
			dict["barcodeId"] = is_barcoded ? $tr.find("td:eq(0)").text() : "";
			dict["Workflow"] = $tr.find(".irWorkflow").val();
			dict["Gender"] = $tr.find(".irGender").val();
			dict["Relation"] = $tr.find(".irRelation").val();
			dict["RelationRole"] = $tr.find(".irRelationRole").val();
			dict["setid"] = $tr.find(".irSetID").val();
			userInputInfo["userInputInfo"].push(dict);
		}
	}

	return userInputInfo;
}

function create_user_input_info(accountId, accountName, userInputInfoDict) {
	if (typeof userInputInfoDict == 'undefined') {
		return get_user_input_info_from_ui();
	} else {
		var userInputInfo = {"userInputInfo" : [], "accountId" : accountId, "accountName" : accountName};
		var row = 1;
		$.each(userInputInfoDict, function(k,v){
			if (typeof v['sampleName'] != 'undefined') {
				var dict = {};
				dict["barcodeId"] = typeof v["barcodeId"] != 'undefined' ? v["barcodeId"] : "";
				dict["row"] = row.toString();
				dict["sample"] = v["sampleName"];
				dict["Workflow"] = v["workflow"];
				dict["Relation"] = v["relation"];
				dict["RelationRole"] = v["relationRole"];
				dict["Gender"] = v["gender"];
				dict["setid"] = v["irSetID"];
				userInputInfo["userInputInfo"].push(dict);

				row++;
			}
			
		});
		return userInputInfo;
	}
}

function show_apprise($form, message) {
	
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
				else {return false;}
			}
	);

	return false;
}

function validate_user_input_from_iru($form, accountId, accountName, userInputInfoDict) {
	
	flag = false;
	var $div = $("#errors");
	$div.html('');

	var url = "/rundb/api/v1/plugin/IonReporterUploader/extend/validateUserInput/";
	var data = {"userInput" : create_user_input_info(accountId, accountName, userInputInfoDict)};
	if (data["userInput"]["userInputInfo"].length == 0) {
		$div.html("You must enter at least one sample name");
		return false;
	}
	var results = call_validation_api(data, accountId);
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
	
	if (error_messages.length == 0 && warning_messages.length == 0) {return true;}
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
	return show_apprise($form, results["advices"]["onTooManyErrors"]);
	

}