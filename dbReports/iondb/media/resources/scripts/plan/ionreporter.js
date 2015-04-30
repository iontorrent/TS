var IONREPORTER = IONREPORTER || {};

//this is the REST API url for looking up all workflows for a given IR Account
IONREPORTER.workflow_url = "/rundb/api/v1/plugin/IonReporterUploader/extend/workflows/";
//IONREPORTER.workflow_url1 = "/rundb/api/v1/plugin/IonReporterUploader/extend/workflowsWithoutOncomine/";
//IONREPORTER.workflow_url2 = "/rundb/api/v1/plugin/IonReporterUploader/extend/workflowsWithOncomine/";
//this the REST API url for retrieving all IR Accounts and their configuration
IONREPORTER.ion_accounts_url = "/rundb/api/v1/plugin/IonReporterUploader/extend/configs/";

IONREPORTER.ir_rel_type_to_sample_group_map = {};
IONREPORTER.sample_group_to_pk_map = {};
IONREPORTER.workflow_to_sample_group_map = {};
IONREPORTER.ir_rel_type_pk_map = {};
IONREPORTER.sample_group_to_workflow_map = {};
IONREPORTER.default_sample_grouping = [];
IONREPORTER.workflow_to_application_type_map = {};
IONREPORTER.$none_input;

/**
    This function creates the None Radio Option for the IR Accounts.
    If the None option is pressed, then the accountId is set to 0
    and the account name is set to None and the workflow is cleared
*/
function create_none_ir_account() {
    var $div = $("#ir_accounts");
    var $none_lbl = $("<label></label>");;
    $none_lbl.addClass('radio');
    $none_lbl.text("None");

    IONREPORTER.$none_input = $("<input type='radio'/>");

    IONREPORTER.$none_input.attr({'name' : 'irOptions', 'value' : '0'});
    IONREPORTER.$none_input.on('click', function(){

        $("#error").text("");
        $('input[name="irAccountId"]').val('0');
        $('input[name="irAccountName"]').val('None');
        $("#selectedIR").text('None');
        $("#selectedWorkflow").text('');
        $("#selectedGroup").text('');
        $("input[name=irVersion]").val('0');
        $("#workflows").hide();
        var $div = $(".sampleGroupOptionsContent");
        $div.html('');
        $.each(IONREPORTER.default_sample_grouping, function(i){
            sample_group = IONREPORTER.default_sample_grouping[i];
            var $label = $("<label></label>", {'class' : 'radio', 'width' : '150px'});

            $label.text(sample_group);
            var $input = $("<input/>", {'type' : 'radio', 'name' : 'sampleGrouping', 'value' : IONREPORTER.sample_group_to_pk_map[sample_group]});
            if ("{{helper.getStepDict.Ionreporter.getCurrentSavedFieldDict.sampleGrouping}}" == IONREPORTER.sample_group_to_pk_map[sample_group]) {
                $input.attr('checked', true);
            }
            $input.on('click', function(){
                $("#selectedGroup").text($input.parent().text().trim());
            });
            $label.append($input);
            $div.append($label);

        });

    });

    $none_lbl.append(IONREPORTER.$none_input);
    $div.append($none_lbl);
    return $div;

}

/**
 * Filter IR workflows based on runType and application group
 */
function get_workflow_url() {
    var applicationGroupName = $('input[name=applicationGroupName]').val();
    var runType_name = $('input[name=runType_name]').val();
    var runType_nucleotideType = $('input[name=runType_nucleotideType]').val();
    var planCategories = $('input[name="planCategories"]').val();
    console.log("ionreporter.get_workflow_url() applicationGroupName=", applicationGroupName, "; runType_name=", runType_name, "; runType_nucleotideType=", runType_nucleotideType, "; planCategories=", planCategories);

    var myURL = IONREPORTER.workflow_url;

	myURL += "?format=json";
	var isFilterSet = false;

	if (runType_nucleotideType.toLowerCase() == "dna" ||  (runType_nucleotideType == "" && applicationGroupName.toLowerCase() == "dna")) {
		myURL += "&filterKey=DNA_RNA_Workflow&filterValue=";
		myURL += "DNA";

		isFilterSet = true;
	}
	else if (runType_nucleotideType.toLowerCase() == "rna" || (runType_nucleotideType == "" && applicationGroupName.toLowerCase() == "rna")) {
		myURL += "&filterKey=DNA_RNA_Workflow&filterValue=";
		myURL += "RNA";

		isFilterSet = true;
	}

    if (applicationGroupName == "DNA + RNA") {
   	/*for mixed single & paired type support
    	if (runType_nucleotideType.toLowerCase() == "dna_rna") {
    		myURL += "&filterKey=DNA_RNA_Workflow&filterValue=";
    		myURL += "DNA_RNA";

    		isFilterSet = true;
    	}
    	*/

        //myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";
        
        if (planCategories.toLowerCase().indexOf("oncomine") != -1) {            
//            if (!isFilterSet) {
//                myURL += "&filterKey=Onconet_Workflow&filterValue=false";  
//            }
            myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";     
        }
        else if (planCategories.toLowerCase().indexOf("onconet") != -1) {            
            if (!isFilterSet) {
                myURL += "&filterKey=Onconet_Workflow&filterValue=true";
            }
//            myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=false";
        }
    }
    else {
    	if (runType_name.toLowerCase() != "amps") {    		
    		if (!isFilterSet) {
        		myURL += "&filterKey=Onconet_Workflow&filterValue=false";
    		}
    	    myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=false";
    	}
    	else {
            if (planCategories.toLowerCase().indexOf("oncomine") != -1) {
                myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";
            }
            else if (planCategories.toLowerCase().indexOf("onconet") != -1) {
                myURL += "&andFilterKey2=Onconet_Workflow&andFilterValue2=true";
            }    	   
    	}
    }

    return myURL;
}


/**
    This function retrieves the workflows, relationship type, and application types for a given IR Account
    by its given ID.  It receives the account id and creates a workflow drop down list
    and creates radio buttons for the Sample Grouping (Relationship Types) based on a TSS to IR Mapping of names
*/
function get_workflow_and_sample_groupings(id, fullName) {
    var myURL = get_workflow_url();

    var all_workflows = [];
    $('input[name="irAccountId"]').val(id);
    $('input[name="irAccountName"]').val(fullName);
    $("#selectedIR").text(fullName);
    $("#error").text("");

    //First we call the API to retrieve all workflows
    $.when($.ajax({
                url : myURL+"&id="+id,
                timeout: 6000, //in milliseconds
                error: function(jqXHR, textStatus, errorThrown){
                    if(textStatus==="timeout") {
                        //The API Server timedout
                        $("#loading").hide();
                        $("#error").text('The IR Server connection has timed out');
                        return;
                    }
                }
            }).then(function(data, textStatus, jqXHR) {

                IONREPORTER.sample_group_to_workflow_map = {};

                var workflows = data["userWorkflows"];
                var all_relationship_types = [];

                //clear the workflow select drop down list
                var $select = $("select[name=irworkflow]");
                $select.empty();
                $select.append($("<option>Upload Only</option>"));

                if (typeof workflows != 'undefined') {
                    //loop through the workflows
                    $.each(workflows, function(i){
                        //add an empty option
                        var $opt = $("<option></option>");
                        var workflowName = workflows[i]["Workflow"];
                        //add the application type that is coupled with this workflow
                        //to an object
                        IONREPORTER.workflow_to_application_type_map[workflowName] = workflows[i]["ApplicationType"];
                        //add the workflow to a list of all workflows
                        all_workflows.push(workflowName);
                        //add the relationship type (i.e. the Sample Grouping) that is coupled with this workflow
                        //to an object
                        IONREPORTER.workflow_to_sample_group_map[workflowName] = workflows[i]["RelationshipType"];
                        //create a SET of all workflows by checking if the workflow is already in the array
                        if ($.inArray(workflows[i]["RelationshipType"], all_relationship_types) == -1) {
                            all_relationship_types.push(workflows[i]["RelationshipType"]);
                        }
                        //create an MAP between the relationship type (Sample Group) and the workflows
                        //a relationship type can have more than one workflow
                        if (typeof IONREPORTER.sample_group_to_workflow_map[workflows[i]["RelationshipType"]] == 'undefined') {
                            IONREPORTER.sample_group_to_workflow_map[workflows[i]["RelationshipType"]] = [];
                        }
                        if ($.inArray(workflowName, IONREPORTER.sample_group_to_workflow_map[workflows[i]["RelationshipType"]]) == -1) {
                            IONREPORTER.sample_group_to_workflow_map[workflows[i]["RelationshipType"]].push(workflowName);
                        }
                        if (workflowName == IONREPORTER.existing_workflow){
                            $opt.attr('selected', 'selected');
                        }
                        $opt.attr('value', workflowName);
                        $opt.text(workflowName);
                        $select.append($opt);
                    });

                } else {
                    //The API call errored out and we have to inform the user thusly
                    $("#error").text("Error connecting to IR Server");
                    $("#loading").hide();
                    $('#selectedWorkflow').text('');
                    $("#sample_grouping").hide();
                    $("#workflows").hide();
                    $("input[name=applicationType]").val('');
                    return;
                }

                $("#workflows").show();
                $('#selectedWorkflow').text($('select[name=irworkflow]').val());

                found_workflow = false;
                $.each(all_workflows, function(i){
                    if (all_workflows[i] == IONREPORTER.existing_workflow) {
                        found_workflow = true;
                    }
                });

                //we have to check if the user has selected a new worflow that's different from the template
                //or has selected a new IR account
                if (!found_workflow && IONREPORTER.existing_workflow.length > 0 && IONREPORTER.prev_account_id != "-1" ) {
                    $("#error").text("The previous workflow for this plan is no longer available.  Please select another workflow");
                }

                //now we do the mapping between IR and TSS Sample Group Names
                var sample_groups = [];
                var $div = $(".sampleGroupOptionsContent");
                $div.html('');
                $.each(all_relationship_types, function(i){
                    var term = all_relationship_types[i];
                    if ($.inArray(term, sample_groups) == -1) {
                        sample_groups.push(term);
                    }
                });

                //now we loop through the samples groups and create radio buttons for each of them
                $.each(sample_groups, function(i){
                    sample_group = sample_groups[i];
                    var $label = $("<label></label>", {'class' : 'radio', 'width' : '150px'});
                    $div.append($label);
                    $label.text(sample_group);
                    if (typeof IONREPORTER.sample_group_to_pk_map[sample_group] == 'undefined') {
                        $("#error").text("The sample group " + sample_group + " is not supported in TSS");
                    } else {
                        var $input = $("<input/>", {'type' : 'radio', 'name' : 'sampleGrouping', 'value' : IONREPORTER.sample_group_to_pk_map[sample_group]});
                        $label.append($input);
                        //now we create the OnClick event for the Sample Group radio button
                        //once a sample group is selected, we filter out the workflow drop down list
                        $input.on('click', function(){
                            $("#selectedGroup").text($input.parent().text().trim());
                            var self = $(this);
                            var $select = $("select[name=irworkflow]");
                            $select.empty();
                            $select.append($("<option>Upload Only</option>"));
                            $.each(IONREPORTER.ir_rel_type_pk_map, function(term, pk){
                                if (pk == self.val()) {
                                    var workflows = IONREPORTER.sample_group_to_workflow_map[term];
                                    if (typeof workflows != 'undefined') {
                                        $.each(workflows, function(i){
                                            var workflow = workflows[i];
                                            var $opt = $("<option></option>", {'value' : workflow, 'text' : workflow});
                                            if (workflow == IONREPORTER.existing_workflow) {
                                                $opt.attr('selected', true);
                                            }
                                            $select.append($opt);
                                        });
                                    }
                                }
                            });
                            $select.change();
                        });
                    }
                    if (IONREPORTER.sample_group_to_pk_map[sample_group] == IONREPORTER.existing_sample_grouping) {
                        $input.attr('checked', true);
                        $input.click();
                    }
                });

                $("#sample_grouping").show();
                $("#loading").hide();
            })
        );
}

/**
    This function uses the REST API to retrieve all IR Accounts and their configurations
    and create Radio buttons for each account.  When an Account is selected, another REST API call is
    made to retrieve the workflows for that account
*/
function get_ir_accounts() {
    $div = create_none_ir_account();

    var jqhxhr = $.ajax({

        type : "get",
        url : IONREPORTER.ion_accounts_url+"?format=json",
        timout: 6000, //in milliseconds
        success : function(data){
            //The API call is successful
            if (data != "undefined") {
                $("#error").text("");
                var accounts = data;

                //boolean to check if the IR account was previously selected, either in the template
                //or part of a previous step in the plan run phase    
                var matched_prev = false;


                //loop through all accounts and create radio buttons for each
                $.each(accounts, function(i){
                    var account = accounts[i];
                    var id = account["id"];
                    var version = account["version"];
                    var checked = account["default"];
                    var name = account["name"];

                    if (IONREPORTER.prev_account_id == id){
                        matched_prev = true;
                    }

                    //we extract the version, i.e. 1.6 or 4.0
                    version = version.substring(2, version.length);
                    version = version.substring(0, 1) + "." + version.substring(1, version.length);

                    var fullName = name + " ";

                    //add orgname, firstname, lastname  and version if they are there.

                    var detailsStr = "";
                    if ('details' in account){
                        detailsStr += " (";

                        detailsStr += "Version: " + version;

                        if ('firstname' in account['details'] && 'lastname' in account['details']){
                            detailsStr += " | User: " + account['details']['firstname'] + " " + account['details']['lastname'] ;
                        } else {
                            detailsStr = detailsStr.substring(0, detailsStr.length-2);
                        }

                        if ('orgname' in account['details']){
                            detailsStr += " | Org: " + account['details']['orgname'];
                        }


                        detailsStr += ")";
                    }

                    if (detailsStr != " ()") {
                        fullName += detailsStr;
                    }

                    //create a radio button for the account 
                    var $lbl = $("<label></label>");
                    $lbl.addClass('radio');
                    $lbl.text(fullName);

                    var $input = $("<input type='radio'/>");
                    $input.attr({'name' : 'irOptions', 'value' : version});
                    //attach OnClick event for the account
                    //the onclick event fires the REST API call to retrieve
                    //workflows and relationship types (sample groupings)
                    $input.on('click', function(){

                        $("input[name=irAccountId]").val(id);
                        $('input[name="irAccountName"]').val(fullName);
                        $('input[name=irVersion]').val(version);
                        //fire API CALL
                        get_workflow_and_sample_groupings(id, fullName);
                        if (version == "1.6") {
                            $("#new_workflow").hide();
                        } else {
                            $("#new_workflow").show();
                        }
                    });
                    $lbl.append($input);

                    $div.append($lbl);

                    //now we check if this account was selected in the template or a previous step
                    if(IONREPORTER.prev_account_id == id) {
                        $input.attr('checked', true);
                        $("input[name=irAccountId]").val(id);
                        $('input[name="irAccountName"]').val(fullName);
                        $("#selectedIR").text(fullName);
                        $input.click();

                    } else {
                        //now we know this is the default account
                        if (checked && !matched_prev) {
                            $input.attr('checked', true);
                            $("input[name=irAccountId]").val(id);
                            $('input[name="irAccountName"]').val(fullName);
                            $("#selectedIR").text(fullName);
                            $('input[name=irVersion]').val(version);
                            if (IONREPORTER.prev_account_id == "-1") {
                                $input.click();
                            }

                        } else {
                            $input.attr('checked', false);
                        }

                    }

                });

                //this checks if the previously selected IR account
                //is accessible to this user
                if (!matched_prev && IONREPORTER.prev_account_id != "-1") {
                    IONREPORTER.$none_input.attr('checked', true);
                    IONREPORTER.$none_input.click();

                    $("input[name=irAccountId]").val('0');
                    $('input[name="irAccountName"]').val('None');
                    $("#selectedIR").text('None');
                    if (IONREPORTER.prev_account_id != "0" && IONREPORTER.prev_account_id != "-1"){$("#irAccountDoesNotExistError").show();}
                }

                }
            $("#loading").hide();
        } ,
        error: function() {
            IONREPORTER.$none_input.attr('checked', true);
            $("input[name=irAccountId]").val('0');
            $('input[name="irAccountName"]').val('None');
            $("#selectedIR").text('None');
            $("#loading").hide();
        }

    });

}

/**
 This function is used to open a pop-up window to allow the user to create a new workflow
 */
function goToCreateWorkflowUrl() {
    var url = '/rundb/api/v1/plugin/IonReporterUploader/extend/newWorkflow/';
    $.ajax({
        async: false, //Bypass the popup blocker in chrome.
        url: url,
        data: {
            "format": "json",
            "id": $("input[name=irAccountId]").val()
        },

        success: function (data) {
            if (data["status"] != "false") {
                if (data["method"] == "get") {
                    window.open(data["workflowCreationLandingPageURL"], '_blank');
                } else {
                    var form = $("<form/>", {
                            action: data["workflowCreationLandingPageURL"],
                            method: "post",
                            target: "_blank"
                        }
                    );
                    form.append($("<input/>", {
                        name: "authToken",
                        value: data["token"]
                    }));
                    form.appendTo("body");
                    form.submit();
                }
            } else {
                alert("Error fetching IR workflow creation url.")
            }
        },
        error: function () {
            alert("Failed to retrieve the workflow creation url, make sure you are able to connect to IR.");
        }

    });
}

$(document).ready(function () {
    //we block the UR and show the spinner until Ajax calls are complete
    $(document).ajaxStart($.blockUI).ajaxStop($.unblockUI);

    $("#loading").parent().parent().css('overflow', 'auto');
    //retrieve all IR accounts and their workflows/application types/relationship types    
    get_ir_accounts();

    //attach event to the workflow drop down list to show
    //the name of the workflow in the summary box 
    $('select[name=irworkflow]').click(function(){
        var irworkflow = $(this).val() ? $(this).val() : 'Upload Only';
        $('#selectedWorkflow').text(irworkflow);
    });

    //refresh button which retrieves the workflows, etc for the selected account
    $("#ir_refresh").on('click', function(){
        get_workflow_and_sample_groupings($('input[name="irAccountId"]').val(), $('input[name="irAccountName"]').val());
    });

    //let stuff be configed
    $('#iruConfig').click(function(e){
        e.preventDefault();
        var url = $(this).attr('href');
        var modal = $("#modal-window");
        var content = $("#modal-content");
        $("h3", modal).text("Ion Reporter Uploader Configuration");
        content.attr('src', url);
        content.ready(function() {
            $('iframe.auto-height').iframeAutoHeight({minHeight: 400, heightOffset: 20});
            modal.modal("show");
        });
    });

    $('.closeIRU').on('click', function () {
        $("#ir_accounts").html("");
        $(document).ajaxStart($.blockUI).ajaxStop($.unblockUI);
        $("#loading").parent().parent().css('overflow', 'auto');
        get_ir_accounts();
    });

    //attach change even to the workflow that auto selects the sample group
    //that corresponds to that workflow
    $("select[name=irworkflow]").on('change', function(e){
        $("input[name=applicationType]").val(IONREPORTER.workflow_to_application_type_map[$(this).val()]);
        $("#selectedWorkflow").text($(this).val());
        if ($(this).val().length > 0) {
            var pk = IONREPORTER.sample_group_to_pk_map[IONREPORTER.workflow_to_sample_group_map[$(this).val()]];
                $.each($("input[name=sampleGrouping]"), function(){
                    if ($(this).val() == pk) {
                        $(this).attr('checked', true);
                        $("#selectedGroup").text($(this).parent().text().trim());
                    }
                });
            }
        });

    });
