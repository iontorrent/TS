/**
    NOTE:  IRU and TS terminology differs slightly.  In IRU, the column "Relation" is equivalent to TS' "RelationRole",
    and IRU's column RelationshipType is TS' "Relation" in the JSON blob that is saved to the selectedPlugins BLOB.

    In this script, we do a one off on the column "RelationshipType", and create a hidden column for it and when
    we encounter the column "Relation", we name the associated drop down list "RelationRole" for the sake of sanity.
*/
var USERINPUT = USERINPUT || {};

USERINPUT.user_input_url = "/rundb/api/v1/plugin/IonReporterUploader/extend/userInput/";

USERINPUT.sample_group_to_workflow_map = {};
USERINPUT.workflow_to_application_type_map = {};

USERINPUT.workflow_to_ocp_map = {};
USERINPUT.workflow_ocp_list = [];
USERINPUT.decorated_workflow_for_display_map = {};
USERINPUT.gender_list = [];
USERINPUT.relation_to_gender_map = {};

function retrieve_application_type(data) {
    var column_map = data["sampleRelationshipsTableInfo"]["column-map"];
    $.each(column_map, function(i){
        var _dict = column_map[i];
        if (_dict['Workflow'] == USERINPUT.workflow) {
            $("input[name=applicationType]").val(_dict['ApplicationType']);
        }
    });

}

function create_iru_ui_elements(data) {
    
    var columns = data["sampleRelationshipsTableInfo"]["columns"];

    //sorts the columns by the Order key
    columns.sort(function(columnA, columnB){
        return columnA.Order - columnB.Order;
    });

    var $th = $("#chipsets thead tr");
    var $rows = $("#chipsets tbody tr");

    $.each(columns, function(i){
        var th_count = $th.children().length;
        var last_th = $th.find("th:eq("+(th_count-1)+")");

        var name = columns[i]["Name"];

        if (name == "Workflow") {
        	name = "Ion Reporter Workflow";
        }

        if (name == "SetID") {  
        	name = "Analysis Set ID"  
        	var $new_th = $("<th></th>", {"style" : "display: table-cell;", "name" : "ir"+name, "class" : "k-header k-widget", "rel" : "tooltip", "title" : "After file transfer, in Ion Reporter Software, samples with the same Analysis Group ID are considered related samples and are launched in the same analysis, such as a normal sample and its corresponding tumor samples. Do not give unrelated samples the same Analysis Group ID value (even if that value is zero or blank)."});
        		
        } else if (name == "RelationshipType" || name == "NucleotideType" || name == "CellularityPct" || name == "CancerType") {
        //20140317-temp } else if (name == "RelationshipType") { 
            //we are not creating a header for the relationshipType
            //we are simply creating a hidden element below
            return;
        } else {
                var widthClass = "input-medium";
                var widthStyle = "display: table-cell; ";
                if (name == "Ion Reporter Workflow") {
                        widthClass = "";
                        widthStyle = "width : 350px; display : table-cell; ";
                }
            	var $new_th = $("<th></th>", {"style" : widthStyle, "name" : "ir"+name, "class" : "k-header k-widget " + widthClass});

        }
        
        $new_th.text(name);
        $th.append($new_th);
    });


    //loop through the rows and create an entry for each column

    var isDualNucleotideType = $('input[id=isDualNucleotideTypeBySample]').val();
    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();         
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

    $.each($rows, function(i, row){
        var $row = $(row);
        //console.log("at iru_get_user_input.create_iru_ui_elements() each i=", i, "; row=", $row);
        var isToDisable = false;
    	
    	
        if ((isDualNucleotideType == "True") && (isBarcodeKitSelection == "True") && (isSameSampleForDual) && (i % 2 != 0)) {
        	isToDisable = true;
        }

        //console.log("iru_get_user_input isSameSampleForDual=", isSameSampleForDual, "; isToDisable=", isToDisable);
        
        $.each(columns, function(j, column){
            //columns are sorted so get them in order
            var name = column["Name"];
            var $elem, $new_td;

            if (name == "SetID") {           
                $elem = $("<input/>", {"type" : "text", "name" : "ir"+name, "style" : "width:100px; display: table-cell", "class" : "ir"+name});
                
                if (isToDisable) {
                	$elem = $("<input/>", {"type" : "text", "name" : "ir"+name,  "disabled" : "disabled", "style" : "width:100px; display: table-cell", "class" : "ir"+name});
                }
            } else if (name == "RelationshipType" || name == "NucleotideType" || name == "CellularityPct" || name == "CancerType") {
            	//20140317-temp
            	//no-op for now
            	
            	return;
            } else {
                var widthClass = "input-medium";
                var widthStyle = "display: table-cell; ";
                if (name == "Workflow") {
                        widthClass = "";
                        widthStyle = "width : 350px; display : table-cell; ";
                }
            	
                // we need to rename "Relation" to RelationRole as explained 
                //at the top of the script
                if (name == "Relation") {name = "RelationRole";}
                $elem = $("<select></select>", {"style" : widthStyle, "name" : "ir"+name, "class" : widthClass +" ir"+name });

                if (isToDisable) {
                	$elem = $("<select></select>", {"style" : widthStyle, "name" : "ir"+name, "disabled" : "disabled", "class" : widthClass +" ir"+name});
                }
            }
            $new_td = $("<td></td>");
            $new_td.append($elem);
            $row.append($new_td);
           
        });
        // This implementation is identical to what get's created if IR returns this column, so
        $row.find("td").last().append(
             $("<input/>", {"type" : "hidden", "name" : "irRelationshipType", "class" : "irRelationshipType"})
        );

    });

    USERINPUT.irGenderSelects = $(".irGender");
    USERINPUT.irWorkflowSelects = $(".irWorkflow");
    USERINPUT.irSetIDInputs = $(".irSetID");
    USERINPUT.irRelationRoleSelects = $(".irRelationRole");
    //20140306-WIP
    //USERINPUT.irCancerType = $(".ircancerType");
    //USERINPUT.irCellularityPct = $(".ircellularityPct");

} 


/**
 *  If OCP is enabled and the plan's application group is DNA_RNA, show the OCP workflows.  
 *  Otherwise, hide them. 
 */
function filter_workflow_ui_elements($workflowSelects) {
	var isOcpEnabled = $('input[id=isOCPEnabled]').val();
	var isOCPApplGroup = $('input[id=isOCPApplicationGroup]').val();	

	//if both isOcpEnabled and isOCPAplGroup are true, we'll show the OCP workflows. Otherwise, hide it!
	if (isOcpEnabled.toLowerCase() == "true" && isOCPApplGroup.toLowerCase() == "true") {
		return;
	}
	
    var $index = 0;
    var values = USERINPUT.workflow_ocp_list;
    $.each(values, function(i, ocpWorkflow){
	    $workflowSelects.children().filter(function(index, option) {
	    	//console.log("filter_workflow_ui_elements() index=", index, "; option.value=", option.value);
	    	
	        $index = index;
	        return option.value === ocpWorkflow;
	    }).remove();;
    });	

}

/**
    This is an abstract function that populates the Gender and Workflow drop down lists
    based on a name parameter and a list of values to populate
*/
function add_ir_general_to_select(name, values) {
//	if (name == "Workflow") {
//		console.log("add_ir_general_to_select() name=", name, "; values=", values);
//	}
	
	if (name == "Gender") {
		USERINPUT.gender_list.length = 0;
	}
	
    var $selects = $(".ir"+name+"");
    $.each($selects, function(j){
        var $select = $(this);
        
        $select.append($("<option></option>"));
        $.each(values, function(i,value){
            var $opt = $("<option></option>");
            $opt.attr('value', value);

            if (name == "Workflow" && !(jQuery.isEmptyObject(USERINPUT.decorated_workflow_for_display_map)) && (value in USERINPUT.decorated_workflow_for_display_map)) {
                var displayedValue = USERINPUT.decorated_workflow_for_display_map[value];
                $opt.text(displayedValue);
            }
            else {
            	$opt.text(value);
            }
            $select.append($opt);
            
            if (name == "Gender") {
            	//keep a distinct list of gender choices
            	if ($.inArray(value, USERINPUT.gender_list) === -1) {
            		USERINPUT.gender_list.push(value);
            	}
            }
        });
    });
}

/**
    This function extracts that invalid value for the relation type
    out of the restrictions list returned by the API
*/
function set_invalid_value(restrictions) {
    var  invalid_value;
    $.each(restrictions, function(i){
    	var restriction = restrictions[i];
        if (typeof restriction["Disabled"] != 'undefined' && restriction["For"]["Name"] == 'RelationshipType') {
    		invalid_value = restriction["For"]["Value"];
    	}
    });

    return invalid_value;
}


/**
    This function creates a two dimentional map of workflows to relationshipType and relationRole values
    extracted from the data returned by the API
*/
function populate_sample_grouping_to_workflow_map(data) {
    var column_map = data["sampleRelationshipsTableInfo"]["column-map"];
    var restrictions = data["sampleRelationshipsTableInfo"]["restrictionRules"];

    $.each(column_map, function(i){
        var cm = column_map[i];
        var workflow = cm["Workflow"];
        var relationshipType = cm["RelationshipType"];
        USERINPUT.workflow_to_application_type_map[workflow] = cm["ApplicationType"];

        if ("OCP_Workflow" in cm) {
        	USERINPUT.workflow_to_ocp_map[workflow] =  cm["OCP_Workflow"];

        	if (cm["OCP_Workflow"].toLowerCase() == "true") {
            	//keep a distinct list of ocp workflows
            	if ($.inArray(workflow, USERINPUT.workflow_ocp_list) === -1) {
            		USERINPUT.workflow_ocp_list.push(workflow);
            	}
            	var displayedWorkflow =  workflow + " (" + cm["DNA_RNA_Workflow"] + ")";
                if ($.inArray(displayedWorkflow, USERINPUT.workflow_with_sample_grouping_ocp_list) === -1) {
                    USERINPUT.decorated_workflow_for_display_map[workflow] = displayedWorkflow;
                }            	
        	}
        	else {
                USERINPUT.workflow_to_ocp_map[workflow] = "false";
            }       	
        }
        else {
        	USERINPUT.workflow_to_ocp_map[workflow] = "false";
        	if (workflow.toLowerCase() == "upload only") {
        	   USERINPUT.decorated_workflow_for_display_map[workflow] = workflow;
        	}
        }
        
        //console.log("populate_sample_grouping_to_workflow_map() workflow_ocp_list=", USERINPUT.workflow_ocp_list);
        
        $.each(restrictions, function(j){
            var restriction = restrictions[j];
            if (typeof restriction["For"] != 'undefined') {
                if (restriction["For"]["Name"] == 'RelationshipType' && restriction["For"]["Value"] == relationshipType) {
                        
                    USERINPUT.sample_group_to_workflow_map[workflow] = {};
                    USERINPUT.sample_group_to_workflow_map[workflow]["relationshipType"] = restriction["For"]["Value"]
                    USERINPUT.sample_group_to_workflow_map[workflow]["relations"] = restriction["Valid"]["Values"];
                        
                }
                else if (restriction["For"]["Name"] == 'Relation') {
                	USERINPUT.relation_to_gender_map[restriction["For"]["Value"]] = restriction["Valid"]["Values"];
                }
            }
        });

    });

    //console.log("populate_sample_grouping_to_workflow_map relation_to_gender_map=", USERINPUT.relation_to_gender_map);
}

/**
    This function presets the IR fields based on either what was in the sample set if it is a new plan
    or what was saved if it is an existing plan
*/
function preset_ir_fields(counter, irSampleName, irGender, irCancerType, irCellularityPct, irWorkflow, irRelationshipType, irRelationRole, irSetID) {
	//console.log("at iru_get_user_input.preset_ir_fields() irSampleName=", irSampleName, "; irWorkflow=", irWorkflow);
	
    var $genderSelect = $(USERINPUT.irGenderSelects[counter]);
    var $workflowSelect = $(USERINPUT.irWorkflowSelects[counter]);
    var $relationRoleSelect = $(USERINPUT.irRelationRoleSelects[counter]);
    var $irSetIDInput = $(USERINPUT.irSetIDInputs[counter]);
        
    if (irGender.length > 0) {
        var irGenderTerms = USERINPUT.ir_sample_to_tss_sample[irGender];
            
        if (irGenderTerms.length > 1) {
            irGender = irGenderTerms[1];
        } else {
            irGender = irGenderTerms[0];
        }

        var matchingGender = $genderSelect.find("option").filter(function () { 
                    return this.value.toLowerCase() == irGender.toLowerCase(); 
        }).attr('value');    
        $genderSelect.val(matchingGender); 

    }

    if (irSampleName != '') {
        $workflowSelect.find("option[value='"+irWorkflow+"']").attr('selected', true);        
    } else {
        $workflowSelect.find("option[value='"+USERINPUT.workflow+"']").attr('selected', true);
    }

    $workflowSelect.change();

    if (irRelationRole.length > 0) {
        var irRelationRoleTerms = USERINPUT.ir_sample_to_tss_sample[irRelationRole];
        if (typeof irRelationRoleTerms != 'undefined') {
            
            $.each(irRelationRoleTerms, function(i){
                var irRelationRole1 = irRelationRoleTerms[i];
                    
                var matchingRole = $relationRoleSelect.find("option").filter(function () { 
                        return this.value.toLowerCase() == irRelationRole1.toLowerCase(); 
                }).attr('value');    
                if (matchingRole) {
                    $relationRoleSelect.val(matchingRole);
                    return;
                }

                
            });
        } else {
            var matchingRole = $relationRoleSelect.find("option").filter(function () { 
                    return this.value.toLowerCase() == irRelationRole.toLowerCase(); 
            }).attr('value');    
            $relationRoleSelect.val(matchingRole);
        }
    }

    if (irRelationshipType.length > 0) {
        var $tr = $genderSelect.parent().parent();
        var $relationshipTypeHiddenInput = $tr.find(".irRelationshipType");
        $relationshipTypeHiddenInput.val(irRelationshipType);
    }

    $irSetIDInput.val(irSetID);
}

/**
    This function populates the relation roles based on Workflow
*/
function set_relations_from_workflow($relation, relations, workflow, $gender) {
	//console.log("set_relations_from_workflow() workflow=", workflow, "; relations=", relations);
        
    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

    $relation.empty();
    $relation.append($("<option></option>"));

    $.each(relations, function(i){
        var $opt = $("<option></option>");
        $opt.val(relations[i]);
        $opt.text(relations[i]);
        $relation.append($opt);
    });
    
    if(relations.length == 1) {
    	$relation.val(relations[0]);
    	if (!isSameSampleForDual) {
    		$gender.attr('disabled', false);
    	}
    }
}


/**
    This function populates the genders based on relation selected
*/
function set_genders_from_relation($gender, relation_gender_map, relation) {
	console.log("set_genders_from_relation() relation=", relation, "; relation_gender_map=", relation_gender_map);
	
    $gender.empty();
    $gender.append($("<option></option>"));

    $.each(relation_gender_map, function(i){
        var $opt = $("<option></option>");
        $opt.val(relation_gender_map[i]);
        $opt.text(relation_gender_map[i]);
        $gender.append($opt);
    });
    
//    if (relation_gender_map.length == 0) {
//        //$gender.attr('disabled', true); 
//    }
//    else {
//    	$gender.attr('disabled', false);    	
//    }

    if (relation_gender_map.length == 1) {
    	$gender.val(relation_gender_map[0]);
    }
}


/**
This function populates the genders with the complete list we have at hand
*/
function reset_genders($gender, values) {
	//console.log("ENTER reset_genders() values=", values);
	
    $gender.empty();
    $gender.append($("<option></option>"));
    
    $.each(values, function(i,value){
        var $opt = $("<option></option>");
        $opt.attr('value', value);
        $opt.text(value);
        $gender.append($opt);
    });	
    
    if (values.length == 1) {
    	$gender.val(values[0]);
    	//$gender.attr('disabled', false);
    }
//    if (values.length == 0) {                  	
//    	$gender.attr('disabled', true);
//    }
//    else {
//    	$gender.attr('disabled', false);
//    }
}

/**
    This function searches the API returned result for a relationship type 
    for a given workflow
*/
function find_relationship_type(workflow, columns_map) {
    var relationshipType = "";
    $.each(columns_map, function(i){
        var column = columns_map[i];
        if (column["Workflow"] == workflow) {
            relationshipType = column["RelationshipType"];
        }
    });
    return relationshipType;
}


/**
* Filter IR workflows based on runType and application group
*/
function get_workflow_url() {
	var applicationGroupName = $('input[name=applicationGroupName]').val();  
	var runType_name = $('input[name=runType_name]').val();
	var runType_nucleotideType = $('input[name=runType_nucleotideType]').val();
	
	var planCategories = $('input[name=planCategories]').val();
	console.log("iru_get_user_input.get_workflow_url() applicationGroupName=", applicationGroupName, "; runType_name=", runType_name, "; runType_nucleotideType=", runType_nucleotideType, "; planCategories=", planCategories);
  
	var myURL = USERINPUT.user_input_url;
  	
	myURL += "?format=json&id=" + USERINPUT.account_id;
	var isFilterSet = false;
	
	if (runType_nucleotideType.toLowerCase() == "dna" || (runType_nucleotideType == "" && applicationGroupName.toLowerCase() == "dna")) {
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


function load_and_set_ir_fields() {
    var myURL = get_workflow_url();
    	
    var jqhxhr = $.ajax({
        type : "get",
        url : myURL,
        timeout: 6000,//in milliseconds
        success: function(data){
            if (typeof data["sampleRelationshipsTableInfo"] == 'undefined') {
                // failed to retrieve info
                $("#loading").hide();
                USERINPUT.is_ir_connected = false;
            	$("#error").text("Cannot contact Ion Reporter Server!");
                $("input[name=irDown]").val('1');
                $("#chipsets tr").find("[name^=ir]").hide();
            	return;
            }
            else {
            	USERINPUT.is_ir_connected = true;
            }
            
            $("#error").text("");

            populate_sample_grouping_to_workflow_map(data);

            if(USERINPUT.irWorkflowSelects === undefined){
                create_iru_ui_elements(data);
            }
            retrieve_application_type(data);

            prepareSampleIRConfiguration();
            
            var columns = data["sampleRelationshipsTableInfo"]["columns"];
            var restrictions = data["sampleRelationshipsTableInfo"]["restrictionRules"];
            
            if (typeof columns != 'undefined') {

                $("input[name=irDown]").val('0');
                var invalid_value = set_invalid_value(restrictions);

                // populate IR dropdowns (except Relation which is Workflow-dependent)
                $.each(columns, function(i){
                    var result = columns[i];
                    var name = result["Name"];
                    var values = result["Values"];
                    if (typeof values != 'undefined' && name != "Relation") {
                        add_ir_general_to_select(name, values);
                    }
                });

                // create Workflows onChange event to populate Relation and hidden RelationshipType
                USERINPUT.irWorkflowSelects.on('change', function(){

                    var $tr = $(this).parent().parent();
                    var $relation = $tr.find(".irRelationRole");
                    var $relationshipType = $tr.find(".irRelationshipType");
                    var $gender = $tr.find(".irGender");
                    
                    var workflow = $(this).val();
                    var workflow_map = USERINPUT.sample_group_to_workflow_map[workflow];

                	//console.log("iru_get_user_input.load_and_set_ir_fields() workflow=", workflow, "; workflow_map=", workflow_map);


                    var isBarcodeKitSelection = $('input[id=isBarcodeKitSelectionRequired]').val();                    
                    var isSameSampleForDual = $('input[id=isOncoSameSample]').is(":checked");

                    if (typeof workflow_map != 'undefined') {
                    	//console.log("iru_get_user_input.load_and_set_ir_fields() GOING to call set_relations_from_workflow -- workflow=", workflow, "; invalid_value=", invalid_value);
                            
                        var relationshipType = workflow_map["relationshipType"];
                        var relations = workflow_map["relations"];

                        // set Relation (aka RelationRole) selects
                        set_relations_from_workflow($relation, relations, workflow, $gender);

                        if (relationshipType != invalid_value) {
                        	//we could be disabling an IR field intentionally, don't override it here
                        	if (! isSameSampleForDual) {
                        		$relation.attr('disabled', false);
                        	}
                        	else {
                        		if (isSameSampleForDual && isBarcodeKitSelection && $tr.index() == 0) {

                        			var row1 = $('#row1');                                    
                                    var $relationshipTypeHiddenInput1 = row1.find(".irRelationshipType");

                                    //console.log("iru_get_user_input relationshipType=", relationshipType);
                                    $relationshipTypeHiddenInput1.val(relationshipType);
                        		}
                        	}
                        } else {
                            $relation.empty();
                            $relation.attr('disabled', true);
                        }
                        
                        // set hidden RelationshipType
                        $relationshipType.val(relationshipType);
                        
                    } else { // we remove this Workflow from the drop down list
                        var $index = 0;
                        $(this).children().filter(function(index, option) {
                            $index = index;
                            return option.value===workflow;
                        }).remove();
                        //now change the value to the next option
                        $(this).children(":eq("+($index+1)+")").attr('selected', true);
                        $(this).change();

                    }
                });
                

                // create relation onChange event to populate gender                
                USERINPUT.irRelationRoleSelects.on('change', function(){

                    var $tr = $(this).parent().parent();
                    var $gender = $tr.find(".irGender");
                       
                    var relation = $(this).val();
                    var relation_gender_map = USERINPUT.relation_to_gender_map[relation];

                    if (typeof relation_gender_map != 'undefined') {
                        // set gender selects
                        set_genders_from_relation($gender, relation_gender_map, relation);                       
                    } 
                    else {
                    	reset_genders($gender, USERINPUT.gender_list);
                    }
                });
                
                
                // Set all IR fields based on USERINPUT values
                $.each(USERINPUT.preset_ir_fields, function(i){
                    var _array = USERINPUT.preset_ir_fields[i];
                    //console.log("iru_get_user_input preset_ir_fields each _array=", _array);
                    
                    //planByTemplate  pre_ir_fields _array= ["0", "s 1", "Male", "", "", "my ir workflow", "Self", "Self", "2"] (length = 9)
                    //planBySampleSet pre_ir_fields _array= ["0", "s 1", "Male", "", "", "", "1"] (length = 7)
                    //console.log("iru_get_user_input i=", i, "; _array.length=",  _array.length, "; pre_ir_fields _array=", _array);
                    //TODO: TEMP workaround for planBySampleSet for now
                    if (_array.length == 7) {
                    	preset_ir_fields(_array[0], _array[1], _array[2], "", "", _array[3], _array[4], _array[5],_array[6]);
                    }
                    else {
                    	preset_ir_fields(_array[0], _array[1], _array[2], _array[3], _array[4], _array[5],_array[6], _array[7], _array[8]);
                    }
                });
                
                //filter_workflow_ui_elements(USERINPUT.irWorkflowSelects);
                
                $("#chipsets").change();
            } 
                
            $("#loading").hide();
            
        },
        error: function(jqXHR, textStatus, errorThrown){
            if(textStatus==="timeout") {
                //The API Server timedout
                $("#loading").hide();
                $("#error").text('The IR Server connection has timed out');
                $("input[name=irDown]").val('1');
                return;
            }
        }
    });
}

$(document).ready(function(){
    //handler for asynchroneous Ajax calls to block and unblock the UI
    $(document).ajaxStart($.blockUI).ajaxStop($.unblockUI);

    if (USERINPUT.account_id != "0" && USERINPUT.account_id != "-1"){
        load_and_set_ir_fields();
    } 
});
