
function get_decorated_workflow_name(workflowObj, workflowName, irReference, ionLabel, separator) {
    var decorated_workflow_name = workflowName;
    var isOCP = false;

    var isImmuneRepertoire = false;
    var DNA_LABEL = "DNA";
    var RNA_LABEL = "RNA";
    if ("tag_IMMUNE_REPERTOIRE_SHORT_ASSAY" in workflowObj && workflowObj["tag_IMMUNE_REPERTOIRE_SHORT_ASSAY"].toLowerCase() == "true") {
    	isImmuneRepertoire = true;
    	if ("tag_DNA" in workflowObj && workflowObj["tag_DNA"].toLowerCase() == "true") {
    		var displayedWorkflow =  workflowName + " (" + DNA_LABEL + ")";
    		decorated_workflow_name += " (" + DNA_LABEL;
    	}
    	else {
	    	if ("tag_RNA" in workflowObj && workflowObj["tag_RNA"].toLowerCase() == "true") {
	    		var displayedWorkflow =  workflowName + " (" + RNA_LABEL + ")";
	    		decorated_workflow_name += " (" + RNA_LABEL;
	    	}
    	}
    }
    else if ("OCP_Workflow" in workflowObj && workflowObj["OCP_Workflow"].toLowerCase() == "true") {
        isOCP = true;
        var displayedWorkflow =  workflowName + " (" + workflowObj["DNA_RNA_Workflow"] + ")";
        decorated_workflow_name += " (" + workflowObj["DNA_RNA_Workflow"];
    }     
    if (irReference !== "") {
        decorated_workflow_name += separator + irReference;
    }
    if ("tag_isFactoryProvidedWorkflow" in workflowObj && workflowObj["tag_isFactoryProvidedWorkflow"].toLowerCase() == "true") {
        decorated_workflow_name +=  separator + ionLabel;
    }
    if (isOCP || isImmuneRepertoire) {
        decorated_workflow_name += ")";
    }

    return decorated_workflow_name;
}


/**
 * Filter IR workflows based on runType and application group
 */
function get_workflow_url(start_url, account_id) {
    var applicationGroupName = $('input[name=applicationGroupName]').val();
    var runType_name = $('input[name=runType_name]').val();
    var runType_nucleotideType = $('input[name=runType_nucleotideType]').val();
    var planCategories = $('input[name="planCategories"]').val();
    console.log("get_workflow_url() applicationGroupName=", applicationGroupName, "; runType_name=", runType_name, "; runType_nucleotideType=", runType_nucleotideType, "; planCategories=", planCategories);

    var myURL = start_url + "?format=json&id=" + account_id;
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

	if (runType_nucleotideType.toLowerCase() == "dna_rna" && applicationGroupName.toLowerCase() == "immune_repertoire") {
        myURL += "&filterKey=tag_IMMUNE_REPERTOIRE_SHORT_ASSAY&filterValue=";
        myURL += "true";

        isFilterSet = true;
	}
	else {
        if (runType_name.toLowerCase().startsWith("amps_hd")){
            myURL += "&andFilterKey2=tag_AMPLISEQHD&andFilterValue2=true";

            if (applicationGroupName == "DNA + RNA"){
                myURL += "&filterKey=DNA_RNA_Workflow&filterValue=DNA_RNA";
            } else if (applicationGroupName == "DNA + RNA 1"){
                myURL += "&filterKey=ApplicationType&filterValue=AmpliSeqHD_Single_Pool";
            }
        }
	    else if (applicationGroupName == "DNA + RNA") {
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
	            else {
	                myURL += "&andFilterKey2=Onconet_Workflow&andFilterValue2=true";
	            }
	        }
	    }
	    else {
	        if (runType_name.toLowerCase() != "amps") {
	            if (!isFilterSet) {
	                myURL += "&filterKey=Onconet_Workflow&filterValue=false";
	            }
				if (applicationGroupName == "onco_liquidBiopsy") {
					myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";
				}
				else {
                    myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=false";
	            }
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
    }
    return myURL;
}
