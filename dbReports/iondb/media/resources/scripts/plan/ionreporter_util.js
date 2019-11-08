
function get_decorated_workflow_name(workflowObj, workflowName, irReference, ionLabel, separator) {
    var decorated_workflow_name = workflowName;
    var isOCP = false;

    var isImmuneRepertoire = false;
    var DNA_LABEL = "DNA";
    var RNA_LABEL = "RNA";
    if ("ApplicationType" in workflowObj && workflowObj["ApplicationType"].toLowerCase() == "immunerepertoire") {
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


/* IR workflow filtering:
 * 1. only one pair of "filterKey" and "filterValue" will be active
 *    and it is the last one in URL query
 * 2. additional filter need to use "andFilterKey2" and "andFilterValue2"
 */
function build_ir_workflow_filters(filterKey, filterValue, isFilterSet = false) {
    var subURL = "";
    if (isFilterSet) {
        subURL = "&andFilterKey2=" + filterKey + "&andFilterValue2=" + filterValue;
    } else {
        subURL = "&filterKey=" + filterKey + "&filterValue=" + filterValue;
    }
    return subURL;
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
        // myURL += "&filterKey=DNA_RNA_Workflow&filterValue=";
        // myURL += "DNA";
        myURL += build_ir_workflow_filters("DNA_RNA_Workflow", "DNA", false);
        isFilterSet = true;
    }
    else if (runType_nucleotideType.toLowerCase() == "rna" || (runType_nucleotideType == "" && applicationGroupName.toLowerCase() == "rna")) {
        // myURL += "&filterKey=DNA_RNA_Workflow&filterValue=";
        // myURL += "RNA";
        myURL += build_ir_workflow_filters("DNA_RNA_Workflow", "RNA", false);
        isFilterSet = true;
    }

    // if Immune Repertoire, only use ApplicationType filtering
	if (applicationGroupName.toLowerCase() == "immune_repertoire") {
        myURL += build_ir_workflow_filters("ApplicationType", "ImmuneRepertoire", isFilterSet);
        console.log("myURL: " + myURL);
        return myURL;
    }

    // if Application Categories contains 'CarrierSeq'
    if (planCategories.toLowerCase().indexOf("carrierseq") != -1) {
        myURL += build_ir_workflow_filters("tag_CARRIERSEQ", "true", isFilterSet);
        console.log("myURL: " + myURL);
        return myURL;
    }
    
    // use build_ir_workflow_filters to visualize filter key and value
    if (runType_name.toLowerCase().startsWith("amps_hd")){
        // myURL += "&andFilterKey2=tag_AMPLISEQHD&andFilterValue2=true";
        myURL += build_ir_workflow_filters("tag_AMPLISEQHD", "true", true);

        if (applicationGroupName == "DNA + RNA"){
            // myURL += "&filterKey=DNA_RNA_Workflow&filterValue=DNA_RNA";
            myURL += build_ir_workflow_filters("DNA_RNA_Workflow", "DNA_RNA", false);
        } else if (applicationGroupName == "DNA + RNA 1"){
            // myURL += "&filterKey=ApplicationType&filterValue=AmpliSeqHD_Single_Pool";
            myURL += build_ir_workflow_filters("ApplicationType", "AmpliSeqHD_Single_Pool", false);
        }
    }
    else if (applicationGroupName == "DNA + RNA") {
        /*for mixed single & paired type support
        if (runType_nucleotideType.toLowerCase() == "dna_rna") {
            myURL += "&filterKey=DNA_RNA_Workflow&filterValue=";
            myURL += "DNA_RNA";
            isFilterSet = true;
        }
        myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";
        */
        if (planCategories.toLowerCase().indexOf("oncomine") != -1) {
            /* 
            if (!isFilterSet) {
                myURL += "&filterKey=Onconet_Workflow&filterValue=false";
            }
            myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";
            */
            myURL += build_ir_workflow_filters("OCP_Workflow", "true", true);
        }
        else if (planCategories.toLowerCase().indexOf("onconet") != -1) {
            /*
            if (!isFilterSet) {
                myURL += "&filterKey=Onconet_Workflow&filterValue=true";
            }
            else {
                myURL += "&andFilterKey2=Onconet_Workflow&andFilterValue2=true";
            }
            */
            myURL += build_ir_workflow_filters("Onconet_Workflow", "true", isFilterSet);
        }
    }
    else {
        if (runType_name.toLowerCase() != "amps") {
            if (!isFilterSet) {
                // myURL += "&filterKey=Onconet_Workflow&filterValue=false";
                myURL += build_ir_workflow_filters("Onconet_Workflow", "false", isFilterSet);
            }

            if (applicationGroupName == "onco_liquidBiopsy") {
                // myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";
                myURL += build_ir_workflow_filters("OCP_Workflow", "true", true);
            }
            else {
                // myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=false";
                myURL += build_ir_workflow_filters("OCP_Workflow", "false", true);
            }
        }
        else {
            if (planCategories.toLowerCase().indexOf("oncomine") != -1) {
                // myURL += "&andFilterKey2=OCP_Workflow&andFilterValue2=true";
                myURL += build_ir_workflow_filters("OCP_Workflow", "true", true);
            }
            else if (planCategories.toLowerCase().indexOf("onconet") != -1) {
                // myURL += "&andFilterKey2=Onconet_Workflow&andFilterValue2=true";
                myURL += build_ir_workflow_filters("Onconet_Workflow", "true", true);
            }
        }
    }
    console.log("myURL: " + myURL);
    return myURL;
}
