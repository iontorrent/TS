
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
