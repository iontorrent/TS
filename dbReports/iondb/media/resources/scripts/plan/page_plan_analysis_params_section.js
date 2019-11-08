
function hide_summary_view() {
    $("#sidebar").hide();
    $("#mainContent").removeClass("span8");
    $("#mainContent").addClass("span12");
    $("#showSummary").show();
}

function show_summary_view() {  
    $("#sidebar").show();
    $("#mainContent").removeClass("span12");
    $("#mainContent").addClass("span8");
    $("#showSummary").hide();
}
    


$(document).ready(function () {      
 
    $("#analysis_params_select").change(function()  {
        var selectedAnalysis = $(this).val();
	//console.log("page_plan_analysis_params_section.html - selectedAnalysis=", selectedAnalysis);
	if (selectedAnalysis){
	    updateAnalysisParamsDetails(selectedAnalysis);
	}
    });

    $('[name=analysisParamsCustom]').change(function(){
	if (this.value == 'False'){
	    $("#analysis_params_select .best_match").prop('selected',true).change();
	}else{
	    if (!$(".extra_analysis_params_info").is(":visible")){
		$(".extra-analysis-params-inline").text(gettext('workflow.step.analysisparams.action.viewupdate.label.collapse'));//"Details -"
		$(".extra_analysis_params_info").show();
		hide_summary_view();
	    }
	}
	updateAnalysisParamsDisplay();
    });

    $('#analysis_params_section textarea').on('change keypress paste', function(){
	$("#analysis_params_select").val('');
    });
    
    $(".extra_analysis_params_info").hide();
 
    $(".extra-analysis-params-inline").click(function() {
        if ($(".extra_analysis_params_info").is(":visible")) {
            $(".extra-analysis-params-inline").text(gettext('workflow.step.analysisparams.action.viewupdate.label.expand'));
            $(".extra_analysis_params_info").hide();
            show_summary_view();
        } else {
            $(".extra-analysis-params-inline").text(gettext('workflow.step.analysisparams.action.viewupdate.label.collapse'));
            $(".extra_analysis_params_info").show();
            hide_summary_view();
        }
    });

    function  updateAnalysisParamsDetails(selectedAnalysis) {
    	console.log("selected analysis entry=", selectedAnalysis);

    	var paramSet = ANALYSIS_ENTRIES[selectedAnalysis];
    	//console.log("...paramSet=", paramSet, "; analysisArgsSelected=", ANALYSIS_ENTRIES_ANALYSIS[selectedAnalysis]);
        $("#beadFindSelected").val(ANALYSIS_ENTRIES_BEADFIND[selectedAnalysis]);
        $("#analysisArgsSelected").val( ANALYSIS_ENTRIES_ANALYSIS[selectedAnalysis]);
        $("#preBaseCallerSelected").val(ANALYSIS_ENTRIES_PREBASECALLER[selectedAnalysis]);
        $("#calibrateSelected").val(ANALYSIS_ENTRIES_CALIBRATE[selectedAnalysis]);
        $("#baseCallerSelected").val(ANALYSIS_ENTRIES_BASECALLER[selectedAnalysis]);
        $("#alignmentSelected").val(ANALYSIS_ENTRIES_ALIGNMENT[selectedAnalysis]);
        $("#ionStatsSelected").val(ANALYSIS_ENTRIES_IONSTATS[selectedAnalysis]);

        $("#thumbnailBeadFindSelected").val(ANALYSIS_ENTRIES_THUMBNAIL_BEADFIND[selectedAnalysis]);
        $("#thumbnailAnalysisArgsSelected").val(ANALYSIS_ENTRIES_THUMBNAIL_ANALYSIS[selectedAnalysis]);
        $("#thumbnailPreBaseCallerSelected").val(ANALYSIS_ENTRIES_THUMBNAIL_PREBASECALLER[selectedAnalysis]);
        $("#thumbnailCalibrateSelected").val(ANALYSIS_ENTRIES_THUMBNAIL_CALIBRATE[selectedAnalysis]);
        $("#thumbnailBaseCallerSelected").val(ANALYSIS_ENTRIES_THUMBNAIL_BASECALLER[selectedAnalysis]);
        $("#thumbnailAlignmentSelected").val(ANALYSIS_ENTRIES_THUMBNAIL_ALIGNMENT[selectedAnalysis]);
        $("#thumbnailIonStatsSelected").val( ANALYSIS_ENTRIES_THUMBNAIL_IONSTATS[selectedAnalysis]);
    }
    
    function updateAnalysisParamsDisplay(){
	var default_args = $('[name=analysisParamsCustom]:checked').val() == "False";
	$('#analysis_params_section').each(function(){
	    $(this).find('input, textarea').attr("readonly", default_args);
	    $(this).find('select').attr("disabled", default_args);
	});
    }
    updateAnalysisParamsDisplay();
 
});  //end of document.ready
