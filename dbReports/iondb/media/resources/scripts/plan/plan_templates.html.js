
function commonKendoGrid(target, url, msg) {
	//console.log("ENTER commaonKendoGrid url=", url);
	//console.log("commaonKendoGrid msg=", msg);
	
    return {
        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : url,
                    contentType : 'application/json; charset=utf-8',
                    type : 'GET',
                    dataType : 'json'
                },
                parameterMap : function(options) {
                    return buildParameterMap(options);
                }
            },
            
            schema : {
                data : "objects",
                total : "meta.total_count",
                model : {
                    fields : {
                        id : {
                            type : "number",
                            editable : false
                        },
                        planDisplayedName : {
                            type : "string",
                            editable : false,
                        },
                        applicationGroupDisplayedName : {
                            type : "string",
                            editable : false
                        },
                        barcodeKitName : {
                            type : "string",
                            editable : true
                        },
                        date : {
                        	type : "string",
                        	editable : false
                        },
                        irAccountName : {
                        	type : "string"
                        },
                        irWokflow : {
                        	type : "string",
                        	editable : false
                        },
                        isFavorite : {
                        	type : "boolean",
                        	editable : false                        	
                        },
                        isSystem : {
                        	type : "boolean",
                        	editable : false
                        },                     
                        runType : {
                        	type : "string",
                        	editable : false                        	
                        },
                        reference : {
                        	type : "string",
                        	editable : false                           	
                        },
                        targetRegionBedFile : {
                        	type : "string",
                        	editable : false                           	
                        },
                        hotSpotRegionBedFile : {
                        	type : "string",
                        	editable : false                           	
                        },                        
                        sequencingInstrumentType : {
                        	type : "string",
                        	editable : false
                        },                     
                        templatePrepInstrumentType : {
                        	type : "string",
                        	editable : false                        	
                        },
                        notes : {
                        	type : "string",
                        	editable : false
                        },                          
                        username : {
                        	type : "string",
                        	editable : false
                        }                      
                    }
                }
            },            
            
            serverPaging : true,
            pageSize : 10,
            serverSorting : false
        },
        sortable: {
        	mode : "multiple",
        	allowUnsort : true
        },
        height : '460',
//        scrollable: {
//            virtual: true
//        },
        scrollable : {
            virtual : false
        },
        pageable : true,
                
		dataBinding : onDataBinding,
		dataBound : onDataBound,
        columns : [{
            field : "planDisplayedName",
            title : "Template Name",
            width : "18%",
            template : kendo.template($("#PlanDisplayedNameTemplate").html())
        }, {
        	field : "sequencingInstrumentType",
        	title : "Instr.",
            width : "5%",
        	template : kendo.template($("#SeqInstrumentTemplate").html())
        }, {
        	field : "templatePrepInstrumentType",
        	title : "OT/IC",
            width : "5%",        	      	
        	template : kendo.template($("#TemplatePrepInstrumentTemplate").html())        	
        }, {        	
        	field : "barcodeKitName",
        	title: "Barcode Kit",
            width : "8%",        	           
        	template : kendo.template($("#BarcodeKitNameTemplate").html())    
        }, {
        	field : "reference",
        	title : "Reference",
            width : "13%",
        	template : kendo.template($("#ReferenceTemplate").html())                 
        }, {
        	field : "irAccountName",
        	title : "Ion Reporter Account", 
            width : "13%",
        }, {
        	field : "irworkflow",
        	title : "Ion Reporter Workflow",
            width : "13%",           
        }, {
        	field : "date",
        	title : "Date",
        	width : "8%",
        	template : '#= kendo.toString(new Date(Date.parse(date)),"yyyy/MM/dd hh:mm tt") #'
        }, {        	
        	field : "isSystem",
        	title : "Source",
            width : "7%",          
        	template : kendo.template($("#IsSystemTemplate").html())                 
        }, {        	
            title : " ",
            width : '4%',
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }],
//        columnResizeHandleWidth : 6,        
        columnResizeHandleWidth : 5,
        /*
        dataBound : function(e) {
            commonDataBoundEvent(target, msg);
        }
        */
    };
}


function getDisplayedValue(value) {
	if ((typeof value !== 'undefined') && value) {
		return value;
	}
	else {
		return "";
	}
}


function getDisplayedBedFileValue(value) {
	if ((typeof value !== 'undefined') && value) {
	    var lastIndex = value.lastIndexOf("/") + 1;
		return value.substr(lastIndex);
    }
    else {
    	return "";
    }
}

function onDataBinding(arg) {
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    $('body').prepend(busyDiv);

}

function onDataBound(arg) {
    $('body').css("cursor", "default");
    $('.myBusyDiv').empty();
    $('body').remove('.myBusyDiv');
    
    var source = "#tab_contents";
    bindActions(source);
}


function commonDataBoundEvent(target, msg) {
    $(target).addClass('plan-table');
    $(target).parent().children('div.k-pager-wrap').show();
    if ($(target).data("kendoGrid").dataSource.data().length === 0) {
        var encodingTemplate = kendo.template($("#emptyRowTemplate").html());
        $(target + ' tbody').html(encodingTemplate({
            msg : msg
        }));
        $(target).parent().children('div.k-pager-wrap').hide();
    }
    bindActions(target);
}

function bindActions(source) {
    $(source + ' .review-plan').click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');

        $('body #modal_review_plan').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_review_plan").modal("show");

            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");

            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });
    $(source + ' .plan-run').click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');

        $('body #modal_plan_wizard').remove();
        $.get(url, function(data) {
            $('body').append(data);
            setTab('#ws-8');
            $("#modal_plan_wizard").data('source', source);
            $("#modal_plan_wizard").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
            // $(that).trigger('remove_from_project_done', {values: e.values});
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });

    $(source + " .edit-plan").click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');

        $('body #modal_plan_wizard').remove();
        $.get(url, function(data) {
            $('body').append(data);

            setTab('#ws-1');
            $("#modal_plan_wizard").data('source', source);
            $("#modal_plan_wizard").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
            // $(that).trigger('remove_from_project_done', {values: e.values});
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });
    $(source + " .copy-plan").click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        $('body #modal_plan_wizard').remove();
        $.get(url, function(data) {
            $('body').append(data);
            setTab('#ws-8');
            $("#modal_plan_wizard").data('source', source);
            $("#modal_plan_wizard").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
        return false;
    });

    $(source + " .delete-plan").click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        
        url = $(this).attr('href');
        $('body #modal_confirm_delete').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_confirm_delete").data('source', source);
            $("#modal_confirm_delete").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);
        });
    });

    $(source + ' .batch-plan').click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();

        url = $(this).attr('href');
        $('body #modal_batch_planning').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $('#modal_batch_planning').modal("show");
            return false;
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);
        });
    });
}


$(document).bind('modal_confirm_delete_done modal_plan_wizard_done', function(e) {
    var target = $(e.target).data('source');
    
    //target is #tab_contents
    console.log("modal_confirm_delete_done modal_plan_wizard_done target=", target);
    
    var selectedTab = window.location.hash.substring(1);
    
    console.log("selectedTab=", selectedTab);
    var id = "#" + selectedTab;
    console.log("id=", id);
    refreshKendoGrid(id);
});


$(function () {
 function switch_tab() {
    var selectedTab = window.location.hash.substring(1);
    
    console.log("selectedTab=", selectedTab);
    
    if (selectedTab == '')
    	selectedTab = 'favorites';
    $("#left_side_nav > li").removeClass("active");
    $("#"+selectedTab+"_nav").addClass("active");
    $("#tab_contents > div").hide();
    $("#"+selectedTab+"_tab").show();
    
    var basePlannedExperimentUrl = "/rundb/api/v1/plantemplatebasicinfo/?format=json&planExecuted=False&isSystemDefault=False";
    var orderByOptions = "&order_by=-date&order_by=planDisplayedName";
    
    var grid = null;
    if (selectedTab == 'favorites') {    
    	var isOCPEnabled = $('input[id=isOCPEnabled]').val();
    	
    	if (isOCPEnabled == "True") {    		
    		var favorites = $("#favorites").kendoGrid(commonKendoGrid("#favorites",
    	        basePlannedExperimentUrl + "&isFavorite=true&categories__in=Onconet,Oncomine,," + orderByOptions,
    	        'No Favorites yet'));
        	grid = favorites;
    	}
    	else {
            var favorites = $("#favorites").kendoGrid(commonKendoGrid("#favorites",
        	        basePlannedExperimentUrl + "&isFavorite=true&categories__in=Onconet,," + orderByOptions,
        	        'No Favorites yet'));    	
        	grid = favorites;    		
    	} 
    }
    
    if (selectedTab == 'recents') {
    	var isOCPEnabled = $('input[id=isOCPEnabled]').val();
    	
    	if (isOCPEnabled == "True") {    		
    		var recents = $("#recents").kendoGrid(commonKendoGrid("#recents",
    	        basePlannedExperimentUrl + "&categories__in=Onconet,Oncomine,," + orderByOptions,
    	        'No Recents yet'));
        	grid = recents;
    	}
    	else {
            var recents = $("#recents").kendoGrid(commonKendoGrid("#recents",
        	        basePlannedExperimentUrl + "&categories__in=Onconet,," + orderByOptions,
        	        'No Recents yet'));    	
        	grid = recents;    		
    	}    	
 	}
    
    if (selectedTab == 'ampliseq_dna') {
    	var isOCPEnabled = $('input[id=isOCPEnabled]').val();
    	
    	if (isOCPEnabled == "True") {    		
    		var ampliSeqs = $("#ampliseq_dna").kendoGrid(commonKendoGrid("#ampliseq_dna",
    	        basePlannedExperimentUrl + "&runType__in=AMPS,AMPS_EXOME&applicationGroup__name__iexact=DNA&categories__in=Onconet,Oncomine,," + orderByOptions,
    	        'No Ampliseq DNA templates yet'));
        	grid = ampliSeqs;
    	}
    	else {
            var ampliSeqs = $("#ampliseq_dna").kendoGrid(commonKendoGrid("#ampliseq_dna",
        	        basePlannedExperimentUrl + "&runType__in=AMPS,AMPS_EXOME&applicationGroup__name__iexact=DNA&categories__in=Onconet,," + orderByOptions,
        	        'No Ampliseq DNA templates yet'));    	
        	grid = ampliSeqs;    		
    	}
    }
    
    if (selectedTab == 'ampliseq_rna') {
      var ampliSeqRna = $("#ampliseq_rna").kendoGrid(commonKendoGrid("#ampliseq_rna",
            basePlannedExperimentUrl + "&runType=AMPS_RNA&applicationGroup__name__iexact=RNA" + orderByOptions,
            'No Ampliseq RNA templates yet'));
      grid = ampliSeqRna;
    }

    if (selectedTab == 'genericseq') {
    	var genericSeqs = $("#genericseq").kendoGrid(commonKendoGrid("#genericseq",
            basePlannedExperimentUrl + "&runType=GENS" + orderByOptions,
            'No Generic Sequencing templates yet'));  
    	grid = genericSeqs;
    }

    if (selectedTab == 'pharmacogenomics') {
      var pharmacogenomics = $("#pharmacogenomics").kendoGrid(commonKendoGrid("#pharmacogenomics",
    	        basePlannedExperimentUrl + "&runType=AMPS&applicationGroup__name__iexact=PGx" + orderByOptions,
    	        'No Pharmacogenomics templates yet'));
      grid = pharmacogenomics;
    }
    
    if (selectedTab == 'rna_seq') {
      var rnaSeqs = $("#rna_seq").kendoGrid(commonKendoGrid("#rna_seq",
    	        basePlannedExperimentUrl + "&runType=RNA" + orderByOptions,
    	        'No RNASeq templates yet'));
      grid = rnaSeqs;
    }

    if (selectedTab == 'targetseq') {
      var targetSeqs = $("#targetseq").kendoGrid(commonKendoGrid("#targetseq",
    	        basePlannedExperimentUrl + "&runType=TARS" + orderByOptions,
    	        'No TargetSeq templates yet'));
      grid = targetSeqs;
    }

    if (selectedTab == 'whole_genome') {
      var wholeGenomes = $("#whole_genome").kendoGrid(commonKendoGrid("#whole_genome",
    	        basePlannedExperimentUrl + "&runType=WGNM" + orderByOptions,
    	        'No Whole Genome templates yet'));
      grid = wholeGenomes;
    }

    if (selectedTab == '16s_targetseq') {
      var targetSeq_16s = $("#16s_targetseq").kendoGrid(commonKendoGrid("#16s_targetseq",
    	        basePlannedExperimentUrl + "&runType=TARS_16S" + orderByOptions,
    	        'No 16S Target Sequencing templates yet'));
      grid = targetSeq_16s;
    }

    if (selectedTab == 'fusions') {
    	var isOCPEnabled = $('input[id=isOCPEnabled]').val();
    	console.log("isOCPEnabled=", isOCPEnabled);
    	
    	if (isOCPEnabled == "True") {    		
    		var fusions = $("#fusions").kendoGrid(commonKendoGrid("#fusions",
    	        basePlannedExperimentUrl + "&runType__in=AMPS,AMPS_EXOME,AMPS_RNA,AMPS_DNA_RNA&applicationGroup__uid__iexact=APPLGROUP_0005&categories__in=Onconet,Oncomine,," + orderByOptions,
    	        'No DNA + RNA templates yet'));
        	grid = fusions;
    	}
    	else {
            var fusionss = $("#fusions").kendoGrid(commonKendoGrid("#fusions",
        	        basePlannedExperimentUrl + "&runType__in=AMPS,AMPS_EXOME,AMPS_RNA,AMPS_DNA_RNA&applicationGroup__uid__iexact=APPLGROUP_0005&categories__in=Onconet,," + orderByOptions,
        	        'No DNA + RNA templates yet'));    	
        	grid = fusions;    		
    	}
    }
    
    if (grid) {
      //grid.refresh();
      //a is undefined
      var a = grid.dataSource;
      console.log("grid.dataSource=", a);
      
      //grid.setDataSource(a);
    }
  };
  
  window.onhashchange = function(e) { 
	  switch_tab(); 
  };
  
  switch_tab();
});


$(document).ready(function() {
    $('.add-new-plan').click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        source = $(this).attr('ref');
        $('body #modal_plan_wizard').remove();
        $.get(url, function(data) {
            $('body').append(data);

            setTab('#ws-1');
            $("#modal_plan_wizard").data('source', source);
            $("#modal_plan_wizard").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
            // $(that).trigger('remove_from_project_done', {values: e.values});
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });
    $('.add-new-plan-run').click(function(e) {
        //browser bug: mouse cursor will not change if user has not moved the mouse.
        $('body').css("cursor", "wait");

        e.preventDefault();
        $('#error-messages').hide().empty();

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        source = $(this).attr('ref');
        $('body #modal_plan_wizard').remove();
        $.get(url, function(data) {
            $('body').append(data);
            setTab('#ws-1');
            $("#modal_plan_wizard").data('source', source);
            $("#modal_plan_wizard").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
            // $(that).trigger('remove_from_project_done', {values: e.values});
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });

    $('.upload-plan').click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        url = $(this).attr('href');

        $('body #modal_batch_planning_upload').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $('#modal_batch_planning_upload').modal("show");
            return false;
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);
        });
    });
});
