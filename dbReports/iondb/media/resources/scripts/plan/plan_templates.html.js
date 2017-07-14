function commonKendoGrid(target, url, msg) {
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
                        applicationCategoryDisplayedName : {
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
            serverSorting : true,
            sort: { field: "date", dir: "desc" }
        },
        sortable: true,
        height : '460',
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
        	title : "Instr",
            width : "5%",
            sortable : false,
        	template : kendo.template($("#SeqInstrumentTemplate").html())
        }, {
        	field : "templatePrepInstrumentType",
        	title : "OT/IC",
            width : "5%",
            sortable : false,
        	template : kendo.template($("#TemplatePrepInstrumentTemplate").html())
         }, {        	
        	field : "runType",
        	title: "App",
            width : "5%",
            sortable : false,
        	template : kendo.template($("#RunTypeColumnTemplate").html())         	
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
            sortable : false,
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
            width : "55px",
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
    
    $(source + ' .toggle-template-favorite').click(function(e) {
        e.preventDefault();
        var url = $(this).attr('href');
        $.get(url, function(data) {
            window.location.reload();
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
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
    
    if (selectedTab == '')
    	selectedTab = 'favorites';
    $("#left_side_nav > li").removeClass("active");
    $("#"+selectedTab+"_nav").addClass("active");
    $("#tab_contents > div").hide();
    $("#"+selectedTab+"_tab").show();

    var basePlannedExperimentUrl = "/rundb/api/v1/plantemplatebasicinfo/";

    var $selectedNav = $("#"+selectedTab+"_nav")
    var grid = $("#"+selectedTab).kendoGrid( commonKendoGrid("#"+selectedTab,
                basePlannedExperimentUrl + "?format=json" + $selectedNav.data('api_filter'),
                "No" + $selectedNav.text() + "Templates") );

    // show warnings if missing files
    var checkFilesUrl = basePlannedExperimentUrl + "check_files" + "?format=json" + $selectedNav.data('api_filter') + "&application="+selectedTab;
    var $warnings = $("#"+selectedTab +"_tab .template_warnings");
    var install_url = "/plan/plan_templates/install_files/";
    $warnings.empty().hide();

    $.get(checkFilesUrl, function(files){
        if (selectedTab == "recently_created")
            // skip if viewing "All" tab
            return

        var install_btn = "";
        if (files.install_lock){
            install_btn = '<a href="#" class="btn install_files" style="margin-left:10px;" disabled>Installing ...</a>';
        } else if (files.files_available) {
            install_btn = '<a href="'+install_url+'" class="btn install_files" style="margin-left:10px;">Install</a>';
        }

        if (files.references.length > 0){
            $warnings.append('References are not installed: ');
            $warnings.append(install_btn);
            $warnings.append("<ul><li>" + files.references.join('</li><li>') + '</li></ul></div>');
            $warnings.show();
        }
        if (files.bedfiles.length > 0){
            $warnings.append("BED files are not installed:");
            if (files.references.length == 0) $warnings.append(install_btn);
            $warnings.append("<ul><li>" + files.bedfiles.join('</li><li>') + '</li></ul></div>');
            $warnings.show();
        }
        $warnings.children('.btn').off('click').on('click', function(e){
            e.preventDefault();
            if($(this).attr('disabled') == 'disabled')
                return false;
            
            $('#error-messages').hide().empty();
            $('body #modal_upload_and_install_files').remove();
            $.get(install_url, files, function(data){
                $('body').append(data);
                $('#modal_upload_and_install_files').modal("show");
                return false;
            });
        });
    });
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
    
    $('#upload_template').click(function(e) {
        e.preventDefault();
        $('#modal_load_template .alert-error').empty().hide();

        var filename = $('#modal_load_template :file').val();
        if (!filename){
            $('#modal_load_template .alert-error').empty().append('Please select a CSV file').show();
            return false;
        }

        $('#importTemplateForm').ajaxSubmit({
            dataType : 'json',
            async: false,
            beforeSubmit: $.blockUI({baseZ: 2000}),
            success: function(data) {
                if (data){
                    var error = "<span>"+data.status_msg+"</span>";
                    for (var key in data.msg) {
                        error += "<br><ul class='unstyled'>";
                        error += "<li><strong>" + key + " contained error(s):</strong> ";
                        error += "<ul>";
                        $.each(data.msg[key], function(i,msg){
                            error += "<li>" + msg +"</li>";
                        });
                        error += "</ul>";
                        error += "</li>";
                        error += "</ul>";
                    }
                    if (data.status == 'failed'){
                        $('#modal_load_template .alert-error').html(error).show();
                    } else {
                        $('#modal_load_template .alert-success').show();
                        $('#modal_load_template .alert-warning').html(error).show();
                        $('#modal_load_template .btn').hide();
                        $('#modal_load_template #close_on_success').show();
                    }
                } else {
                    $('#modal_load_template .alert-success').show();
                    setTimeout(function(){
                        $('#modal_load_template').modal('hide');
                      }, 1000);
                }
                $.unblockUI();
            },
            error: function(data){
                $('#modal_load_template .alert-error').empty().append(data.responseText).show();
                $.unblockUI();
            }
        });
        return false;
    });

    $('#modal_load_template').on('hide', function(){
        if ( $('#modal_load_template .alert-success').is(':visible')){
            // show new template
            window.location.hash = 'recently_created';
            if ($('#recently_created').data('kendoGrid')){
                $('#recently_created').data('kendoGrid').refresh();
            }
        }
        $('#importTemplateForm')[0].reset();
        $('#modal_load_template .alert').hide();
        $('#modal_load_template .btn').show();
        $('#modal_load_template #close_on_success').hide();
    });
});
