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
            serverFiltering: true,
            serverPaging : true,
            pageSize : 10,
            serverSorting : true,
            sort: { field: "date", dir: "desc" }
        },
        sortable: true,
        scrollable : false,
        pageable : {
            messages: {
                display: gettext('template.pageable.messages.display'), //"{0} - {1} of {2} templates"
                empty: gettext('template.pageable.messages.empty'), //"No templates to display"
            }
        },
		dataBinding : onDataBinding,
		dataBound : onDataBound,
        columns : [{
            field : "planDisplayedName",
            title : gettext('template.fields.planDisplayedName.label'), //"Template Name"
            width : "20%",
            template : kendo.template($("#PlanDisplayedNameTemplate").html())
        }, {
        	field : "sequencingInstrumentType",
        	title : gettext('template.fields.sequencingInstrumentType.label'), //"Instr",
            width : "40px",
            sortable : false,
        	template : kendo.template($("#SeqInstrumentTemplate").html())
        }, {
        	field : "templatePrepInstrumentType",
        	title : gettext('template.fields.templatePrepInstrumentType.label'), //"Sample Prep",
            width : "55px",
            sortable : false,
        	template : kendo.template($("#TemplatePrepInstrumentTemplate").html())
         }, {
        	field : "runType",
        	title: gettext('template.fields.runType.label'), //"Res App",
            width : "37px",
            sortable : false,
        	template : kendo.template($("#RunTypeColumnTemplate").html())
        }, {
        	field : "barcodeKitName",
        	title: gettext('template.fields.barcodeKitName.label'), //"Barcodes",
            width : "10%",
        	template : kendo.template($("#BarcodeKitNameTemplate").html())
        }, {
        	field : "reference",
        	title : gettext('template.fields.reference.label'), //"Reference",
            width : "15%",
        	template : kendo.template($("#ReferenceTemplate").html())
        }, {
            field : "projects",
            title : gettext('template.fields.projects.label'), //"Project",
            width: '10%',
            sortable : false,
            template : function(item){
                            var data = { id: item.id, label: gettext('template.fields.projects.label.plural'), values: item.projects.split(',') };
                            return kendo.template($("#PopoverColumnTemplate").html())(data);
                        }
        }, {
            field : "irAccountName",
            title : gettext('template.fields.irAccountName.label'), //"Ion Reporter Account",
            sortable : false,
            width : "10%",
            hidden: true,
        }, {
        	field : "irworkflow",
        	title : gettext('template.fields.irworkflow.label'), //"Ion Reporter Workflow",
            width : "13%",
            hidden: true,
        }, {
        	field : "date",
        	title : gettext('template.fields.date.label'), //"Date",
        	width : "9%",
        	template : '#= kendo.toString(new Date(Date.parse(date)),"MMM d yyyy") #'
        }, {
        	field : "isSystem",
        	title : gettext('template.fields.isSystem.label'), //"Source",
            width : "55px",
        	template : kendo.template($("#IsSystemTemplate").html())
        }, {
            title : " ",
            width : "36px",
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }],
//        columnResizeHandleWidth : 6,        
        columnResizeHandleWidth : 5,
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
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    $('body').prepend(busyDiv);

}

function onDataBound(arg) {
    $('body').css("cursor", "default");
    $('.myBusyDiv').empty();
    $('body').remove('.myBusyDiv');

    var source = "#tab_contents";
    bindActions(source);

    $(source + ' span[rel="popover"]').each(function(i, elem) {
        $(elem).popover({
            content : $($(elem).data('select')).html()
        });
    });
}

function onFailure(data) {
    $('#error-messages').empty().show();
    $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
    console.log("error:", data);
}

function bindActions(source) {
    $(source + ' .review-plan').click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
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

            onFailure(data);
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

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
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

            onFailure(data);
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

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
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

            onFailure(data);
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

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
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

            onFailure(data);
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
            onFailure(data);
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
            onFailure(data);
        });
    });

    $(source + ' .toggle-template-favorite').click(function(e) {
        e.preventDefault();
        var url = $(this).attr('href');
        $.get(url, function(data) {
            window.location.reload();
        }).fail(function(data) {
            onFailure(data);
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
 function switch_tab(kendo_customize_callback) {
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
    if (jQuery.isFunction(kendo_customize_callback)) {
        kendo_customize_callback(grid);
    }
    // start with all filters reset
    clear_filters();
    set_more_filters();

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
            $warnings.append(gettext('templates.messages.references.notinstalled') + ':');
            $warnings.append(install_btn);
            $warnings.append("<ul><li>" + files.references.join('</li><li>') + '</li></ul></div>');
            $warnings.show();
        }
        if (files.bedfiles.length > 0){
            $warnings.append(gettext('templates.messages.bedfiles.notinstalled') + ':');
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
	  switch_tab(kendo_customize_callback);
  };

  if (typeof kendo_customize_callback == 'undefined' || !jQuery.isFunction(kendo_customize_callback)) {
    kendo_customize_callback = function($grid) {};
  }
  switch_tab(kendo_customize_callback);

});


$(document).ready(function() {
    $('.add-new-plan').click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
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

            onFailure(data);
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

        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
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

            onFailure(data);
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
            onFailure(data);
        });
    });

    $('#upload_template').click(function(e) {
        e.preventDefault();
        $('#modal_load_template .alert-error').empty().hide();

        var filename = $('#modal_load_template :file').val();
        if (!filename){
            $('#modal_load_template .alert-error').empty().append(gettext('import_plan_template.messages.validate.file.required')).show(); //Please select a CSV file
            return false;
        }

        $('#importTemplateForm').ajaxSubmit({
            dataType : 'json',
            async: false,
            beforeSubmit: $.blockUI({baseZ: 2000}),
            success: function(data) {
                if (data){
                    var _html = kendo.template($('#ImportPlanTemplateResponseTemplate').html())(data);
                    if (data.status == 'failed') {
                        $('#modal_load_template .alert-error').html(_html).show();
                    } else {
                        $('#modal_load_template .alert-success').show();
                        $('#modal_load_template .alert-warning').html(_html).show();
                        $('#modal_load_template .modal-footer .btn.btn-primary').hide();
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
    $('#modal_load_template_json_bundle').on('hide', function(){
        $('#file_info').html('').hide();
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
        $('#modal_load_template .modal-footer .btn.btn-primary').show();
        $('#modal_load_template #close_on_success').hide();
    });

    $('[name=dateRange]').each(function(){ $(this).daterangepicker($.DateRangePickerSettings.get()); });
    $('.search_trigger').click(function (e) { filter(e); });
    $('[name=search_text]').keypress(function(e){ if (e.which == 13 || e.keyCode == 13) filter(e); });
    $('[name=dateRange], .selectpicker').change(function (e) { filter(e); });
    $('.clear_filters').click(function () { clear_filters(); });
    $('.toggle_more_filters').click(function () { set_more_filters(true); });

    $('[name=plan_search_dropdown_menu] a').click(function(e) {
        var span = $(this).find('span');
        var container = '#' + (window.location.hash.substring(1) || 'favorites') + '_tab';

        $(container + ' [name=search_text]').data('selected_filter', $(this).data('filter'));
        $(container + ' [name=plan_search_dropdown_menu] span').each(function(){
            $(this).removeClass("icon-white icon-check");
            if (this == span.get(0)){
                $(this).addClass("icon-check");
            } else{
                $(this).addClass("icon-white");
            }
        });


        var x = $(container + ' [name=search_subject_nav]');
        x.attr("title", x.data('titlePrefix') + this.text);
        var x = $(container + ' [name=search_text]');
        x.attr("placeholder", x.data('placeholderPrefix') + this.text);
    });
});

// Search bar functions
function clear_filters(){
    var container = '#' + (window.location.hash.substring(1) || 'favorites') + '_tab';
    $(container + ' .search-field :input').val('');
    $(container + ' .search-field .selectpicker').selectpicker('val','');
    $(container + " .list_contents").data("kendoGrid").dataSource.filter([]);
}

function set_more_filters(toggle){
    var more_filters = localStorage.getItem('templates-more-filters') == "true";
    if (toggle){
        more_filters = !more_filters;
    }
    var container = '#' + (window.location.hash.substring(1) || 'favorites') + '_tab';
    var toggle_button = $(container + " .toggle_more_filters");
    var optional_filters = $(container + " .search-field.optional");
	if (!more_filters) {
        toggle_button.text(toggle_button.data("more-text"));
        optional_filters.hide();
    } else {
        toggle_button.text(toggle_button.data("less-text"));
        optional_filters.show();
    }
    localStorage.setItem('templates-more-filters', more_filters);
}

function _get_query_string(val){
    val = val || "";
    if ($.isArray(val)){
        return val.join(",")
    }
    return val;
}

function _daterange_to_filter(val){
    if (!val) return "";
    var range = val.split('-');
    var start = new Date(range[0]).toString('yyyy-MM-dd HH:mm');
    var end = range.length > 1 ? new Date(range[1]).toString('yyyy-MM-dd HH:mm') : start;
    return start + ',' + end.replace('00:00', '23:59');
}

function filter(e){
    e.preventDefault();
    e.stopPropagation();

    var container = '#' + (window.location.hash.substring(1) || 'favorites') + '_tab';
    var filters = [
        {
            field: $(container + ' [name=search_text]').data('selected_filter'),
            operator: "__icontains",
            value: $(container + ' [name=search_text]').val().trim().replace(/ /g, '_')
        },
        {
            field: "date",
            operator: "__range",
            value: _daterange_to_filter($(container + " [name=dateRange]").val())
        },
        {
            field: "platform",
            operator: "",
            value: _get_query_string($(container + ' [name=instrument]').val())
        },
        {
            field: "sampleprep",
            operator: "",
            value: _get_query_string($(container + ' [name=sampleprep]').val())
        },
        {
            field: "projects__name",
            operator: "__in",
            value: _get_query_string($(container + " [name=project]").val())
        },
        {
            field: "experiment__eas_set__barcodeKitName",
            operator: "__in",
            value: _get_query_string($(container +" [name=barcodes]").val())
        },
        {
            field: "experiment__eas_set__reference",
            operator: "__in",
            value: _get_query_string($(container +" [name=reference]").val())
        },
        {
            field: "user",
            operator: "",
            value: _get_query_string($(container + ' [name=user]').val())
        },
    ]
    $(container + " .list_contents").data("kendoGrid").dataSource.filter(filters);
}
