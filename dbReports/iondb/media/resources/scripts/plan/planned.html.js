//models.PlannedExperiment.objects.filter(isReusable=False, planExecuted=False).order_by("-date", "planName")

function onDataBinding(arg) {
	//20130707-TODO - does not work!!
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    $('body').prepend(busyDiv);

}

function onDataBound(arg) {
    $('body').css("cursor", "default");
    $('.myBusyDiv').empty();
    $('body').remove('.myBusyDiv');

    var source = "#grid";
    $(source + ' span[rel="popover"]').each(function(i, elem) {
        $(elem).popover({
            content : $($(elem).data('select')).html()
        });
    });

    $(source + ' .selectall').click(function(e) {
        // e.preventDefault();
        var state = $(this).is(':checked');
        $(source + ' table tbody tr td input[type="checkbox"]').each(function(i, j) {
            $(this).attr('checked', state);
            id = $(this).attr("id");
            if (state)
                checked_ids.push(id);
            else
                checked_ids.splice(checked_ids.indexOf(id), 1);
        });
    });

    $("#grid :checkbox.selected").each(function() {
        if ($.inArray($(this).attr("id").toString(), checked_ids) > -1) {
            $(this).attr('checked', true);
        }
    });
    $(source + ' :checkbox.selected').change(function() {
        id = $(this).attr("id");
        if ($(this).attr("checked"))
            checked_ids.push(id);
        else
            checked_ids.splice(checked_ids.indexOf(id), 1);
    });

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

    $(source + " .delete-plan").click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        url = $(this).attr('href');
        $('body #modal_confirm_delete').remove();
        $('modal_confirm_delete_done');
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

            }).always(function(data) {/*console.log("complete:", data);*/
            });
        });
	$(source + " .transfer_plan").click(function(e) {
        e.preventDefault();
        url = $(this).attr('href');
        $('body #modal_plan_transfer').remove();
        $.get(url, function(data) {
            $('body').append(data);
                $("#modal_plan_transfer").data('source', source);
                $("#modal_plan_transfer").modal("show");
                return false;
            }).done(function(data) {
                console.log("success:", url);
            }).fail(function(data) {
                $('#error-messages').empty().show();
                $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
                console.log("error:", data);
            });
        });
}

$(document).ready(function() {
    //20130711-moved-to-be-global var checked_ids = [];
    var grid = $("#grid").kendoGrid({
        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : "/rundb/api/v1/plannedexperiment/?isReusable=False&planExecuted=False",
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
                            type : "number"
                        },
                        planShortID : {
                            type : "string"
                        },
                        planDisplayedName : {
                            type : "string"
                        },
                        barcodeId : {
                            type : "string"
                        },
                        library : {
                            type : "string"
                        },
                        runType : {
                            type : "string"
                        },
                        projects : {
                            type : "string"
                        },
                        date : {
                            type : "string"
                        },
                        planStatus : {
                            type : "string"
                        },
                        sampleSetDisplayedName : {
                        	type : "string"
                        },
                        sampleGroupingName : {
                        	type : "string"
                        },
                         libraryPrepType : {
                            type : "string"
                        },
                        combinedLibraryTubeLabel : {
                            type : "string"
                        },
                        sampleTubeLabel : {
                            type : "string"
                        },
                        chipBarcode : {
                            type : "string"
                        },                        
                    }
                }
            },
            serverFiltering: true,
            serverSorting : true,
            sort : [{
                field : "date",
                dir : "desc"
            }, {
                field : "planDisplayedName",
                dir : "asc"
            }],
            serverPaging : true,
            pageSize : 10
        },
        groupable : false,
        scrollable : false,
        selectable : false,
        sortable : true,
        pageable : {
            messages: {
                display: "{0} - {1} of {2} planned runs",
                empty: "No planned runs to display"
            }
        },
		dataBinding : onDataBinding,
		dataBound : onDataBound,
		
        columns : [{
            field : "id",
            title : "",
            sortable : false,
            width: '25px',
            headerTemplate: "<span rel='tooltip' title='(De)select All'><input  class='selectall' type='checkbox'></span>",
            template : "<input id='${id}' name='runs' type='checkbox' class='selected'>"
        }, {
            field : "sampleSetDisplayedName",
            title : "Sample Set",
            width: '10%',
            sortable : false,
            template : function(item){
                          var data = { id: item.id, label: "Sample Sets", values: item.sampleSetDisplayedName.split(',') };
                          return kendo.template($("#PopoverColumnTemplate").html())(data);
                       }
        }, {
            field : "planShortID",
            title : "Run Code",
            width: '70px',
            // template: "<a href='/data/project/${id}/results'>${name}</a>"
            template :kendo.template($('#PlanShortIdColumnTemplate').html())
        }, {
            field : "planDisplayedName",
            title : "Planned Run Name",
            width: '25%',
            sortable : true
        }, {
            field : "barcodeId",
            title : "Barcodes",
            width: '15%',
            sortable : false
        }, {
            field : "library",
            title : "Reference",
            width: '10%',
            sortable : false
        }, {
            field : "runType",
            title : "Res App",
            sortable : true,
            width: '32px',
            template : kendo.template($('#RunTypeColumnTemplate').html())
        }, {
            field : "sampleGroupingName",
        	title : "Group",
            width: '10%',
        	sortable : true
        }, {
            field : "libraryPrepType",
        	title : "Library Prep Type",
        	width: '75px',
        	sortable : false,
        	template : kendo.template($('#LibTypeColumnTemplate').html())
        }, {
            field : "combinedLibraryTubeLabel",
			width: '80px',
        	title : "Combined Library Tube Label",
        	sortable : false
        },{    
            field : "projects",
            title : "Project",
            width: '10%',
            sortable : false,
            template : function(item){
                          var data = { id: item.id, label: "Projects", values: item.projects.split(',') };
                          return kendo.template($("#PopoverColumnTemplate").html())(data);
                       }
        }, {
            title : "Sample",
            sortable : false,
            width: '10%',
            template : kendo.template($('#SampleColumnTemplate').html())
        }, {
            field : "sampleTubeLabel",
            title : "Sample Tube Label",
            width: '10%',
            sortable : false,
        }, {
            field : "chipBarcode",
            title : "Chip Barcode",
            width: '10%',
            sortable : false,
        }, {
            field : "date",
            title : "Last Modified",
            width : '9%',
            template : '#= kendo.toString(new Date(Date._parse(date)),"MMM d yyyy") #'
        }, {
            field : "planStatus",
            title : "Status",
            width: '70px',
            sortable : true,
            template : '<span style="text-transform: capitalize;">#=planStatus#</span>'
        }, {        	
            title : " ",
            width : '36px',
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }],
    });

    switch_to_view(window.location.hash.replace('#',''));

    var today = Date.parse('today');
    $('#dateRange').daterangepicker({
        dateFormat: 'M d yy',
        presetRanges: [
            {text: 'Today', dateStart: today, dateEnd: today},
            {text: 'Last 7 Days', dateStart: 'today-7days', dateEnd: today},
            {text: 'Last 30 Days', dateStart: 'today-30days', dateEnd: today},
            {text: 'Last 60 Days', dateStart: 'today-60days', dateEnd: today},
            {text: 'Last 90 Days', dateStart: 'today-90days', dateEnd: today}
        ],
    });
    $('.search_trigger').click(function (e) { filter(e); });
    $('#search_text').keypress(function(e){ if (e.which == 13 || e.keyCode == 13) filter(e); });
    $('#dateRange, .selectpicker').change(function (e) { filter(e); });

    $('#clear_filters').click(function () { console.log("going to reload!!"); window.location.reload(true); });
    
    $('#plan_search_dropdown_menu a').click(function(e) {
        var span = $(this).find('span');
        $('#search_text').data('selected_filter', $(this).data('filter'));

        $('#plan_search_dropdown_menu span').each(function(){
            $(this).removeClass("icon-white icon-check");
            if (this == span.get(0)){
                $(this).addClass("icon-check");
            } else{
                $(this).addClass("icon-white");
            }
        });
        
        $("#search_subject_nav").attr("title", "Search by " + this.text);
        $("#search_text").attr("placeholder", "Search by " + this.text);
    });


    $('.delete_selected').click(function(e) {
        e.preventDefault();
        e.stopPropagation();
        $('#error-messages').hide().empty();
        $('body #modal_confirm_delete').remove();
        var checked_ids = $("#grid input:checked").map(function() {
            return $(this).attr("id");
        }).get();
        console.log(checked_ids);

        if (checked_ids.length > 0) {
            url = $(this).attr('href').replace('0', checked_ids.join());
            // alert(url);
            $.get(url, function(data) {
                return false;
            }).done(function(data) {
                console.log("success:", url);
                $('body').append(data);
                $('#modal_confirm_delete').modal("show");
            }).fail(function(data) {
                $('#error-messages').empty().show();
                $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
                console.log("error:", data);

            }).always(function(data) {/*console.log("complete:", data);*/
            });
        }

    });
    $('.clear_selection').click(function(e) {
        checked_ids = [];
        $("#grid input:checked").attr('checked', false);
    });
    
    $('.setview').click(function(e){
        switch_to_view(this.id);
    });
    
    $(document).bind('modal_confirm_delete_done modal_plan_wizard_done modal_plan_transfer_done', function(e) {
        console.log(e.target, e.relatedTarget);
        refreshKendoGrid('#grid');
    });

    //Force plan link. Used to force plans from pending -> planned which is normally done by chef
    $("#grid").on("click", ".force-planned", function (event) {
        var confirmText = "Are you sure?";
        event.preventDefault();
        if (confirm(confirmText)) {
            //Show busy div
            var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
            $('body').prepend(busyDiv);
            $.ajax({
                url: "/rundb/api/v1/plannedexperiment/" + $(this).data("id") + "/",
                type : 'PATCH',
                data: JSON.stringify({planStatus: "planned"}),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
        	}).fail(function(data) {
            	$('body').remove('.myBusyDiv');

            	$('#error-messages').empty().show();
            	$('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            	console.log("error:", data);

            }).always(function () {
                $('body').remove('.myBusyDiv');
                refreshKendoGrid('#grid');
            });
        }
    });

});

function switch_to_view(view){
    //var view = window.location.hash;
    if (view.length==0) view = 'all';
    console.log('switching view to', view);
    
    var data = $("#grid").data("kendoGrid");
    var base_url = data.dataSource.transport.options.read.url.split('&sampleSet')[0];
    
    $('.view-toggle').children().removeClass('active');
    $('#'+view).addClass('active');

    // hide/show sampleSet columns
    if(view=='bySample'){
        data.hideColumn('id');
        data.hideColumn('projects');
        data.showColumn('sampleSetDisplayedName');
        data.showColumn('sampleGroupingName');
    } else {
        data.showColumn('id');
        data.showColumn('projects');
        data.hideColumn('sampleSetDisplayedName');
        data.hideColumn('sampleGroupingName');
    }
    
    // update dataSource url
    if(view=='byTemplate'){
        base_url += '&sampleSets__isnull=True'
    } else if(view=='bySample'){
        base_url += '&sampleSets__isnull=False'
    }
    data.dataSource.transport.options.read.url = base_url;
    data.dataSource.read();
}


function _get_query_string(val){
    val = val || "";
    if ($.isArray(val)){
        return val.join(",")
    }
    return val;
}

function filter(e){
    e.preventDefault();
    e.stopPropagation();

    var date = "";
    var daterange = $("#dateRange").data("daterange");
    if (daterange) {
        var start = daterange.start.toString('yyyy-MM-dd HH:mm');
        var end = daterange.end.toString('yyyy-MM-dd HH:mm').replace('00:00', '23:59');
        date = start + ',' + end;
    }

    var filters = [
        {
            field: $('#search_text').data('selected_filter'),
            operator: "__icontains",
            value: $("#search_text").val().trim().replace(/ /g, '_')
        },
        {
            field: "date",
            operator: "__range",
            value: date
        },
        {
            field: "planStatus",
            operator: "",
            value: $("#id_status").val()
        },
        {
            field: "runType",
            operator: "",
            value: $("#id_runtype").val()
        },
        {
            field: "projects__name",
            operator: "__in",
            value: _get_query_string($("#id_project").val())
        },
        {
            field: "experiment__eas_set__barcodeKitName",
            operator: "__in",
            value: _get_query_string($("#id_barcodes").val())
        },
        {
            field: "experiment__eas_set__reference",
            operator: "__in",
            value: _get_query_string($("#id_reference").val())
        }
    ]
    $("#grid").data("kendoGrid").dataSource.filter(filters);
}
