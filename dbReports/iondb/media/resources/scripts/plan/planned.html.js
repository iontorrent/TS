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
    $(source + ' table thead th:first').html("<span rel='tooltip' title='(De)select All'><input  class='selectall' type='checkbox'>&nbsp; Select </span>");

    $(source + ' .selectall').click(function(e) {
        // e.preventDefault();
        var state = $(this).is(':checked');
        $(source + ' table tbody tr td input[type="checkbox"]').each(function(i, j) {
            console.log(i, j);
            $(this).attr('checked', state);
            id = $(this).attr("id");
            if (state)
                checked_ids.push(id);
            else
                checked_ids.splice(checked_ids.indexOf(id), 1);
            console.log($(this).attr('checked'));
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
    $(source + " .edit-or-copy-plan").click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');

        $('body #modal_plan_wizard').remove();
        $('body #modal_plan_run').remove();
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
            pageSize : 50
        },
        height : '446',
        groupable : false,
        scrollable : {
            virtual : true
        },
        selectable : false,
        sortable : true,
        pageable : true,

		dataBinding : onDataBinding,
		dataBound : onDataBound,
		
        columns : [{
            field : "id",
            title : "Select",
            sortable : false,
            width: '70px',
            template : "<input id='${id}' name='runs' type='checkbox' class='selected'>"
        }, {
            field : "sampleSetDisplayedName",
            title : "Sample Set",
            sortable : false,
            template : kendo.template($('#SampleSetColumnTemplate').html())
        }, {
            field : "planShortID",
            title : "Run Code",
            width: '70px',
            // template: "<a href='/data/project/${id}/results'>${name}</a>"
            template :kendo.template($('#PlanShortIdColumnTemplate').html())
        }, {
            field : "planDisplayedName",
            title : "Run Plan Name",
            width: '25%',
            sortable : true
        }, {
            field : "barcodeId",
            title : "Barcodes",
            width: '10%',
            sortable : true
        }, {
            field : "runType",
            title : "App",
            sortable : true,
            width: '75px',
            template : kendo.template($('#RunTypeColumnTemplate').html())
        }, {
            field : "sampleGroupingName",
        	title : "Group",
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
            sortable : false
        }, {
            title : "Sample",
            sortable : false,
            template : kendo.template($('#SampleColumnTemplate').html())
        }, {
            field : "sampleTubeLabel",
            title : "Sample Tube Label",
            sortable : false,
        }, {
            field : "chipBarcode",
            title : "Chip Barcode",
            sortable : false,
        }, {
            field : "date",
            title : "Last Modified",
            template : '#= kendo.toString(new Date(Date._parse(date)),"yyyy/MM/dd hh:mm tt") #'
        }, {
            field : "planStatus",
            title : "Status",
            width: '70px',
            sortable : true            	
        }, {        	
            title : " ",
            width : '4%',
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }],
    });

    switch_to_view(window.location.hash.replace('#',''));
    
    $('#dateRange').daterangepicker({dateFormat: 'yy-mm-dd'});
    $('.search_trigger').click(function (e) { filter(e); });

    $('#clear_filters').click(function () { console.log("going to reload!!"); window.location.reload(true); });
    
    $(function () {  
        $('#search_subject_nav').click(function(e) { 
            $("#plan_search_dropdown_menu").show();
        });
    });

    $(function () {  
        $('.search_chipBarcode').click(function(e) { 
            set_search_subject_chipBarcode(e);
        });
    });    
    
    $(function () {      
        $('.search_planName').click(function(e) {
            set_search_subject_planName(e);
        });    
    });    

    $(function () {          
        $('.search_sampleTubeLabel').click(function(e) { 
            set_search_subject_sampleTubeLabel(e);
        });
    });

    $(function () {
        $('.search_combinedLibraryTubeLabel').click(function(e) {
            set_search_subject_combinedLibraryTubeLabel(e);
        });
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

function set_search_subject_chipBarcode(e) {
    e.preventDefault();
    $('.search_chipBarcode_selected').removeClass("icon-white icon-check");         
    $('.search_chipBarcode_selected').addClass("icon-check");  
    $('.search_planName_selected').removeClass("icon-white icon-check"); 
    $('.search_planName_selected').addClass("icon-white"); 
    $('.search_sampleTubeLabel_selected').removeClass("icon-white icon-check"); 
    $('.search_sampleTubeLabel_selected').addClass("icon-white"); 
    $('.search_combinedLibraryTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_combinedLibraryTubeLabel_selected').addClass("icon-white");
    
    $("label[for='searchSubject']").text("chipBarcode");  
    $("#search_subject_nav").attr("title", "Search by chip barcode");  
    $("#plan_search_dropdown_menu").toggle();        
}

function set_search_subject_planName(e) {
    e.preventDefault();        
    $('.search_chipBarcode_selected').removeClass("icon-white icon-check");  
    $('.search_chipBarcode_selected').addClass("icon-white");  
    $('.search_planName_selected').removeClass("icon-white icon-check"); 
    $('.search_planName_selected').addClass("icon-check"); 
    $('.search_sampleTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_sampleTubeLabel_selected').addClass("icon-white");
    $('.search_combinedLibraryTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_combinedLibraryTubeLabel_selected').addClass("icon-white");
                   
    $("label[for='searchSubject']").text("planName");  
    $("#search_subject_nav").attr("title", "Search by plan name or code"); 
    $("#plan_search_dropdown_menu").toggle();                   
} 

function set_search_subject_sampleTubeLabel(e) {
    console.log("ENTER set_search_subject_sampleTubLabel");
    
    e.preventDefault();   
    $('.search_chipBarcode_selected').removeClass("icon-white icon-check"); 
    $('.search_chipBarcode_selected').addClass("icon-white");       
    $('.search_planName_selected').removeClass("icon-white icon-check"); 
    $('.search_planName_selected').addClass("icon-white");     
    $('.search_sampleTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_sampleTubeLabel_selected').addClass("icon-check");
    $('.search_combinedLibraryTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_combinedLibraryTubeLabel_selected').addClass("icon-white");

    $("label[for='searchSubject']").text("sampleTubeLabel");  
    $("#search_subject_nav").attr("title", "Search by sample tube label");
    $("#plan_search_dropdown_menu").toggle();   
} 

function set_search_subject_combinedLibraryTubeLabel(e) {
    console.log("ENTER set_search_subject_combinedLibraryTubeLabel");

	e.preventDefault();
    $('.search_chipBarcode_selected').removeClass("icon-white icon-check");
    $('.search_chipBarcode_selected').addClass("icon-white");
    $('.search_planName_selected').removeClass("icon-white icon-check");
    $('.search_planName_selected').addClass("icon-white");
    $('.search_sampleTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_sampleTubeLabel_selected').addClass("icon-white");
    $('.search_combinedLibraryTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_combinedLibraryTubeLabel_selected').addClass("icon-check");

    $("label[for='searchSubject']").text("combinedLibraryTubeLabel");
    $("#search_subject_nav").attr("title", "Search by combined library tube label");
    $("#plan_search_dropdown_menu").toggle();
}

function filter(e){
    e.preventDefault();
    e.stopPropagation();

    var daterange = $("#dateRange").val();
    if (daterange) {
        if (!/ - /.test(daterange)) { daterange = daterange + ' - ' + daterange; }
        daterange = daterange.replace(/ - /," 00:00,") + " 23:59";
    }

    var subjectToSearch = $("label[for='searchSubject']").text();
    console.log("filter - subjectToSearch=", subjectToSearch);
    
    if (subjectToSearch == "planName") {
    $("#grid").data("kendoGrid").dataSource.filter([
        {
            field: "date",
            operator: "__range",
            value: daterange
        },
        {
            field: "name_or_id",
            operator: "",
            value: $("#search_text").val()
        }
    ]);
    }
    else if (subjectToSearch == "chipBarcode") {
    $("#grid").data("kendoGrid").dataSource.filter([
        {
            field: "date",
            operator: "__range",
            value: daterange
        },
        {
            field: "chipBarcode",
            operator: "__icontains",
            value: $("#search_text").val()
        }
    ]);
    }
    else if (subjectToSearch == "sampleTubeLabel") {
    $("#grid").data("kendoGrid").dataSource.filter([
        {
            field: "date",
            operator: "__range",
            value: daterange
        },
        {
            field: "sampleTubeLabel",
            operator: "__icontains",
            value: $("#search_text").val()
        }
    ]);
    }
    else if (subjectToSearch == "combinedLibraryTubeLabel") {
    $("#grid").data("kendoGrid").dataSource.filter([
        {
            field: "date",
            operator: "__range",
            value: daterange
        },
        {
            field: "combinedLibraryTubeLabel",
            operator: "",
            value: $("#search_text").val()
        }
    ]);
    }
}
