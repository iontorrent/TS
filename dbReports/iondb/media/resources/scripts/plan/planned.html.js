//models.PlannedExperiment.objects.filter(isReusable=False, planExecuted=False).order_by("-date", "planName")

$(document).ready(function() {
    var checked_ids = [];
    var grid = $("#grid").kendoGrid({
        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : "/rundb/api/v1/plannedexperiment/?isReusable=False&parentPlan=None&planExecuted=False",
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
                        //runMode: { type: "string" },
                        runType : {
                            type : "string"
                        },
                        projects : {
                            type : "string"
                        },
                        sampleDisplayedName : {
                            type : "string"
                        },
                        date : {
                            type : "string"
                        }
                    }
                }
            },
            serverSorting : true,
            sort : [{
                field : "date",
                dir : "desc"
            }, {
                field : "planDisplayedName",
                dir : "asc"
            }],
            serverPaging : true,
            pageSize : 100
        },
        height : '446',
        groupable : false,
        scrollable : {
            virtual : true
        },
        selectable : false,
        sortable : true,
        pageable : false,
        columns : [{
            field : "id",
            title : "Select",
            sortable : false,
            template : "<input id='${id}' name='runs' type='checkbox' class='selected'>"
        }, {
            field : "planShortID",
            title : "Run Code"
            // template: "<a href='/data/project/${id}/results'>${name}</a>"
        }, {
            field : "planDisplayedName",
            title : "Run Plan Name",
            sortable : true
        }, {
            field : "barcodeId",
            title : "Barcodes",
            sortable : true
            // } , {
            // field: "runMode",
            // title: "Run Type",
            // sortable: true,
            // template: '<span rel="tooltip" title="#= TB.runModeDescription(runMode)#">#= TB.runModeShortDescription(runMode)#</span>'
        }, {
            field : "runType",
            title : "Application",
            sortable : true,
            template : kendo.template($('#RunTypeColumnTemplate').html())
        }, {
            field : "projects",
            title : "Project",
            sortable : false
        }, {
            title : "Sample",
            sortable : false,
            template : kendo.template($('#SampleColumnTemplate').html())
        }, {
            field : "date",
            title : "Last Modified",
            template : '#= kendo.toString(new Date(Date.parse(date)),"yyyy/MM/dd hh:mm tt") #'
        }, {
            title : " ",
            width : '4%',
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }],
        dataBound : function(e) {
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

                    $('#error-messages').empty();
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

                    $('#error-messages').empty();
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
                    $('#error-messages').empty();
                    $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
                    console.log("error:", data);

                }).always(function(data) {/*console.log("complete:", data);*/
                });
            });
        }
    });

    $('.delete_selected').click(function(e) {
        e.preventDefault();
        e.stopPropagation();
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
                $('#error-messages').empty();
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
    $(document).bind('modal_confirm_delete_done modal_plan_wizard_done', function(e) {
        console.log(e.target, e.relatedTarget);
        refreshKendoGrid('#grid');
    });

}); 