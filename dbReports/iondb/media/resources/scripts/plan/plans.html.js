
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
                total : "meta.total_count"
            },
            serverPaging : true,
            pageSize : 5
        },
        height : 'auto',
        scrollable : false,
        pageable : true,
        rowTemplate : kendo.template($("#rowTemplate").html()),
        dataBound : function(e) {
            commonDataBoundEvent(target, msg);
        }
    };
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
    console.log(e.target, e.relatedTarget);
    var target, grid;

    // refreshKendoGrid('#favorites');
    // refreshKendoGrid('#recents');
    // var target = $(e.target).data('source');
    $('.main .plan-grid table').each(function(i, element) {
        refreshKendoGrid(element);
    });
});

$(document).ready(function() {
    var basePlannedExperimentUrl = "/rundb/api/v1/plannedexperiment/?format=json&isReusable=true&planExecuted=False&isSystemDefault=False";
    var orderByOptions = "&order_by=-date&order_by=planDisplayedName";

    var favorites = $("#favorites").kendoGrid(commonKendoGrid("#favorites",
        basePlannedExperimentUrl + "&isFavorite=true&username=" + username + orderByOptions,
        'No Favorites yet'));
    var recents = $("#recents").kendoGrid(commonKendoGrid("#recents",
        basePlannedExperimentUrl + orderByOptions,
        'No Recents yet'));
    var ampliSeqs = $("#ampliSeqs").kendoGrid(commonKendoGrid("#ampliSeqs",
        basePlannedExperimentUrl + "&runType=AMPS" + orderByOptions,
        'No Ampliseq DNA templates yet'));
    var wholeGenomes = $("#wholeGenomes").kendoGrid(commonKendoGrid("#wholeGenomes",
        basePlannedExperimentUrl + "&runType=WGNM" + orderByOptions,
        'No Whole Genome templates yet'));
    var targetSeqs = $("#targetSeqs").kendoGrid(commonKendoGrid("#targetSeqs",
        basePlannedExperimentUrl + "&runType=TARS" + orderByOptions,
        'No TargetSeq templates yet'));
    var rnaSeqs = $("#rnaSeqs").kendoGrid(commonKendoGrid("#rnaSeqs",
        basePlannedExperimentUrl + "&runType=RNA" + orderByOptions,
        'No RNASeq templates yet'));
    var genericSeqs = $("#genericSeqs").kendoGrid(commonKendoGrid("#genericSeqs",
        basePlannedExperimentUrl + "&runType=GENS" + orderByOptions,
        'No Generic Sequencing templates yet'));

    var ampliSeqRna = $("#ampliSeqRna").kendoGrid(commonKendoGrid("#ampliSeqRna",
        basePlannedExperimentUrl + "&runType=AMPS_RNA" + orderByOptions,
        'No Ampliseq RNA templates yet'));

    var ampliSeqExome = $("#ampliSeqExome").kendoGrid(commonKendoGrid("#ampliSeqExome",
        basePlannedExperimentUrl + "&runType=AMPS_EXOME" + orderByOptions,
        'No Ampliseq Exome templates yet'));

    var targetSeq_16s = $("#16sTargetSeq").kendoGrid(commonKendoGrid("#16sTargetSeq",
        basePlannedExperimentUrl + "&runType=TARS_16S" + orderByOptions,
        'No 16S Target Sequencing templates yet'));

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
