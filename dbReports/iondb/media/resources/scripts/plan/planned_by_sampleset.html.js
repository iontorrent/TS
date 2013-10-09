
function onDataBinding(arg) {
	console.log("at planned_by_sampleset.html.js onDataBinding...");
	//20130707-TODO - does not work!!
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    $('body').prepend(busyDiv);

}

function onDataBound(arg) {
	console.log("at planned_by_sampleset.html.js onDataBound...");
    $('body').css("cursor", "default");
    $('.myBusyDiv').empty();
    $('body').remove('.myBusyDiv');
    
    var source = "#grid";
    $(source + ' span[rel="popover"]').each(function(i, elem) {
        $(elem).popover({
            content : $($(elem).data('select')).html()
        });
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
}

$(document).ready(function() {
    var checked_ids = [];
    var grid = $("#grid").kendoGrid({
        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : "/rundb/api/v1/plannedexperiment/?isReusable=False&planExecuted=False&isSystem=False&sampleSet__isnull=False",
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
                        sampleSet : {
                        	type : "string"
                        },
                        sampleSetDisplayedName : {
                        	type : "string"
                        },
                        sampleSetGroupType : {
                        	type : "string"
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
                        },
                        planStatus : {
                            type : "string"
                        },
                        sampleSet : {
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
            	field : "sampleSetDisplayedName",
            	dir : "asc"
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
		
		dataBinding : onDataBinding,
		dataBound : onDataBound,

        columns : [{
        	field : "sampleSetDisplayedName",
            title : "Sample Set",
            sortable : true
            //template :kendo.template($('#SampleSetColumnTemplate').html())
        }, {
            field : "planShortID",
            title : "Run Code",
            // template: "<a href='/data/project/${id}/results'>${name}</a>"
            template :kendo.template($('#PlanShortIdColumnTemplate').html())
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
//        	title : "Group",
//            sortable : false,
//            template : kendo.template($('#GroupColumnTemplate').html())            
//        }, {
        	field : "sampleSetGroupType",
        	title : "Group",
        	sortable : true
        }, {
            title : "Sample",
            sortable : false,
            template : kendo.template($('#SampleColumnTemplate').html())
        }, {
            field : "date",
            title : "Last Modified",
            template : '#= kendo.toString(new Date(Date.parse(date)),"yyyy/MM/dd hh:mm tt") #'
        }, {
            field : "planStatus",
            title : "Status",
            sortable : true            	
        }, {        	
            title : " ",
            width : '4%',
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }],
    });

    $('.delete_selected').click(function(e) {
        e.preventDefault();
        e.stopPropagation();
        $('#error-messages').hide().empty();
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
    
    //20130626-TODO
//    $(".shortcode128").each(function () {
//    	console.log("at planned.html.js shortcode128!!!");
//    	
//    	$(this).barcode(
//    			$(this).data("barcode"), 
//    			"code128", {
//    			barWidth: 2, 
//    			barHeight: 30, 
//                bgColor: $(this).parent().parent().css("background-color") 
//        });
//    });
    
    $(document).bind('modal_confirm_delete_done modal_plan_wizard_done', function(e) {
        console.log(e.target, e.relatedTarget);
        refreshKendoGrid('#grid');
    });
    

}); 


//TODO: can we just fetch by sampleSetId? can we gather all the data upfront in the master table?
function detailInit(e) {
	var detailRow = e.detailRow;
	var sampleSetPk = e.data.sampleSet.id;

	//var detailUrl = "/rundb/api/v1/samplesetiteminfo/?sampleSet=" + sampleSetPk + "&order_by=sample__displayedName";
	var detailUrl = "/rundb/api/v1/samplesetiteminfo/?order_by=sample__displayedName";

	//var detailUrl = "/rundb/api/v1/samplesetiteminfo";
	
	console.log("planned_by_sampleset.html.js detailInit() sampleSetPk=", sampleSetPk, "; detailUrl=", detailUrl);
	
	detailRow.find(".samples").kendoGrid({
		dataSource : {
			type : "json",
			transport : {
              read : {
                  url : detailUrl,
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
						sampleExternalId : {
							type : "string"
						},
						sampleDisplayedName : {
							type : "string"
						},
						sampleDescription : {
							type : "string"
						},
						relationshipGroup : {
							type : "number"
						},
						relationshipRole : {
							type : "string"
						},
						gender : {
							type : "string"
						}
					}
				}
			},
          pageSize: 5,
			serverPaging : true,
			serverSorting : false,
			filter : {
				field : "sampleSetPk",
				operator : "eq",
				value : e.data.id
			}			
      },
      sortable: {
      	mode: "multiple",
      	allowUnsort: true
      },
      pageable: {pageSizes:[5,10,20,50]},	
		columns : [
		{
			field : "sampleExternalId",
			title : "Sample ID",
		}, {
			field : "sampleDisplayedName",
			title : "Name",
		}, {
			field : "gender",
			title : "Gender",			
		}, {
			field : "sampleDescription",
			title : "Description",
		}, {
			field : "relationshipRole",
			title : "Type",
		}, {
			field : "relationshipGroup",
			title : "Group",
		}, {
			command : ["edit"], 
			title : "&nbsp;",
			width : "172px"
		}],
		editable : "inline"
	});
}