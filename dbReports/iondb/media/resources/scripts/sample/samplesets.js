function count(value) {
	return value === null ? 0 : value.length;
}
function sample_type(value) {
	return value === null ? 0 : value.type;
}
$(function() {
	$('#modal_planexperiment').on('hidden', function(e){
		var $this = $(this);
		$this.data('modal', null);
	});
	$('#modal_planexperiment').on('shown', function(e){
		var $this = $(this);
		var expName = $this.data('modal') ? $this.data('modal').options.name : 'MISSING';
		$this.find('.modal-header h3 span').text( expName );
	});
	
});

function show_busy(show){
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    if (show){
        $('body').css("cursor", "wait");
        $('body').prepend(busyDiv);
    } else {
        $('body').css("cursor", "default");
        $('.myBusyDiv').remove();
    }
}

function onDataBinding(arg) {
    show_busy(true);
}

function onDataBound(arg) {
    show_busy(false)
    
    var source = '#sampleset_grid';
    bindActions(source);
    
    checked_ids = [];
}


function bindActions(source) {

    $(".edit_sampleset").click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        show_busy(true);
        
        var url = $(this).attr('href');
        $('body #modal_add_sampleset_popup').remove();

        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_add_sampleset_popup").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            show_busy(false);
        });
    });

    $(".plan-run").click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        show_busy(true);

        var url = $(this).attr('href');
        $('body #modal_planexperiment').remove();

        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_planexperiment").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            show_busy(false);
        });
    });
    	
    $(".delete_set").click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        show_busy(true);

        var url = $(this).attr('href');
        $('body #modal_confirm_delete').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_confirm_delete").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            show_busy(false);
        });
    });
    
    $(".libprep_summary").click(function(e){
		e.preventDefault();
		var url = $(this).attr('href');
		console.log('here', url)
	    $('body #modal_libraryprep_detail').remove();
	    $.get(url, function(data) {
	       $('body').append(data);
	       $('#modal_libraryprep_detail').modal("show");
	    });
	});
    
    $(source + ' [name=selected_sets]').click(function(e){
        var id = $(this).attr("id");
        if ($(this).attr("checked")){
            checked_ids.push(id);
        } else {
            checked_ids.splice(checked_ids.indexOf(id), 1);
        }
        $('#plan_from_selected').attr('disabled', checked_ids.length == 0)
    });
}
var checked_ids = [];

function onDetailDataBinding(arg) {
	//20130707-TODO-set cursor earlier
    show_busy(true);
}

function onDetailDataBound(arg) {
	console.log("at samplesets.js onDetailDataBound...");
    show_busy(false);

  var source = '#samplesetitem_attribute_grid';
  detailBindActions(source);
}

function detailBindActions(source) {
    $(".edit_sample_in_sampleset").unbind('click').click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        show_busy(true);

        var url = $(this).attr('href');
        
        $('body #modal_add_samplesetitem_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_add_samplesetitem_popup").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            show_busy(false);
        });
    });
   

    $(".remove_sample_from_set").unbind('click').click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        show_busy(true);

        var url = $(this).attr('href');
        
        $('body #modal_confirm_delete').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_confirm_delete").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {       	
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            show_busy(false);
        });        

    });
}


$(document).ready(function() {

    $('.modal_add_sample_attribute').click(function(e) {

        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        show_busy(true);

        url = $(this).attr('href');
            
        $('body #modal_add_attribute_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_add_attribute_popup").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            show_busy(false);
        });
    });
    
    $('#plan_from_selected').click(function(e){
        e.preventDefault();
        $('#error-messages').hide().empty();

        var ids = checked_ids.toString();
        if (ids == ""){
            console.log('no SampleSets selected');
            return;
        }

        var url = $(this).attr('href').replace('999999',ids);
        $('body #modal_planexperiment').remove();
        show_busy(true);

        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_planexperiment").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
        }).fail(function(data) {
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {/*console.log("complete:", data);*/
            show_busy(false);
        });
    });
    var urlRead = $("#sampleset_grid").data('urlRead');
    var urlUpdate = $("#sampleset_grid").data('urlUpdate');
    var grid = $("#sampleset_grid").kendoGrid({
        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : urlRead,
                    contentType : 'application/json; charset=utf-8',
                    type : 'GET',
                    dataType : 'json'
                },
                update : {
                    url : urlUpdate,
                    contentType : 'application/json; charset=utf-8',
                    type : 'GET',
                    dataType : 'json'
                },                
//                parameterMap : function(options) {
//                    return buildParameterMap(options);
//                }
                parameterMap : function(options, operation) {
                	if (operation !== "read" && options.models) {
                		return {
                			models : kendo.stringify(options.models)
                		};
                	}
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
                        displayedName : {
                            type : "string",
                            editable : true,
                            validation : {required : true}
                        },
                        lastModifiedDate : {
                            type : "string",
                            editable : false
                        },
                        description : {
                            type : "string",
                            editable : true
                        },
                        sampleCount : {
                        	type : "number",
                        	editable : false
                        },
                        SampleGroupType_CV : {
                       	    type : "string"
                        },
                        sampleGroupTypeName : {
                        	type : "string"
                        },
                        status : {
                        	type : "string",
                        	editable : false
                        },
                        libraryPrepType : {
                            type : "string",
                            editable : false                        
                        },
                        libraryPrepTypeDisplayedName : {
                            type : "string",
                            editable : false
                        },
                        pcrPlateSerialNum : {
                            type : "string",
                            editable : false                        
                        },
                        combinedLibraryTubeLabel : {
                            type : "string",
                            editable : false                        
                        },
                        libraryPrepKitDisplayedName: {
                            type : "string"
                        },
                    }
                }
            },
            pageSize: 50,
			serverPaging : true,
			serverSorting : false,
			serverFiltering : true,
        },
        sortable: {
        	mode: "multiple",
        	allowUnsort: true
        },
        scrollable : {
            virtual : false 
        },         
        //pageable: {pageSizes:[5,10,20,50]},
        pageable: true,
		detailTemplate : kendo.template($("#template").html()),
		detailInit : detailInit,
		dataBinding : onDataBinding,
		dataBound : onDataBound,        
        columns : [{
            field : "id",
            title : gettext('samplesets.fields.id.label'), //"Select",
            sortable : false,
            width: '50px',
            template : "<input id='${id}' name='selected_sets' type='checkbox' # if(!readyForPlanning){ # disabled # } # >"
        }, {
            field : "displayedName",
            title : gettext('samplesets.fields.displayedName.label'), //"Set Name"
//            sortable : true
        }, {
            field : "lastModifiedDate",
            title : gettext('samplesets.fields.lastModifiedDate.label'), //"Date",
//            sortable : true,
            template : '#= kendo.toString(new Date(Date.parse(lastModifiedDate)),"yyyy/MM/dd hh:mm tt") #'
        }, {
        	field : "sampleCount",
        	title: gettext('samplesets.fields.sampleCount.label'), //"# Samples"
        }, {
        	field : "description",
        	title : gettext('samplesets.fields.description.label'), //"Description"
        }, {
        	field : "sampleGroupTypeName",
        	title : gettext('samplesets.fields.sampleGroupTypeName.label'), //"Grouping"
        }, {
            field : "libraryPrepType",
            title : gettext('samplesets.fields.libraryPrepType.label'), //"Lib Prep Type",
            template : kendo.template($('#LibPrepTypeColumnTemplate').html())
        }, {
            field : "libraryPrepKitDisplayedName",
            title : gettext('samplesets.fields.libraryPrepKitDisplayedName.label'), //"Lib Prep Kit"
        }, {
            field : "pcrPlateSerialNum",
            title : gettext('samplesets.fields.pcrPlateSerialNum.label'), //"PCR Plate Serial #"
        }, {
            field : "combinedLibraryTubeLabel",
            title : gettext('samplesets.fields.combinedLibraryTubeLabel.label'), //"Combined Tube Label"
        }, {       
        	field : "status",
        	title : gettext('samplesets.fields.status.label'), //"Status"
//        	title : "Grouping",
//        	sortable : true,
//        	template : kendo.template($('#GroupingColumnTemplate').html())
  		
        }, {        	
            title : " ",
            width : '4%',
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }],
        columnResizeHandleWidth : 6
    });


    $(document).bind('modal_confirm_delete_done', function () {
    	refreshKendoGrid('#sampleset_grid');
	});

    $('.search_trigger').click(function (e) { filter(e); });

    $('#clear_filters').click(function () { console.log("going to reload!!"); window.location.reload(true); });
    
    $(function () {  
        $('#search_subject_nav').click(function(e) { 
            $("#sampleset_search_dropdown_menu").show();
        });
    });	
    $(function () {      
        $('.search_sampleSetName').click(function(e) {
            set_search_subject_sampleSetName(e);
        });    
    });    

    $(function () {
        $('.search_combinedTubeLabel').click(function(e) {
            set_search_subject_combinedTubeLabel(e);
        });
    });      	
}); 


function detailInit(e) {
	var detailRow = e.detailRow;
	var sampleSetPk = e.data.id;
	var isFusions = e.data.sampleGroupTypeName.indexOf('Fusions') > -1

    var detailUrl = detailRow.find(".samples").data('urlDetail');

	console.log("samplesets.js detailInit() sampleSetPk=", sampleSetPk, "; detailUrl=", detailUrl);
	
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
							type : "string",
						},
						sampleDisplayedName : {
							type : "string",
						},
						description : {
							type : "string",
						},
						sampleCollectionDate : {
							type : "string",
						},
						sampleReceiptDate : {
							type : "string",
						},
						relationshipGroup : {
							type : "number",

						},
						relationshipRole : {
							type : "string",
						},
						gender : {
							type : "string",
						},
						dnabarcode : {
							type : "string"
						},
						dnabarcodeKit : {
							type : "string"
						},						
						cancerType : {
							type : "string"
						},
						cellarlarityPct: {
							type : "number"
						},
                        nucleotideType: {
                            type : "string"
                        },
                        sampleSource: {
                            type : "string"
                        },
                        panelPoolType: {
                            type : "string"
                        },
                        pcrPlateRow: {
                            type : "string"
                        },
                        tubePosition: {
                            type : "string"
                        },
                        biopsyDays : {
                        	type : "number",
                        },
                        cellNum : {
                        	type : "string",
                        },
                        coupleId : {
                        	type : "string",
                        },
                        embryoId : {
                        	type : "string",
                        },
                        population : {
                        	type : "string",
                        },
                        mouseStrains : {
                        	type : "string",
                        },
					}
				}
			},
            pageSize: 10000,
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
        pageable: false,
        columns : getColumns(isFusions),
		dataBinding : onDetailDataBinding,
		dataBound : onDetailDataBound,
		columnResizeHandleWidth : 6
	});
}


function getColumns(isFusions) {
    var columnArray = [];
    var custom_sample_list = [];

    function _customAttributeTemplate(i) {
        return "# var _value = data.attribute_dict[customAttributes_json[" + i + "]] # \n" +
                "# if((typeof _value !== 'undefined') && _value) { #\n" +
                "#= _value # \n" +
                "# } else { # \n" +
                "#= '' # \n" +
                "# } #\n";
    }

	for (var i = 0; i < customAttributes_json.length; i++) {
    	console.log("samplesets.js - getColumns - LOOP - customAttributes_json[i]=", customAttributes_json[i]);
        custom_sample_list.push({
           field: customAttributes_json[i],
           title: customAttributes_json[i],
           sortable: false,
           //workaround
           template: kendo.template(_customAttributeTemplate(i))
        });
	}
	
    var default_columnArray = [
     {
         field: "sampleDisplayedName", 
         title: gettext('samplesets.SampleSetItemInfo.fields.sampleDisplayedName.label'), //"Sample Name",
         sortable: true,
         //template: kendo.template($('#sample_name_kendo_template').html())
     } , {
         field: "sampleExternalId",
         title: gettext('samplesets.SampleSetItemInfo.fields.sampleExternalId.label'), //"Sample ID",
         sortable: true,
         //template: kendo.template($('#sample_id_kendo_template').html())
     } , {
         field: "pcrPlateRow",
         title: gettext('samplesets.SampleSetItemInfo.fields.pcrPlateRow.label'), //"PCR Plate Position",
         sortable: true     
     } ,{
         field: "controlType",
         title: gettext('samplesets.SampleSetItemInfo.fields.controlType.label'), //"Control Type",
         sortable: false
     } , {
         field: "dnabarcode",
         title: gettext('samplesets.SampleSetItemInfo.fields.dnabarcode.label'), //"Barcode",
         sortable: true         
     } , {
         field: "description",
         title: gettext('samplesets.SampleSetItemInfo.fields.description.label'), //"Description",
         sortable: false,
         //template: kendo.template($('#sample_barcoding_id_kendo_template').html())    
     } , {
         field: "sampleCollectionDate",
         title: gettext('samplesets.SampleSetItemInfo.fields.sampleCollectionDate.label'), //"Sample Collection Date",
         template : '#= sampleCollectionDate != null ? kendo.toString(kendo.parseDate(sampleCollectionDate), "yyyy-MM-dd") : " " #',
         sortable: false,
     } , {
         field: "sampleReceiptDate",
         title: gettext('samplesets.SampleSetItemInfo.fields.sampleReceiptDate.label'), //"Sample Receipt Date",
         template : '#= sampleReceiptDate != null ? kendo.toString(kendo.parseDate(sampleReceiptDate), "yyyy-MM-dd") : " " #',
         sortable: false,
     } , {
         field: "nucleotideType",
         //title: isFusions? gettext('samplesets.SampleSetItemInfo.fields.nucleotideType.label.isFusions') /*"DNA/Fusions"*/: gettext('samplesets.SampleSetItemInfo.fields.nucleotideType.label'), //"DNA/RNA",
         title: "Nucleotide Type",
         sortable: true,
         template: kendo.template($('#sample_nucleotideType_kendo_template').html())  
     } , {
         field: "sampleSource",
         title: "Sample Source",
         sortable: true
     } , {
         field: "panelPoolType",
         title: "Panel Pool Type",
         sortable: true
     } , {
         field: "tubePosition",
         title: "Primer Pool Position", //Primer Pool Position
         sortable: true,
         width:'5%'
     } , {
         field: "gender",
         title: gettext('samplesets.SampleSetItemInfo.fields.gender.label'), //"Gender",
         sortable: true,
         //template: kendo.template($('#sample_libperpbarcode_kendo_template').html())
     } , {    	 
         field: "relationshipRole",
         title: gettext('samplesets.SampleSetItemInfo.fields.relationshipRole.label'), //"Type",
         sortable: true,
         //template : kendo.template($('#createdByFullname').html())
     } , {
         field: "relationshipGroup",
         title: gettext('samplesets.SampleSetItemInfo.fields.relationshipGroup.label'), //"Group",
         sortable: true,
         editor : relationshipGroupDropDownSelector,
         //template: "#=relationshipGroup.displayedName#"
     } , {
         field: "cancerType",
         title: gettext('samplesets.SampleSetItemInfo.fields.cancerType.label'), //"Cancer Type",
         sortable: true,
         //template: kendo.template($('#sample_cancerType_kendo_template').html())
     } , {
         field: "cellularityPct",
         title: gettext('samplesets.SampleSetItemInfo.fields.cellularityPct.label'), //"Cellularity %",
         sortable: true,
         //template: kendo.template($('#sample_cellularityPct_kendo_template').html()) 
     } , {
         field: "biopsyDays",
         title: gettext('samplesets.SampleSetItemInfo.fields.biopsyDays.label'), //"Biopsy Days",
         sortable: true,
     } , {
         field: "cellNum",
         title: gettext('samplesets.SampleSetItemInfo.fields.cellNum.label'), //"Cell Num",
         sortable: true,
     } , {
         field: "coupleId",
         title: gettext('samplesets.SampleSetItemInfo.fields.coupleId.label'), //"Couple ID",
         sortable: true,
     } , {
         field: "embryoId",
         title: gettext('samplesets.SampleSetItemInfo.fields.embryoId.label'), //"Embryo ID",
         sortable: true,         
     }, {
         field: "population",
         title: gettext('samplesets.SampleSetItemInfo.fields.population.label'), //"Population",
         sortable: true,
     }, {
         field: "mouseStrains",
         title: gettext('samplesets.SampleSetItemInfo.fields.mouseStrains.label'), //"Mouse Strains"
         sortable: true,
     }];

//    var default_last_columnArray = [
//     {
//		command : ["edit", "destroy"], 
//		title : "&nbsp;",
//		width : "172px"
//     }];
    
    var default_last_columnArray = [
     {
	    title : " ",
	    width : '4%',
	    sortable : false,
	    template : kendo.template($("#SampleActionColumnTemplate").html())
    }];

    for (i =0; i< default_columnArray.length; i++){
    	columnArray.push(default_columnArray[i])
    }
    for (i =0; i< custom_sample_list.length; i++){
        columnArray.push(custom_sample_list[i]);
    }
    for (i =0; i< default_last_columnArray.length; i++){
        columnArray.push(default_last_columnArray[i]);
    }
    return columnArray;
}


function relationshipGroupDropDownSelector(container, options) {
	var selectionUrl = "/rundb/api/v1/samplegrouptype_cv/?isActive=true&order_by=displayedName";
                
	$('<input required data-text-field="displayedName"  data-value-field="id"  data-bind="value:' + options.field + '"/>')
		.appendTo(container)
		.kendoDropDownList({
			autoBind : false,
			dataSource : "json",
			transport : {
                url : selectionUrl,
                contentType : 'application/json; charset=utf-8',
                type : 'GET',
                dataType : 'json'
			}
		});
}

function set_search_subject_sampleSetName(e) {
    e.preventDefault();        
    $('.search_sampleSetName_selected').removeClass("icon-white icon-check"); 
    $('.search_sampleSetName_selected').addClass("icon-check");
    $('.search_combinedTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_combinedTubeLabel_selected').addClass("icon-white");
                   
    $("label[for='searchSubject']").text("sampleSetName");  
    $("#search_subject_nav").attr("title", "Search by sample set name"); 
    $("#sampleset_search_dropdown_menu").toggle();                   
}

function set_search_subject_combinedTubeLabel(e) {
    e.preventDefault();
    $('.search_sampleSetName_selected').removeClass("icon-white icon-check");
    $('.search_sampleSetName_selected').addClass("icon-white");
    $('.search_combinedTubeLabel_selected').removeClass("icon-white icon-check");
    $('.search_combinedTubeLabel_selected').addClass("icon-check");

    $("label[for='searchSubject']").text("combinedTubeLabel");
    $("#search_subject_nav").attr("title", "Search by combined library tube label");
    $("#sampleset_search_dropdown_menu").toggle();
}

function filter(e){
    e.preventDefault();
    e.stopPropagation();

    var subjectToSearch = $("label[for='searchSubject']").text();
    var searchText = $("#search_text").val();
    console.log("filter - subjectToSearch=", subjectToSearch, "; searchText=", searchText);
    
    if (subjectToSearch == "sampleSetName") {
    $("#sampleset_grid").data("kendoGrid").dataSource.filter([
        {
            field: "displayedName",
            operator: "__icontains",
            value: $("#search_text").val()
        }
    ]);
    }
    else if (subjectToSearch == "combinedTubeLabel") {
    $("#sampleset_grid").data("kendoGrid").dataSource.filter([
        {
            field: "combinedLibraryTubeLabel",
            operator: "__icontains",
            value: $("#search_text").val()
        }
    ]);
    }
}
