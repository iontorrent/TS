
function getDisplayValue(value) {
	if ((typeof value !== 'undefined') && value) {
		return value;
    }
    else {
    	return "";
    }
}



function onDataBinding(arg) {
	//20130707-TODO-the busy cursor neds to be shown earlier!!
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    $('body').prepend(busyDiv);

}

function onDataBound(arg) {
	console.log("at input_samples.js onDataBound...");
	
	//20130707-TODO-test
    $('body').css("cursor", "default");
    $('.myBusyDiv').empty();
    $('body').remove('.myBusyDiv');
    
    var source = '#input_samplesetitem_grid';
    bindActions(source);
}

function bindActions(source) {
	
    $(".edit_sample").click(function(e) {
    	console.log("at input_samples.js - bindActions - edit_sample e=", e);

        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);
        libraryPrepType = get_libraryPrepType();
        url = $(this).attr('href');
        
        console.log("at input_sample.js - bindActions - edit_sample - url=", url);
        
        $('body #modal_add_samplesetitem_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_add_samplesetitem_popup").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);            
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            if (libraryPrepType == "amps_hd_on_chef_v1"){
                $('#pcrPlateRow').addClass('hide');
            }
            else{
                $('#pcrPlateRow').removeClass('hide');
            }
            delete busyDiv;
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

    $(".remove_sample_from_set").unbind('click').click(function(e) {
        e.preventDefault();
        $('#error-messages').hide().empty();
        show_busy(true);

        var url = $(this).attr('href');

        $('body #modal_confirm_delete').remove();
        $.get(url, function(data) {
            if (data == "true"){
                refreshKendoGrid("#input_samplesetitem_grid");
                show_busy(false);
            }
            $('body').append(data);
            $("#modal_confirm_delete").modal("show");
            return false;
        }).done(function(data) {
            show_busy(false);
            console.log("success:", url);
        }).fail(function(data) {
            show_busy(false);
            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);
        });

    });

}


$(function() {
	
    $('.modal_save_samplesetitems').click(function(e) {
    	
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');

        $('body #modal_save_samplesetitems_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
    		//$( "#modal_add_attribute_popup" ).data('source', "#input_samplesetitem_grid");
            $("#modal_save_samplesetitems_popup").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
            
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });

});


function get_libraryPrepType(){
    var libraryPrepType = $('#new_sampleSet_libraryPrepType :selected').val();
    if(! libraryPrepType ){
        libraryPrepType = $('#libraryPrepType :selected').val();
    }
    return libraryPrepType;
}

$(document).ready(function() {
	  
    $('.modal_enter_sample').click(function(e) {
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">' + gettext('global.messages.loading') + '</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        libraryPrepType = get_libraryPrepType();
        url = $(this).attr('href');
            
        $('body #modal_add_samplesetitem_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
    		//$( "#modal_add_attribute_popup" ).data('source', "#input_samplesetitem_grid");
            $("#modal_add_samplesetitem_popup").modal("show");
            return false;
        }).done(function(data) {
            console.log("success:", url);
            
        }).fail(function(data) {
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');

            $('#error-messages').empty().show();
            $('#error-messages').append('<p class="error">' + gettext('global.messages.error.label') + ': ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            if (libraryPrepType == "amps_hd_on_chef_v1"){
                 $('#pcrPlateRow').addClass('hide');
            }
            else{
                $('#pcrPlateRow').removeClass('hide');
            }
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });

    url = $("#edit_amp_sampleSet").val();
    if (!url) {
        url = "/sample/samplesetitem/input/getdata";
    }
    console.log(url);
    
	var grid = $("#input_samplesetitem_grid").kendoGrid({

        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : url,
                    //contentType : 'application/json; charset=utf-8',
                    type : 'GET',
                    //dataType : 'json'
                },
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
    					pending_id : {
    						type : "number",
    						defaultValue: null
    					},
    					externalId : {
    						type : "string",
    					},
    					displayedName : {
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
    					cancerType : {
    						type : "string",
    					},
    					cellularityPct : {
    						type : "number",
                        },
                        barcodeKit : {
                        	type : "string",
                        },
    					barcode : {
    						type : "string",
    					},
                        nucleotideType : {
                            type : "string",
                        },
                        sampleSource : {
                            type : "string",
                        },
                        pcrPlateRow : {
                            type : "string",
                        },
                        panelPoolType : {
                            type : "string",
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
            pageSize: 100,
			serverPaging : false,
			serverFiltering : false,
			serverSorting : false,
        },
	    //pageable: {pageSizes:[5,10,20,50]},
        filterable: false,
        sortable: false,
        pageable: false,        
	    columns : getColumns(),
		dataBinding : onDataBinding,
		dataBound : onDataBound	    
		});

    // hide the pcr plate column in the sample grid based on library prep type selection
	var libraryPrepType = $('#new_sampleSet_libraryPrepType :selected').val();
    if(! libraryPrepType ){
        libraryPrepType = $('#libraryPrepType :selected').val();
    }
    pcrPlateRowDisplay(libraryPrepType);
    $("select#libraryPrepType").change(function () {
       pcrPlateRowDisplay($(this).val());

    });
    $("select#new_sampleSet_libraryPrepType").change(function () {
        pcrPlateRowDisplay($(this).val());
    });
    $(document).bind('modal_confirm_delete_done', function () {
		refreshKendoGrid("#input_samplesetitem_grid");
	});
  
}); 

function pcrPlateRowDisplay(libraryPrepType) {
    var grid = $("#input_samplesetitem_grid").data("kendoGrid");
    if (libraryPrepType == 'amps_hd_on_chef_v1') {
        grid.hideColumn('pcrPlateRow');
    }
    else {
        grid.showColumn('pcrPlateRow')
    }
}

function getColumns() {
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
    	console.log("input_samples.js - getColumns - LOOP - customAttributes_json[i]=", customAttributes_json[i]);
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
         field: "displayedName", 
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.displayedName.label'), //"Sample Name"
         sortable: true,
     } , {
         field: "externalId",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.externalId.label'), //"Sample ID",
         sortable: true,
     } , {
         field: "pcrPlateRow",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.pcrPlateRow.label'), //"PCR Plate Position",
         sortable: true,
     } , {
     /*    field: "barcodeKit",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.barcodeKit.label'), //"Barcode Kit",
         sortable: false,        
     } , {  */
         field: "barcode",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.barcode.label'), //"Barcode",
         sortable: true,         
     } , {    	 
         field: "description",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.description.label'), //"Description",
         sortable: false,         
     } , {
         field: "sampleCollectionDate",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.sampleCollectionDate.label'), //"Sample Collection Date",
         sortable: false,
     }, {
         field: "sampleReceiptDate",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.sampleReceiptDate.label'), //"Sample Receipt Date",
         sortable: false,
     }, {
         field: "nucleotideType",
         //title: gettext('samplesets.PendingSampleSetItemInfo.fields.nucleotideType.label'), //"DNA/ RNA/ Fusions",
         title: "Nucleotide Type",
         sortable: true,
     } , {
         field: "sampleSource",
         title: "Sample Source",
         sortable: true,
     } , {
         field: "panelPoolType",
         title: "Panel Pool Type",
         sortable: true,
     } , {
         field: "gender",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.gender.label'), //"Gender",
         sortable: true,
     } , {
         field: "relationshipRole",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.relationshipRole.label'), //"Type",
         sortable: true,
     } , {
         field: "relationshipGroup",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.relationshipGroup.label'), //"Group",
         sortable: true,
     } , {
         field: "cancerType",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.cancerType.label'), //"Cancer Type",
         sortable: true,
     } , {
         field: "cellularityPct",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.cellularityPct.label'), //"Cellularity %",
         sortable: true,
     } , {
         field: "biopsyDays",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.biopsyDays.label'), //"Biopsy Days",
         sortable: true,
     } , {
         field: "cellNum",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.cellNum.label'), //"Cell Num",
         sortable: true,
     } , {
         field: "coupleId",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.coupleId.label'), //"Couple ID",
         sortable: true,
     } , {
         field: "embryoId",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.embryoId.label'), //"Embryo ID",
         sortable: true,
     }, {
         field: "population",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.population.label'), //"Population",
         sortable: true,
     }, {
         field: "mouseStrains",
         title: gettext('samplesets.PendingSampleSetItemInfo.fields.mouseStrains.label'), //"MouseStrains",
         sortable: true,
     }];

    
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
