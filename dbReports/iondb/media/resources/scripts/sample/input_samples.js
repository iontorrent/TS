
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
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
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
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        
        console.log("at input_sample.js - bindActions - edit_sample - url=", url);
        
        $('body #modal_add_samplesetitem_popup').remove();
        $.get(url, function(data) {

        	if (data.indexOf("Error,") >= 0) {
                apprise(data);
            }
            else {
            	$('body').append(data);
            	$("#modal_add_samplesetitem_popup").modal("show");
            }
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
}


$(function() {
	
    $('.modal_save_samplesetitems').click(function(e) {
    	
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        //url = $(this).attr('href');
        url = "/sample/samplesetitem/input_save/"    
            
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
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });

});


$(document).ready(function() {
	  
    $('.modal_enter_sample').click(function(e) {
    	
        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        //url = $(this).attr('href');
        url = "/sample/samplesetitem/input/add/"    
            
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
            $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
            console.log("error:", data);

        }).always(function(data) {/*console.log("complete:", data);*/
            $('body').css("cursor", "default");
            $('.myBusyDiv').empty();
            $('body').remove('.myBusyDiv');
            delete busyDiv;
        });
    });


    
	var grid = $("#input_samplesetitem_grid").kendoGrid({

        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : "/sample/samplesetitem/input/getdata",
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
    					}					
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
		
		
    $(document).bind('modal_confirm_delete_done', function () {
		refreshKendoGrid("#input_samplesetitem_grid");
	});
  
}); 


function getColumns() {
    var columnArray = [];
    var custom_sample_list = [];
    	

	for (var i = 0; i < customAttributes_json.length; i++) {
    //for (attribute in attributes) {
    	console.log("input_samples.js - getColumns - LOOP - customAttributes_json[i]=", customAttributes_json[i]);
    	//var sampleAttribute = attributes[i];

    	customAttributes_index = i;

    	//document.getElementById("customAttribute").value = customAttributes_json[i]
    	if ( i < 20) {    	
    	custom_sample_list.push({ 
    	   field: customAttributes_json[i], 
    	   title: customAttributes_json[i], 
    	   sortable: false,
    	   //workaround 
    	   template: kendo.template($('#CustomSampleAttributeTemplate_'+i).html())
    	});
    	}
	}
	
    var default_columnArray = [
     {
         field: "displayedName", 
         title: "Sample Name", 
         sortable: true,
     } , {
         field: "externalId",
         title: "Sample ID",         
         sortable: true,
     } , {
         field: "barcodeKit",
         title: "Barcode Kit",        
         sortable: false,        
     } , {
         field: "barcode",
         title: "Barcode",        
         sortable: true,         
     } , {    	 
         field: "description",
         title: "Description",        
         sortable: false,         
     } , {
         field: "gender",
         title: "Gender",       
         sortable: true,
     } , {
         field: "relationshipRole",
         title: "Type",        
         sortable: true,
     } , {
         field: "relationshipGroup",
         title: "Group",        
         sortable: true,
     } , {
         field: "cancerType",
         title: "Cancer Type",       
         sortable: true,
     } , {
         field: "cellularityPct",
         title: "Cellularity %",       
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
