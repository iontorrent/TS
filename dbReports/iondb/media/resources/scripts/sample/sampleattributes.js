

function onDataBinding(arg) {
	//20130707-TODO-the busy cursor neds to be shown earlier!!
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    $('body').prepend(busyDiv);

}

function onDataBound(arg) {
	console.log("at sampleattributes.js onDataBound...");
	//20130707-TODO-test
    $('body').css("cursor", "default");
    $('.myBusyDiv').empty();
    $('body').remove('.myBusyDiv');
    
    var source = '#samplesetitem_attribute_grid';
    bindActions(source);
}


function bindActions(source) {

    $(".edit_sampleattribute").click(function(e) {
    	console.log("at sampleattributes.js - bindActions - edit_sampleattribute e=", e);

        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        
        console.log("at sampleattributes.js - bindActions - edit_sampleattribute - url=", url);
        
        $('body #modal_add_attribute_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);

            $("#modal_add_attribute_popup").modal("show");
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
    
    
    $('.delete_sampleattribute').click(function(e) {
    	console.log("at sampleattribute.js - bindActions - delete_sampleattribute e=", e);

        $('body').css("cursor", "wait");
        e.preventDefault();
        
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        //alert(url);
        
        $('body #modal_confirm_delete').remove();
        $('modal_confirm_delete_done');
        $.get(url, function(data) {
            $('body').append(data);
    		//$( "#modal_confirm_delete" ).data('source', "#samplesetitem_attribute_grid");
            $("#modal_confirm_delete").modal("show");
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



$(document).ready(function() {
  
    $('.modal_add_sample_attribute').click(function(e) {

        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        //url = $(this).attr('href');
        url = "/sample/sampleattribute/add/"    
            
        $('body #modal_add_attribute_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
    		//$( "#modal_add_attribute_popup" ).data('source', "#samplesetitem_attribute_grid");
            $("#modal_add_attribute_popup").modal("show");
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

    
    var checked_ids = [];
    var grid = $("#samplesetitem_attribute_grid").kendoGrid({
        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : "/rundb/api/v1/sampleattribute/?order_by=-lastModifiedDate",
                    contentType : 'application/json; charset=utf-8',
                    type : 'GET',
                    dataType : 'json'
                },
                update : {
                    url : "/rundb/api/v1/sampleattribute/?order_by=-lastModifiedDate",
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
                        description : {
                            type : "string",
                            editable : true
                        }, 
                        isActive : {
                        	type : "boolean",
                        	editable : true
                        },   
                        isMandatory : {
                        	type : "boolean",
                        	editable : true
                        },
                        dataType_name : {
                        	type : "string",
                        	editable : true
                        },
                        sampleCount : {
                        	type : "number",
                        	editable : false
                        },
                        lastModifiedDate : {
                            type : "string",
                            editable : false
                        }           
                    }
                }
            },
            pageSize: 10,
			serverPaging : true,
			serverSorting : false,
        },
        sortable: {
        	mode: "multiple",
        	allowUnsort: true
        },
        pageable: {pageSizes:[5,10,20,50]},
		dataBinding : onDataBinding,
		dataBound : onDataBound,        
        columns : [{
            field : "displayedName",
            title : "Attribute Name"
//            sortable : true
            // template: "<a href='/data/project/${id}/results'>${name}</a>"
        }, {
        	field : "description",
        	title : "Description"
        }, {
        	field : "dataType_name",
        	title: "Data Type",
    	    template : kendo.template($("#SampleAttributeDataTypeColumnTemplate").html())
        }, {
        	field : "isMandatory",
        	title: "Required",
//            sortable : false,        	       	
        	template : kendo.template($("#IsMandatoryColumnTemplate").html())
        }, {
        	field : "isActive",
        	title: "To Show",
//            sortable : false,        	
        	template : kendo.template($("#IsActiveColumnTemplate").html())
        }, {
        	field : "sampleCount",
        	title: "# Samples",
//            sortable : false,        	       		        		
        }, {
            field : "lastModifiedDate",
            title : "Date",
//            sortable : false,
            template : '#= kendo.toString(new Date(Date.parse(lastModifiedDate)),"yyyy/MM/dd hh:mm tt") #'  		
        }, {        	
            title : " ",
            width : '4%',
            sortable : false,
            template : kendo.template($("#ActionColumnTemplate").html())
        }]
    });


    $(document).bind('modal_confirm_delete_done', function () {
		refreshKendoGrid("#samplesetitem_attribute_grid");
	});
  
}); 



function sampleAttributeDataTypeDropDownSelector(container, options) {
	var selectionUrl = "/rundb/api/v1/sampleattributedatatype/?isActive=True&order_by=displayedName";
                
	$('<input required data-text-field="datatype"  data-value-field="id"  data-bind="value:' + options.field + '"/>')
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