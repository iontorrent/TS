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


function onDataBinding(arg) {
	//20130707-TODO-the busy cursor neds to be shown earlier!!
    var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
    $('body').prepend(busyDiv);

}

function onDataBound(arg) {
	console.log("at samplesets.js onDataBound...");
	//20130707-TODO-test
    $('body').css("cursor", "default");
    $('.myBusyDiv').empty();
    $('body').remove('.myBusyDiv');
    
    var source = '#sampleset_grid';
    bindActions(source);
}


function bindActions(source) {

    $(".edit_sampleset").click(function(e) {
    	console.log("at samplesets.js - bindActions - edit_sampleset e=", e);

        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        //url = "/sampleattribute/add/"    
        //url = "/sampleset/" + _id + "/edit/"
        
        console.log("at samplesets.js - bindActions - edit_sampleset - url=", url);
        
        $('body #modal_add_sampleset_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
    		//$( "#modal_add_attribute_popup" ).data('source', "#sampleset_grid");
            $("#modal_add_sampleset_popup").modal("show");
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

    $(".plan-run").click(function(e) {
        console.log("at samplesets.js - bindActions - plan_run e=", e);

        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        
        console.log("at samplesets.js - bindActions - plan_run - url=", url);
        
        $('body #modal_planexperiment').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $("#modal_planexperiment").modal("show");
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
    	
    $(".delete_set").click(function(e) {
    	console.log("at samplesets.js  - bindActions - delete_set e=", e);
    	
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

        	if (data.indexOf("Error,") >= 0) {
                apprise(data);
            }
            else {
            $('body').append(data);
    		//$( "#modal_confirm_delete" ).data('source', "#input_samplesetitem_grid");
            $("#modal_confirm_delete").modal("show");
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


function onDetailDataBinding(arg) {
	//20130707-TODO-set cursor earlier
  var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
  $('body').prepend(busyDiv);

}

function onDetailDataBound(arg) {
	console.log("at samplesets.js onDetailDataBound...");
  $('body').css("cursor", "default");
  $('.myBusyDiv').empty();
  $('body').remove('.myBusyDiv');

  var source = '#samplesetitem_attribute_grid';
  detailBindActions(source);
}

function detailBindActions(source) {
    $(".edit_sample_in_sampleset").click(function(e) {
    	console.log("at samplesets.js - detailBindActions - edit_sample_in_sampleset e=", e);

        $('body').css("cursor", "wait");
        e.preventDefault();
        $('#error-messages').hide().empty();
        var busyDiv = '<div class="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
        $('body').prepend(busyDiv);

        url = $(this).attr('href');
        
        //url = "/sampleattribute/add/"    
        //url = "/sampleset/" + _id + "/edit/"
        
        console.log("at samplesets.js - detailBindActions - edit_sample_in_sampleset - url=", url);
        
        $('body #modal_add_samplesetitem_popup').remove();
        $.get(url, function(data) {
            $('body').append(data);
    		//$( "#modal_add_attribute_popup" ).data('source', "#sampleset_grid");
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
   

    $(".remove_sample_from_set").click(function(e) {
    	
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
        	if (data.indexOf("Error,") >= 0) {
                apprise(data);
            }
            else {
	            $('body').append(data);
	    		//$( "#modal_confirm_delete" ).data('source', "#input_samplesetitem_grid");
	            $("#modal_confirm_delete").modal("show");
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


$(document).ready(function() {
	
	//20130715-TODO
    //$('#relationshipGroup').spinner({ min: 0, max: 10000 });
    
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
    		//$( "#modal_add_attribute_popup" ).data('source', "#sampleset_grid");
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
    var grid = $("#sampleset_grid").kendoGrid({
        dataSource : {
            type : "json",
            transport : {
                read : {
                    url : "/rundb/api/v1/sampleset/?order_by=-lastModifiedDate",
                    contentType : 'application/json; charset=utf-8',
                    type : 'GET',
                    dataType : 'json'
                },
                update : {
                    url : "/rundb/api/v1/sampleset/?order_by=-lastModifiedDate",
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
//                        SampleGroupType_CV : {
//                        	type : "string"
//                        }                        
                        sampleGroupTypeName : {
                        	type : "string"
                        },
                        status : {
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
		detailTemplate : kendo.template($("#template").html()),
		detailInit : detailInit,
		dataBinding : onDataBinding,
		dataBound : onDataBound,        
        columns : [{
            field : "displayedName",
            title : "Set Name"
//            sortable : true
            // template: "<a href='/data/project/${id}/results'>${name}</a>"
        }, {
            field : "lastModifiedDate",
            title : "Date",
//            sortable : true,
            template : '#= kendo.toString(new Date(Date.parse(lastModifiedDate)),"yyyy/MM/dd hh:mm tt") #'
        }, {
        	field : "sampleCount",
        	title: "# Samples"
        }, {
        	field : "description",
        	title : "Description"
        }, {
        	field : "sampleGroupTypeName",
        	title : "Grouping"
        }, {
        	field : "status",
        	title : "Status"        		
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
}); 


function detailInit(e) {
	var detailRow = e.detailRow;
	var sampleSetPk = e.data.id;

	console.log("samplesets.js - detailInit - detailRow=", detailRow);
	console.log("samplesets.js - detailInit - data.id=", sampleSetPk);

	var detailUrl = "/rundb/api/v1/samplesetiteminfo/?order_by=sample__displayedName";
	
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
                update : {
                    url : "/rundb/api/v1/sampleset/?order_by=-lastModifiedDate",
                    contentType : 'application/json; charset=utf-8',
                    type : 'GET',
                    dataType : 'json'
                },
                destroy : {
                    url : "/rundb/api/v1/sampleset/?order_by=-lastModifiedDate",
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
//							nullable: true,
//							editable : true,
//							validation : {
//								required : false
//							}
						},
						sampleDisplayedName : {
							type : "string",
//							nullable : false,
//							editable : true,
//							validation : {
//								required : true
//							}
						},
						sampleDescription : {
							type : "string",
//							nullable : true,
//							editable : true,
//							validation: { 
//								required: false
//							}
						},
						relationshipGroup : {
							type : "number",
//							nullable: true,
//							editable : true,
//							validation: { 
//								required: false
//							},
//							defaultValue: { 
//								id: 1, 
//								displayedName: "fake value"
//							}
						},
						relationshipRole : {
							type : "string",
//							nullable: true,
//							validation: { 
//								required: false
//							}
						},
						gender : {
							type : "string",
//							nullable: true,
//							editable : true,
//							validation: { 
//								required: false
//							}
						}
					}
				}
			},
            pageSize: 100,
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
        //columns: TB.sample.sampleset.getColumns(),
        columns : getColumns(),

		dataBinding : onDetailDataBinding,
		dataBound : onDetailDataBound,
		columnResizeHandleWidth : 6		
//		columns : [
//		{
//			field : "sampleExternalId",
//			title : "Sample ID",
//		}, {
//			field : "sampleDisplayedName",
//			title : "Name",
//		}, {
//			field : "gender",
//			title : "Gender",			
//		}, {
//			field : "sampleDescription",
//			title : "Description",
//		}, {
//			field : "relationshipRole",
//			title : "Type"
//		}, {
//			field : "relationshipGroup",
//			title : "Group",
//			editor : relationshipGroupDropDownSelector,
//			template : "#=relationshipGroup.displayedName#"
//		}, {
//			command : ["edit", "destroy"], 
//			title : "&nbsp;",
//			width : "172px"
//		}],
		//20130709 editable : "inline"
	});
}


function getColumns() {
    var columnArray = [];
    var custom_sample_list = [];
    	
    //var customAttributes = $("#customAttributes").val();
    //var customAttributes = "{{custom_sample_column_list}}";
	//console.log("samplesets.js - getColumns - customAttributes=", customAttributes);
	//console.log("samplesets.js - getColumns - customAttributes_json=", customAttributes_json);

	for (var i = 0; i < customAttributes_json.length; i++) {
    //for (attribute in attributes) {
    	console.log("samplesets.js - getColumns - LOOP - customAttributes_json[i]=", customAttributes_json[i]);
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
    	   //template: '#= data.attribute_dict[customAttributes_json[customAttributes_index]] #'
    	   //20130715-test template: '#= $(this).data.attribute_dict["My_sample_attribute"]  #'
    	   //template: '#= CustomSampleAttributeTemplate(customAttributes_json[i]) #'
    	   //template:"#= data.attr_value_{{attribute.displayedName}}#" 
    	});
		}
	}

/*20130714-donotwork
	var attributeObjs = $("#customAttributeObjs").val().toArray();
	$.each(attributeObjs, function(key, value) {
    	console.log("samplesets.js - getColumns - LOOP - key=", key +"; value=", value);
    	
    	if (key === "displayedName") {
        	custom_sample_list.push({ 
         	   field: value, 
         	   title: value, 
         	   sortable: false,
         	   template: '#= $(this).data.attribute_dict[value]  #'
         	});    		
    	}
	});
*/

	
    var default_columnArray = [
     {
         field: "sampleDisplayedName", 
         title: "Sample Name", 
         sortable: true,
         //template: kendo.template($('#sample_name_kendo_template').html())
     } , {
         field: "sampleExternalId",
         title: "Sample ID",         
         sortable: true,
         //template: kendo.template($('#sample_id_kendo_template').html())
     } , {    	 
         field: "gender",
         title: "Gender",       
         sortable: true,
         //template: kendo.template($('#sample_libperpbarcode_kendo_template').html())
     } , {
         field: "sampleDescription",
         title: "Description",        
         sortable: false,
         //template: kendo.template($('#sample_barcoding_id_kendo_template').html())
     } , {
         field: "relationshipRole",
         title: "Type",        
         sortable: true,
         //template : kendo.template($('#createdByFullname').html())
     } , {
         field: "relationshipGroup",
         title: "Group",        
         sortable: true,
         editor : relationshipGroupDropDownSelector,
         //template: "#=relationshipGroup.displayedName#"
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