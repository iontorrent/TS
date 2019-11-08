$(function() {  //DOM ready handler
    $("#email_table").kendoGrid({
        dataSource: {
            type: "json",
            transport: {
                read: "/rundb/api/v1/emailaddress/",
                parameterMap: function(options) {
                    return buildParameterMap(options)
                }
            },
            schema: {
                data: "objects",
                total: "meta.total_count",
                model: {
                    fields: {
                        id: { type: "number" },
                        email: { type: "string" },
                        selected: { type: "boolean" },
                    }
                }
            },
            serverSorting: true,
            serverPaging: true,
            pageSize: 10,
            sort: ({ field: "id", dir: "desc" })
        },
        height: '219',
        groupable: false,
        scrollable: {
            virtual: true
        },
        selectable: false,
        sortable: true,
        pageable: false,
        columns: EMAIL_COLUMNS,  // Defined by templates/rundb/configure/configure.html
        dataBound: function(e) {
            $(".emailselected").change(function() {
                var checked = $(this).is(":checked"), pk = $(this).data("pk");
                $.ajax({
                    type : "PATCH",
                    url : "/rundb/api/v1/emailaddress/" + pk + "/",
                    contentType : "application/json",
                    data : JSON.stringify({
                        "selected" : checked
                    })
                });
            });
            
            $('.edit_email').click(function (e) {
                e.preventDefault();
                e.stopPropagation();
                url = $(this).attr('href');
                $('#modal_configure_edit_email').remove();
                $.get(url, function(data) {
                    $('body').append(data);
                    $('#modal_configure_edit_email').modal('show');
                });
            });
            $('.delete_email').click(function (e) {
                e.preventDefault();
                e.stopPropagation();
                url = $(this).attr('href');
                $('body #modal_confirm_delete').remove();
                $.get(url, function(data) {
                    $('body').append(data);
                    $( "#modal_confirm_delete" ).data('source', '#email_table');
                    $( "#modal_confirm_delete" ).data('customevents', jQuery.makeArray(['modal_confirm_delete_done']));
                    $( "#modal_confirm_delete" ).modal("show");
                });
            });
            
            // hide/show table as needed
            if ($("#email_table").data('kendoGrid')._data.length > 0){
                $("#email_table").show();
                $("#no_email_configured").hide();
            } else {
                $("#email_table").hide();
                $("#no_email_configured").show();
            }
        }
    });

    $('#add_email').click(function (e) {
        e.preventDefault();
        e.stopPropagation();
        url = $(this).attr('href');
        $('#modal_configure_edit_email').remove();
        $.get(url, function(data) {
            $('body').append(data);
            $('#modal_configure_edit_email').modal('show');
        });
    });
    
    $("#enable_nightly").change(function(){
		var enabled = $(this).is(':checked');
		$.ajax({
			type: "PATCH",
			dataType: 'json',
			url: "/rundb/api/v1/globalconfig/1/",
			data: '{"enable_nightly_email":'+enabled+'}',
			contentType: 'application/json'
		});
    });
    
    $(document).bind('modal_configure_edit_email_done modal_confirm_delete_done', function () {
		refreshKendoGrid("#email_table");
	});
});


