TB.namespace('TB.plan.batchupload');

TB.plan.batchupload.ready = function(plannedUrl) {

    $('#modal_batch_planning_upload').on('hidden', function() {
        $('body #modal_batch_planning_upload').remove();
    });

    var processing;

    $(function() {
        jQuery.fn.uniform.language.required = '%s is required';
        $('#modalBatchPlanningUpload').uniform({
            holder_class : 'control-group',
            msg_selector : 'p.help-block.error',
            error_class : 'alert alert-error',
            prevent_submit : false
        });

        processing = false;
        $("#submitUpload").click(function(e) {
            e.preventDefault();
            if (processing) return false;
            else processing = true;
            $('#modalBatchPlanningUpload').submit();
        });

        $.ajaxSetup({
            async : false
        });

        $('#modalBatchPlanningUpload').ajaxForm({
            beforeSubmit : verify,
            success : handleResponse,
            error : AjaxError,
            dataType : 'json'
        });
    });

    //Check if there is a file
    function verify() {
        $("#postedfile").blur();

        var inputVal = $("#postedfile").val();
        if (!jQuery.fn.uniform.isValid($('#modalBatchPlanningUpload'), jQuery.fn.uniform.defaults)) {
            processing = false;
            return false;
        }

        $('#modal_batch_planning_upload .modal-body #modal-error-messages').addClass('hide').empty();
        $("#loadingstatus").html("<div class='alert alert-info'><img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'> Uploading csv file for plans </div>");
    }

    function AjaxError() {
        $("#loadingstatus").html("<div class='alert alert-error'>Failure uploading file!</div>");
        processing = false;
    }

    //handleResponse will handle both successful upload and validation errors
    function handleResponse(responseText, statusText) {
        console.log("responseText..", responseText);
        $("#loadingstatus").html("");
        var msg = responseText.status;
        console.log(msg);

        hasErrors = false;
        var error = "";
        
        if (msg.toLowerCase().indexOf("error") >= 0) {
            hasErrors = true;
        	error += "<p>" + msg + "</p>";
        }
        
        if (responseText.failed) {
        	if (!hasErrors) {
            	error += "<p>" + msg + "</p>";        		
        	}
        	
            for (var key in responseText.failed) {
                hasErrors = true;
                error += "<ul class='unstyled'>";

                error += "<li><strong> Row " + key + " contained error(s):</strong> ";
                error += "<ul>";
                for (var i = 0; i < responseText.failed[key].length; i++) {
                    error += "<li><strong>  " + responseText.failed[key][i][0] + "</strong> column ";
                    error += " : " + responseText.failed[key][i][1];
                    error += "</li>";
                }
                error += "</ul>";
                error += "</li>";

                error += "</ul>";
            }
            //console.log(error);
        }

        if (hasErrors) {
            $('#modal_batch_planning_upload .modal-body #modal-error-messages').removeClass('hide').html(error);
            processing = false;
        }
        else  {
            $('#modal_batch_planning_upload').modal("hide");
            window.location = plannedUrl;
        }
    }

};
