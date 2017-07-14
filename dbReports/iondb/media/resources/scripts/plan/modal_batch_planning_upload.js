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

        $("#postedfile").click(function(e){
            $(".submitUpload").attr("disabled", false);
        });

        $("#submitUpload").click(function(e) {
            invalidPost = $(".submitUpload").attr("disabled");
            if (!invalidPost) {
                e.preventDefault();
                if (processing) return false;
                else processing = true;
                $('#modalBatchPlanningUpload').submit();
            }
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
        $("#submitUpload").attr("disabled", true);
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
        var singleCSV = responseText.singleCSV;
        var total_errors = responseText.totalErrors;
        var inputPlanCount = responseText.inputPlanCount;
        var plansFailed = responseText.plansFailed;

        hasErrors = false;
        var error = "";

        if (msg.toLowerCase().indexOf("error") >= 0) {
            hasErrors = true;
        	error += "<p>" + msg + "</p>";
        }

        // Do not process if un-supported csv/zip are uploaded
        // just display the errors.
        if (!responseText.status && !inputPlanCount) {
            hasErrors = true;
            for (var key in responseText.failed) {
                error += "<strong>" + key + " contained error(s):</strong> ";
                for (var i = 0; i < responseText.failed[key].length; i++) {
                    error += "<li><strong>  " + responseText.failed[key][i] + "</strong></li>";
                }
            }
        }
        else if (responseText.failed) {
        	if (!hasErrors) {
            	error += "<p>" + msg + "<br>";
            }
            error +=  plansFailed  + " / " + inputPlanCount + " plan(s) failed" + "<br>";
            error += "Total no. of failures: " + total_errors + "</p>";
            if ( plansFailed > 1) {
                error +=  "Only one plan errors will be displayed. To see specific plan or all errors, click button below:";
                dropDown_menu = get_failed_plans_dropdown(responseText.failed);
                error += dropDown_menu;
            }
            error_table = '<ul><table class="table table-striped table-condensed table-bordered"><tbody>';
            skipFirstPlan = true;
            for (var key in responseText.failed) {
                hasErrors = true;
                plan_no = key.match(/\d+/) - 1;
                error += '<div id=' + key;

                //Display first failed plan errors always
                error += (!skipFirstPlan) ? ' style=\"display:none\">' : '>';
                skipFirstPlan = false;

                error += "<ul class='unstyled'>";
                error += "<li><strong>" + key + " (Plan " + plan_no + ") contained error(s):</strong> ";
                error += "<ul>";
                if (singleCSV){
                    error += error_table;
                }
                IRU_flag = false;
                for (var i = 0; i < responseText.failed[key].length; i++) {
                    errorCount_total = responseText.failed[key].length - 1;
                    console.log(errorCount_total);
                    BC_IR_flag = false;
                    error += '<tr>';
                    if (key.indexOf('Row') > -1){
                        error += "<li><strong>  " + responseText.failed[key][i][0] + "</strong> column ";
                        error += " : " + responseText.failed[key][i][1];
                        error += "</li>";
                    } else {
                        var errorLists = responseText.failed[key][i];
                        $.each(errorLists, function(index, value){
                            if ((singleCSV) && (!IRU_flag)){
                                error += '<td class="single_csv">' + value + '</td>';
                            }
                            if (value == "Sample CSV File Name: "){
                                BC_IR_flag = true;
                                error += "<li><strong>" + value + "</strong>" + errorLists[index+1] + "</li>";
                            }
                            if ((!singleCSV) && ((value == "Barcoded samples validation errors:") || (value == "IRU validation errors:"))){
                                BC_IR_flag = true;
                                IRU_flag = true;
                                error = get_iru_bc_error_table(index, value, errorLists, error);
                            }
                            if ((!singleCSV) && (!BC_IR_flag) && (errorLists[index+1])){
                                error += "<li><strong>" + value + "</strong> :   " + errorLists[index+1] + "</li>";
                            }
                        });
                        if (!BC_IR_flag){
                                 error += '</tr>';
                        }
                    }
                }
                if (!BC_IR_flag){
                    error += '</tbody>';
                    error += '</table></ul>';
                }
                error += "</ul>";
                error += "</li>";
                error += "</ul>";
                error += "</div>";
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

// Construct the drop down menu to list the failed plans
function get_failed_plans_dropdown(invalidPlans){
    dropDown_menu = '<div class="dropdown">' +
                    '<button type="button" class="btn btn-danger dropdown-toggle" data-toggle="dropdown">' +
                    '<span class="sr-only">Choose plan to view errors</span>' +
                    '<span class="caret"></span>' +
                    '</button>' +
                    '<ul class="dropdown-menu" role="menu">';

    var all_failed_plans = [];
    for (var col_row in invalidPlans) {
        plan_no = col_row.match(/\d+/) - 1
        if (plan_no) {
            all_failed_plans.push(col_row);
        }
        dropDown_menu += '<li><a href="#" onclick="toggler(\'' + col_row + '\');">Plan ' + plan_no + '</a></li>';
    }

    show_all_item = '<li><a href="#" onclick="toggler(\'show_all\');">Show all errors</a></li>';
    all_failed_plans = '<input name="all_failed_plans" type="hidden" value=' +(JSON.stringify(all_failed_plans)) + '>';
    dropDown_menu += '<li class="divider"></li>';
    dropDown_menu += show_all_item + '</ul></div>';
    dropDown_menu += all_failed_plans;

    return dropDown_menu;
}

// Construct the table to display the IRU and Barcoded validation error message
function get_iru_bc_error_table(index, value, errorLists, error){
     var data;
     //handle any exception if data is not in json format
     try {
        data = JSON.parse(errorLists[index+1]);
        BC_IR_err_count = $.map(data, function(n, i) { return i; }).length;
        error_table =  '<ul><table class="table table-striped table-condensed table-bordered">' +
                       '<thead>' +
                       '<tr><th colspan="2">' + value + BC_IR_err_count + '</th></tr>' +
                       '<tr>' +
                       '<th class="bc_ir" data-field="id">Row #</th>' +
                       '<th data-field="errormsg">Error Message</th>' +
                       '</tr>' +
                       '</thead>' +
                       '<tbody>';
        error += error_table
        for(var key in data){
              error += '<tr><td class="bc_ir" >' + key + '</td><td>' + data[key] + '</td></tr>';
        }
         error += '</tbody>';
         error += '</table></ul>';
     }
     catch (e) {
        error = errorLists[index+1]
     }
    return error;
}

function toggler(divId) {
    var values = $("input[name='all_failed_plans']").val();
    all_failed_plans = $.parseJSON(values);

    $.each(all_failed_plans, function(index, value){
        divId == 'show_all' ? $("#" + value).show(1000) : $("#" + value).hide(1000);
    });

    if (divId != 'show_all'){
        $("#" + divId).show(1000);
    }
}
