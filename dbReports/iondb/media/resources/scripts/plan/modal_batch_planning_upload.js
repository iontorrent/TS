TB.namespace('TB.plan.batchupload');

TB.plan.batchupload.ready = function(plannedUrl, key_barcoded_samples_validation_errors, key_sample_csv_file_name, key_iru_validation_errors) {

    $('#modal_batch_planning_upload').on('hidden', function() {
        $('body #modal_batch_planning_upload').remove();
    });

    var processing;

    $(function() {
        jQuery.fn.uniform.language.required = gettext('uni-form-validation.language.required');
        $('#modalBatchPlanningUpload').uniform({
            holder_class : 'control-group',
            msg_selector : 'p.help-block.error',
            error_class : 'alert alert-error',
            prevent_submit : false
        });

        processing = false;
        $("#postedfile").on('change', function(e){
            $("#postedfile").blur();
        });

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
        $("#loadingstatus").html("<div class='alert alert-info'><img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'>" + gettext('upload_plans_for_template.loadingstatus') + "</div>");
        $("#submitUpload").attr("disabled", true);
    }

    function AjaxError() {
        var _html = kendo.template($('#UploadPlansForTemplateFailureTemplate').html())({});
        $("#loadingstatus").html(_html);
        processing = false;
    }
    function isJSON(data)
    {
        var isJson = false;

        try
        {
            //This works with JSON string and JSON object, not sure about others.
            var json = $.parseJSON(data);
            isJson = (typeof(json) === 'object');
        }
        catch (ex) {console.error('data is not JSON');}

        return isJson;
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

        function getFailedKeyI18N(key) {
            var key18n = key; //default
            var parsedKey = /:?(?<key>column|row)(?<number>\d+)/.exec(key.toLowerCase());
            if (parsedKey) {
                gettext('row')
                gettext('column')
                key18n = interpolate("%(key)s %(number)s", {key: gettext(parsedKey.groups.key.toLowerCase()), number: parsedKey.groups.number}, true)
            }
            return key18n;
        }

        hasErrors = false;
        var error = "";
        var warning = "";

        //DANGEROUS to check for existence of "error" within i18n.. likely BUG!
        if (msg.toLowerCase().indexOf("error") >= 0) {
            hasErrors = true;
            error += "<p>" + msg + "</p>";
        }

        // Do not process if un-supported csv/zip are uploaded
        // just display the errors.
        if (!responseText.status && !inputPlanCount) {
            hasErrors = true;
            for (var key in responseText.failed) {
                error += "<strong>" + interpolate(gettext('upload_plans_for_template.errors.rowOrColumn'), {key:key}, true) + "</strong>"; //'%(key)s contained error(s):'
                for (var i = 0; i < responseText.failed[key].length; i++) {
                    error += "<li><strong>  " + responseText.failed[key][i] + "</strong></li>";
                }
            }
        }
        else if ('NON_ASCII_FILES' in responseText.failed) {
            hasErrors = true;
            error += "<p>" + responseText.failed['NON_ASCII_FILES'] + "<br>";
        }
        else if (responseText.failed) {
            if (!hasErrors) {
                error += "<p>" + msg + "<br>";
            }
            error += interpolate(gettext('upload_plans_for_template.errors.summary'), {plansFailed:plansFailed, inputPlanCount:inputPlanCount}, true) + "<br>"; //%(plansFailed)s of %(inputPlanCount)s plan(s) failed
            error += interpolate(gettext('upload_plans_for_template.errors.totalerrors'), {total_errors:total_errors}, true) + "</p>"; //Total no. of failures: %(total_errors)s
            if ( plansFailed > 1) {
                error +=  gettext('upload_plans_for_template.errors.instructions'); //Only one plan's errors will be displayed. To see specific plan or all errors, click button below:
                dropDown_menu = get_failed_plans_dropdown(responseText.failed);
                error += dropDown_menu;
            }
            error_table = '<ul><table class="table table-striped table-condensed table-bordered"><tbody>';
            skipFirstPlan = true;
            for (var key in responseText.failed) {
                hasErrors = true;
                plan_no = key.match(/\d+/) - 1;
                var key18n = getFailedKeyI18N(key);
                error += '<div id=' + key;

                //Display first failed plan errors always
                error += (!skipFirstPlan) ? ' style=\"display:none\">' : '>';
                skipFirstPlan = false;

                error += "<ul class='unstyled'>";
                error += "<li>" + interpolate(gettext('upload_plans_for_template.errors.rowOrColumn.title'), {rowOrColumnKey: key18n, plan_number:plan_no}, true); //<strong> %(rowOrColumnKey)s (Plan %(plan_number)s) contained error(s):</strong>
                error += "<ul>";
                plan_param = "Plan_" + plan_no
                if (singleCSV){
                    error += error_table;
                }
                IRU_flag = false;
                for (var i = 0; i < responseText.failed[key].length; i++) {
                    errorCount_total = responseText.failed[key].length - 1;
                    console.log(errorCount_total);
                    BC_IR_flag = false;
                    error += '<tr>';



                    if (key.indexOf('Row') > -1) {
                        if (responseText.failed[key][i][0] == key_barcoded_samples_validation_errors) {
                            if ( isJSON(responseText.failed[key][i][1]) ) {
                                barcodeSamplesJson = typeof responseText.failed[key][i][1] === "string" ? $.parseJSON(responseText.failed[key][i][1]) : responseText.failed[key][i][1]
                                for(var barcodeSamplesJson_key in barcodeSamplesJson) {
                                    error += "<li>";
                                    error += interpolate(gettext('upload_plans_for_template.errors.rowOrColumnError'), {columnName:barcodeSamplesJson_key, columnErrors: barcodeSamplesJson[barcodeSamplesJson_key]}, true); //<strong>%(columnName)s</strong> column  : %(columnErrors)s
                                    error += "</li>";
                                }
                            } else { // show general errors related to barcodedSamples
                                error += "<li><strong>  " + responseText.failed[key][i][1] + "</strong>";
                            }
                        } else {
                        error += "<li>";
                        error += interpolate(gettext('upload_plans_for_template.errors.rowOrColumnError'), {columnName:responseText.failed[key][i][0], columnErrors: responseText.failed[key][i][1]}, true);
                        error += "</li>";
                        }
                    } else {
                        var errorLists = responseText.failed[key][i];
                        $.each(errorLists, function(index, value){
                            if ((singleCSV) && (!IRU_flag)){
                                error += '<td class="single_csv">' + value + '</td>';
                            }
                            if (value == key_sample_csv_file_name){
                                BC_IR_flag = true;
                                error += "<li><strong>" + value + "</strong>" + errorLists[index+1] + "</li>";
                            }
                            if ((!singleCSV) && ((value == key_barcoded_samples_validation_errors) || (value == key_iru_validation_errors))){
                                BC_IR_flag = true;
                                IRU_flag = true;
                                //error = get_iru_bc_error_table(index, value, errorLists, error);
                                error = get_iru_bc_error_table(index, value, errorLists, error,  plan_param);
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

        if (responseText.warnings) {
            for (var key in responseText.warnings) {
                plan_no = key.match(/\d+/) - 1;
                warning += "<div>"
                warning += interpolate(gettext('upload_plans_for_template.warnings.rowOrColumn.title'), {rowOrColumnKey: key, plan_number:plan_no}, true); //"<strong> %(rowOrColumnKey)s (Plan %(plan_number)s) contained warning(s):</strong>"
                warning += "<ul>"
                for (var i = 0; i < responseText.warnings[key].length; i++) {
                    warning += '<li>'+ responseText.warnings[key][i] +'</li>';
                }
                warning +='</ul></div>'
                $('#modal_batch_planning_upload .modal-body #modal-error-messages').removeClass('hide').html(warning);
            }
        }

        if (hasErrors) {
            $('#modal_batch_planning_upload .modal-body #modal-error-messages').removeClass('hide').html(error);
            processing = false;
        } else if (warning){
            $('#modal_batch_planning_upload .modal-body #modal-success-messages').html(responseText.status).show();
            $('#modal_batch_planning_upload .modal-body #modal-error-messages').html(warning).show();
            $('#modal_batch_planning_upload #submitUpload').hide();
            $('#modal_batch_planning_upload #dismissUpload').text(gettext("global.action.modal.close"));
        } else {
            $('#modal_batch_planning_upload').modal("hide");
            window.location = plannedUrl;
        }

        $(".warnings").click(function(e){
            window.location = plannedUrl;
        });
    }

};

// Construct the drop down menu to list the failed plans
function get_failed_plans_dropdown(invalidPlans){
    dropDown_menu = '<div class="dropdown">' +
                    '<button type="button" class="btn btn-danger dropdown-toggle" data-toggle="dropdown">' +
                    '<span class="">' + gettext('upload_plans_for_template.errors.choose.dropdown') + '</span>' + //Choose plan to view errors
                    '<span class="caret"></span>' +
                    '</button>' +
                    '<ul class="dropdown-menu" role="menu">';

    var all_failed_plans = [];
    for (var col_row in invalidPlans) {
        plan_no = col_row.match(/\d+/) - 1
        if (plan_no) {
            all_failed_plans.push(col_row);
        }
        dropDown_menu += '<li><a href="#" onclick="toggler(\'' + col_row + '\');">' + interpolate(gettext('upload_plans_for_template.errors.choose.dropdown.choice'), {plan_number:plan_no}, true) + '</a></li>'; //Plan %(plan_number)s
    }

    show_all_item = '<li><a href="#" onclick="toggler(\'show_all\');">' + gettext('upload_plans_for_template.errors.choose.all') + '</a></li>'; //Show all errors
    all_failed_plans = '<input name="all_failed_plans" type="hidden" value=' +(JSON.stringify(all_failed_plans)) + '>';
    dropDown_menu += '<li class="divider"></li>';
    dropDown_menu += show_all_item + '</ul></div>';
    dropDown_menu += all_failed_plans;

    return dropDown_menu;
}

// Construct the table to display the IRU and Barcoded validation error message
function get_iru_bc_error_table(index, value, errorLists, error, plan_param){
     var data;
     table_id = "iru_validation_errors_" + plan_param;
     //handle any exception if data is not in json format
     try {
        data = typeof errorLists[index+1] === "string" ? JSON.parse(errorLists[index+1]) : errorLists[index+1];
        BC_IR_err_count = $.map(data, function(n, i) { return i; }).length;
        error_table =  '<ul><table id=' + table_id + ' class="table table-striped table-condensed table-bordered">' +
                       '<thead>' +
                       '<tr><th colspan="2">' + value + BC_IR_err_count + '</th></tr>' +
                       '<tr>' +
                       '<th class="bc_ir" data-field="id">' + gettext('upload_plans_for_template.errors.column.rowNumber.title') + '</th>' + // Row #
                       '<th data-field="errormsg">' + gettext('upload_plans_for_template.errors.column.errormsg.title') + '</th>' + //Error Message
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
//
// function toHtmlUnorderNestedLists(json) {
//     var error = [];
//     $.each(json, function(planRowOrColumnName, planRowOrColumnErrorObjectsList){
//         plan_number = planRowOrColumnName.match(/\d+/) - 1;
//         error.push("<ul class='unstyled'>");
//         error.push("<li><strong>" + planRowOrColumnName + " (Plan " + plan_number + ") contained error(s):</strong> ");
//
//         if (planRowOrColumnName.indexOf('Row') > -1) {
//             $.each(planRowOrColumnErrorObjectsList, function (index, planRowOrColumnErrorObject) {
//                 if (planRowOrColumnErrorObject && _.size(planRowOrColumnErrorObject) > 0) {
//                     error.push("<ul>");
//                     // if ($.isPlainObject(planRowOrColumnError)) {
//                     //     error.push("<li><strong>  " + index + "</strong> column : ");
//                     // } else if ($.isArray(planRowOrColumnError)) {
//                     //     error.push("<ul><li><strong>  " + planRowOrColumnErrorObject[index] + "</strong> column : ");
//                     // }
//                     $.each(planRowOrColumnErrorObject, function (keyOrIndex, planRowOrColumnError) {
//                         error.push("<ul>");
//                         if ($.isNumeric(keyOrIndex)) {
//                             error.push("<ul><li><strong>  " + planRowOrColumnError + "</strong> column : ");
//                         } else {
//                             error.push("<li><strong>  " + keyOrIndex + "</strong> column : ");
//                         }
//
//                         $.each(planRowOrColumnError, function (j, stringOrListOrObject) {
//                             if (typeof(stringOrListOrObject) === 'string') {
//                                 error.push(stringOrListOrObject);
//                             } else if (typeof(stringOrListOrObject) === 'object') { // list or object
//                                 if (stringOrListOrObject && _.size(stringOrListOrObject) > 0) {
//                                     error.push("<ul>");
//                                     $.each(stringOrListOrObject, function (jj, value) {
//                                         error.push("<li><strong>" + jj + "</strong> : " + value + "</li>");
//                                     });
//                                     error.push("</ul>");
//                                 }
//                             } else {
//                                 error.push(stringOrListOrObject);
//                             }
//
//                         });
//                         error.push("</li>");
//                         error.push("</ul>");
//
//                     });
//                     // error.push("</li>");
//                     // error.push("</ul>");
//                 }
//             });
//         }
//         error.push("</li>");
//         error.push("</ul>");
//     });
//     return error;
// }
