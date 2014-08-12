TB.namespace('TB.data.modal_combine_results');
TB.data.modal_combine_results.ready = function(method) {
    $("#results_table").kendoGrid({
        sortable : true,
        scrollable : false
    });
    var uniformSettings = {
        holder_class : 'control-group',
        msg_selector : 'p.help-block',
        error_class : 'alert alert-error'
    };
    var $uniform = $('#modal_combine_results_form').uniform(uniformSettings);

    $('#modal_combine_results').on('hidden', function() {
        $('body #modal_combine_results').remove();
    });
    $('#modal_combine_results_form').submit(function(e) {
        e.preventDefault();
        return false;
    });

    $('#modal_combine_results .btn-primary').click(function(e) {
        var that = this;
        e.preventDefault();

        var selected = [];
        $('input:checkbox[name=selected_results]:checked').each(function() {
            selected.push($(this).val().split("|")[0]);
        });
        if (selected.length < 2) {
            alert("Please select 2 or more results to combine.");
            return false;
        }
        /*Perform uni-form validation*/
        if (!jQuery.fn.uniform.isValid($('#modal_combine_results_form'), jQuery.fn.uniform.defaults)) {
            $("#modal_combine_results_form").animate({
                scrollTop : 0
            }, "slow");
            $(".error").effect("highlight", {
                "color" : "#F20C18"
            }, 2000);
            return false;
        }

        var json = $('#modal_combine_results_form').serializeJSON(), url = $('#modal_combine_results_form').attr('action'), type = method;

        json.mark_duplicates = $("#mark_duplicates").is(":checked");
        json.selected_pks = selected;

        // if override, send new sample names
        if (json.override_samples){
            var samples = $('#modal-new_samples .sample_name');
            if (barcoded){
                var barcodedSamples = {};
                samples.each(function(){
                    if (this.value){
                        if (barcodedSamples[this.value] === undefined){
                            barcodedSamples[this.value] = {};
                            barcodedSamples[this.value]['barcodes'] = [];
                        }
                        barcodedSamples[this.value]['barcodes'].push( $(this).closest('tr').find('.barcode_id').text() );
                    }
                });
                json.barcodedSamples = barcodedSamples;
            } else {
                json.sample = samples.val();
            }
        }
        
        $('#modal_combine_results #modal-error-messages').hide().empty();
        json = JSON.stringify(json);
        console.log('transmitting :', type, url, json);
        
        var jqxhr = $.ajax(url, {
            type : type,
            data : json,
            contentType : 'application/json',
            dataType : 'html',
            processData : false
        }).done(function(data) {
            //console.log("success:",  data);
            var found = $(data).find('#modal_report_analyze_started');
            if (found.exists()) {
                $('body').append(data);
                $('#modal_report_analyze_started').modal('show');
                $('#modal_report_analyze_started').on('hidden', function() {
                    $('body #modal_report_analyze_started').parent().remove();
                    analysis_liveness_off();
                    // window.location.reload(true);
                });
                analysis_live_ready_cb();
            }
            $('#modal_combine_results').trigger('modal_combine_results_done', {});
            $('#modal_combine_results').modal("hide");
        }).fail(function(data) {
            $('#modal_combine_results #modal-error-messages').empty().append('<p class="error">ERROR: ' + data.responseText + '</p>').show();
            console.log("error:", data);
        });
    });

    $('#modal_combine_results #override_samples').change(function(e) {
        if ($(this).is(':checked')){
            $('#modal_combine_results #modal-new_samples').modal("show");
        }
    });
    
    //If display a kendo-grid then you'll want to removeClass('hide') on dataBound event
    $('#modal_combine_results .btn-primary').removeClass('hide');
    
};
