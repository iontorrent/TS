TB.namespace('TB.configure.reportdatamgmt.modal_add_prunegroup');

TB.configure.reportdatamgmt.modal_add_prunegroup.isValid = function() {
    var form = $('#modal_configure_report_data_mgmt_add_prune_group_form');
    var settings = form.uniform.defaults;
    return jQuery.fn.uniform.isValid(form, settings);
};

TB.configure.reportdatamgmt.modal_add_prunegroup.init = function() {
    $(function() {//on document ready
        $('#modal_configure_report_data_mgmt_add_prune_group').on('hidden', function() {
            $('body #modal_configure_report_data_mgmt_add_prune_group').remove();
        });

        jQuery.fn.uniform.language.required = gettext('uni-form-validation.language.required');
        $('#modal_configure_report_data_mgmt_add_prune_group_form').uniform({
            holder_class : 'control-group',
            msg_selector : 'div.help-block.error',
            error_class : 'alert alert-error'
        });

        $('#modal_configure_report_data_mgmt_add_prune_group_form').submit(function() {
            return false;
            //always return false
        });

        $('#modal_configure_report_data_mgmt_add_prune_group #save').click(function(e) {
            /* The flow (form) is designed to submit the form contents via AJAX POST instead of form submit.
             */
            e.preventDefault();
            if (TB.configure.reportdatamgmt.modal_add_prunegroup.isValid()) {
                var $form = $('#modal_configure_report_data_mgmt_add_prune_group_form');
                var url = $form.attr("action");
                var data = $form.serializeArray();
                $.post(url, data, function(data) {
                    loadPruneGroups();
                    //refreshes the list of Prune Groups on the page behind modal
                    $("#modal_configure_report_data_mgmt_add_prune_group").modal("hide");
                }).fail(function(data) {
                    $('#modal_configure_report_data_mgmt_add_prune_group #modal-error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>').show();
                });
            }
        });

    });
};
