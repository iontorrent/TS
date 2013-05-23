TB.namespace('TB.configure.reportdatamgmt.modal_edit_pruning_config');

TB.configure.reportdatamgmt.modal_edit_pruning_config.isValid = function() {
    var form = $('#modal_configure_report_data_mgmt_edit_pruning_config_form');
    var settings = form.uniform.defaults;
    return jQuery.fn.uniform.isValid(form, settings);
};

TB.configure.reportdatamgmt.modal_edit_pruning_config.init = function() {
    $(function() {//on ready
        $('#modal_configure_report_data_mgmt_edit_pruning_config').on('hidden', function() {
            $('body #modal_configure_report_data_mgmt_edit_pruning_config').remove();
        });

        TB.toggleContent();

        jQuery.fn.uniform.language.required = '%s is required';
        $('#modal_configure_report_data_mgmt_edit_pruning_config_form').uniform({
            holder_class : 'control-group',
            msg_selector : 'div.help-block.error',
            error_class : 'alert alert-error'
        });

        $('#modal_configure_report_data_mgmt_edit_pruning_config_form').submit(TB.configure.reportdatamgmt.modal_edit_pruning_config.isValid);

        $('#modal_configure_report_data_mgmt_edit_pruning_config #save').click(function(e) {
            e.preventDefault();
            console.debug($('#modal_configure_report_data_mgmt_edit_pruning_config_form').serializeArray());
            $('#modal_configure_report_data_mgmt_edit_pruning_config_form').submit();
        });

    });
};
