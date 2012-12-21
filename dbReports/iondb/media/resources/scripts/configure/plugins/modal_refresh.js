TB.namespace('TB.configure.plugins.modal_refresh');

TB.configure.plugins.modal_refresh.selector = $('#modal_confirm_plugin_refresh');

TB.configure.plugins.modal_refresh.reset = function() {
    $('#modal_confirm_plugin_refresh #modal-success-messages').addClass('hide').empty();
    $('#modal_confirm_plugin_refresh #modal-error-messages').addClass('hide').empty();
    $('#modal_confirm_plugin_refresh #modal-plugin-information').addClass('hide');
    $('#modal_confirm_plugin_refresh #modal-plugin-information pre code').empty();
};
TB.configure.plugins.modal_refresh.loading = function() {
	$('#modal_confirm_plugin_refresh #single_msg').toggleClass('hide'); // toggle Refresh button to invisibile
	$('#modal_confirm_plugin_refresh .btn-primary').toggleClass('hide'); // toggle Refresh button to invisibile
    $('#modal_confirm_plugin_refresh #modal-info-messages').toggleClass('hide').html("<img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'> Refreshing plugin's information");
}

TB.configure.plugins.modal_refresh.ready = function() {
    $('#modal_confirm_plugin_refresh .btn-primary').click(function(e) {
        e.preventDefault();
        TB.configure.plugins.modal_refresh.reset();
        var that = this, $this = $(this);
        
        var url = $('#modal_confirm_plugin_refresh').attr('action'), type = $('#modal_confirm_plugin_refresh').attr('method');

        console.log('transmitting :', type, url);
        var jqxhr = $.ajax(url, {
            type : type,
            contentType : 'application/json',
            dataType : 'json',
            processData : false,
            beforeSend : TB.configure.plugins.modal_refresh.loading //toggle loading to on
        }).done(function(data) {
            console.log("success:", url);
            console.log("data:", data);
            
            if ($('#modal_confirm_plugin_refresh').data('customevents')) {
                jQuery.each($('#modal_confirm_plugin_refresh').data('customevents'), function(i, elem) {
                    $('#modal_confirm_plugin_refresh').trigger(elem, {})
                });
            }
            var event = jQuery.Event("modal_confirm_plugin_refresh_done");
            event.plugininfo = data;
            $('#modal_confirm_plugin_refresh').trigger(event);
            
            $('#modal_confirm_plugin_refresh #modal-success-messages').removeClass('hide').empty().append('<p>Successfully refreshed plugin information</p>');
            $('#modal_confirm_plugin_refresh #modal-plugin-information').removeClass('hide');
            $('#modal_confirm_plugin_refresh #modal-plugin-information pre code').text(JSON.stringify(data));
        }).fail(function(data) {
            TB.configure.plugins.modal_refresh.reset();
            $('#modal_confirm_plugin_refresh #modal-error-messages').empty();
            $('#modal_confirm_plugin_refresh #modal-error-messages').removeClass('hide');
            $('#modal_confirm_plugin_refresh #modal-error-messages').append('<p>Error</p><p>' + data.responseText + '</p>');
            console.log("error:", data);
        }).always(function(data) {
            TB.configure.plugins.modal_refresh.loading(); //toggle loading to off
        });
    });
    $('#modal_confirm_plugin_refresh').on('hidden', function() {
        $('body #modal_confirm_plugin_refresh').remove();
    });
};
