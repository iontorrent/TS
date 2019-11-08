$(function() {//DOM ready handler

    jQuery.fn.uniform.language.required = gettext('uni-form-validation.language.required');
    var $form = $('#contacts_form');

    $form.uniform({
        holder_class : 'control-group',
        msg_selector : 'p.help-block.error',
        error_class : 'alert alert-error'
    });

    jQuery.fn.uniform.isValid($form, $form.uniform.defaults);
    //validation form and show errors on page ready

    $('#contacts_form_reset').click(function(e) {
        e.preventDefault();
        $('#contacts_form')[0].reset();
    });

    $('.save_button').click(function(e) {
        e.preventDefault();
        $('#contacts_form').submit();
    });

    $form.submit(function(e) {
        //prevents default form submission behavior
        e.preventDefault();

        var $this = $(this);
        console.log($this);

        if (!jQuery.fn.uniform.isValid($this, $this.uniform.defaults)) {
            return false;
        }

        var formdata = $('#contacts_form').serialize();
        var url = $('#contacts_form').attr('action'), type = $('#contacts_form').attr('method');

        console.debug(url, type, formdata);

        $.ajax({
            url : url,
            type : type,
            dataType : "html",
            data : formdata
        });
    });
});
