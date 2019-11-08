$(document).ready(function(){
    
    $("#planName").on('keyup', function(){
        $("#error").html('');
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            //'Error, Plan Name should contain only numbers, letters, spaces, and the following: . - _'
            $("#error").html(gettext('workflow.step.saveplan.messages.validate.planName.is_valid_chars'));
        }
        //call the check max length function that's in validation.js
        if(!is_valid_length($(this).val(), 512)) {
            //'Error, Plan Name length should be 512 characters maximum'
            $("#error").html(gettext('workflow.step.saveplan.messages.validate.planName.is_valid_length'));
        }
    });

    $("#note").on('keyup', function(){
        $("#noteerror").html('');
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            //'Error, Notes should contain only numbers, letters, spaces, and the following: . - _'
            $("#noteerror").html(gettext('workflow.step.saveplan.messages.validate.note.is_valid_chars'));
        }
        //call the check max length function that's in validation.js
        if(!is_valid_length($(this).val(), 1024)) {
            //'Error, Notes length should be 1024 characters maximum'
            $("#noteerror").html(gettext('workflow.step.saveplan.messages.validate.note.is_valid_length'));
        } 
    });

});
