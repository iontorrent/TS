$(document).ready(function(){
    
    $("#planName").on('keyup', function(){
        $("#error").html('');
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            $("#error").html('Error, Plan Name should contain only numbers, letters, spaces, and the following: . - _');
        }
        //call the check max length function that's in validation.js
        if(!is_valid_length($(this).val(), 512)) {
            $("#error").html('Error, Plan Name length should be 512 characters maximum');   
        }
    });

    $("#note").on('keyup', function(){
        $("#noteerror").html('');
        //call the Regex test function identified in validation.js file
        if (!is_valid_chars($(this).val())) {
            $("#noteerror").html('Error, Notes should contain only numbers, letters, spaces, and the following: . - _');
        }
        //call the check max length function that's in validation.js
        if(!is_valid_length($(this).val(), 1024)) {
            $("#noteerror").html('Error, Notes length should be 1024 characters maximum');   
        } 
    });

});
