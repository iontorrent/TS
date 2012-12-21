$(function() {
    $('#browser-warning').remove();
    var nagTimes = 2;
    //Get the cookie value, if null set to nagTimes
    var warnedUnsupportedBrowser = $.cookie("warnedUnsupportedBrowser") || nagTimes;
    if (warnedUnsupportedBrowser > 0) {
        //load html via ajax
        $.get('/site_media/resources/html/browser-warning.html', function(data) {
            $('body').prepend(data);
            $('#browser-warning').slideDown('slow');
            $('#browser-warning .alert').alert();
            $('#browser-warning .alert').bind('closed', function() { //when alert is closed/dismissed
                //decrement 
                var value = warnedUnsupportedBrowser - 1;
                //create cookie
                $.cookie("warnedUnsupportedBrowser", value, {
                    path : '/'
                });
                TB.utilities.form.focus();
            });
        });
    }
});
