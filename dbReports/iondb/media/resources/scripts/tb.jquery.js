jQuery.extend({
    // Creates a full fledged settings object into target
    // with both ajaxSettings and settings fields.
    // If target is omitted, writes into ajaxSettings.
    _ajaxSetup:jQuery.ajaxSetup,
    ajaxSetup: function( target, settings ) {
            target = jQuery._ajaxSetup(target, settings); 
            if (jQuery.browser.msie && target.type === 'PATCH') {
                target.type = 'PROPPATCH';
            }
            return target;
       }
});

