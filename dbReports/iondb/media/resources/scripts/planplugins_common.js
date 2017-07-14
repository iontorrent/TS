serializeIframe = function(iframe){
    var json_obj;
    if ($.isFunction(iframe[0].contentWindow.serializeForm)){
        json_obj = iframe[0].contentWindow.serializeForm();
    }else{
        json_obj = $(iframe[0].contentDocument.forms).serializeJSON();
    }
    return json_obj;
};

// load plugin configuration json into an iframe
updateIframe = function(iframe, href, json_obj){
    iframe.unbind();
    iframe.css("height", 0);
    iframe.hide();
    iframe.attr("src", href);
    enableIframeResizing(iframe);
    iframe.on("load", function () {
        iframe.show();
        if (json_obj) {
            if ($.isFunction(iframe[0].contentWindow.restoreForm)) {
                // use plugin's restoreForm function if exists
                iframe[0].contentWindow.restoreForm(json_obj);
            } else {
                // call generic form json restore
                $(iframe[0].contentDocument.forms).restoreJSON(json_obj);
                iframe[0].contentWindow.$(':input').trigger('change');
            }
        }
    });

};

// generic restore serialized form
$.fn.restoreJSON = function(data, showObsolete) {
    showObsolete = showObsolete || false;
    var els = $(this).find(':input').get();
    if(data && typeof data == 'object') {
        $.each(els, function() {
            if (this.name && data[this.name]) {
                if(this.type == 'checkbox' || this.type == 'radio') {
                    $(this).attr("checked", (data[this.name] == $(this).val()));
                } else if(showObsolete && this.type == 'select-one' && this.options.length < 2){
                    // add saved values if missing options
                    $(this).append('<option value="' + data[this.name] + '">' + data[this.name] + '</option>');
                    $(this).val(data[this.name]);
                } else {
                    $(this).val(data[this.name]);
                }
                $(this).change();
            } else if (this.type == 'checkbox') {
                $(this).attr("checked", false);
            }
        });
    }
    return $(this);
};
