$('#report_analyze_form').submit(function(e) {
        e.preventDefault();
        return false;
});

$('#submitButton').click(function(e){
    e.preventDefault();
    
    get_barcodedReferences();
    formData = $("#report_analyze_form").serialize();
    formData += "&re-analysis=on"; // a way to distinguish user input from crawler post
    
    URL = $("#report_analyze_form").attr('action');
    METHOD = $("#report_analyze_form").attr('method');
    
    $('#modal_report_analyze_started').remove();
    $('.errorlist, #start_error').remove();
    $('#report_analyze_form .alert.alert-error').removeClass("alert alert-error");
    
    $.ajax({
        type : METHOD,
        url : URL,
        async : false,
        dataType : "html",
        data : formData,
        success : function(data, textStatus, jqXHR) {
            var found = $(data).find('#modal_report_analyze_started')
            if (found.exists()) {
                // reanalysis successfully started
                $('body').append(data);
                $('#modal_report_analyze_started').modal("show");
                $('#modal_report_analyze_started').on('hidden', function(){
                    $('body #modal_report_analyze_started').parent().remove();
                    window.location.reload(true);
                });
                analysis_live_ready_cb();
            } else {
                // display errors for form elements
                var formErrors = $(data).find('#report_analyze_form .errorlist');
                formErrors.each(function(){
                    var elem = $('#report_analyze_form').find('#'+ $(this).siblings(':input').attr('id'));
                    elem.parent().append($(this));
                    elem.closest('.control-group').addClass("alert alert-error");
                    $('a[href=#' +elem.closest('.tab-pane').attr('id')+ ']').tab('show');
                });
                
                $(data).find('#start_error').prependTo('#report_analyze_form');
            }
            return true;
        },
        error : function(jqXHR, status, error) {
            $('#error-messages').empty().append(error).show()
        }
    });
    //always return false because the POSTing is done to the API.
    return false;
});

change_pipetype = function () {
    
    if ($('#id_do_thumbnail').is(':checked')){
        $('.thumb').show();
        $('.full').hide();
    } else {
        $('.thumb').hide();
        $('.full').show();
    }

    var type = $('input[name=pipetype]:checked').val();

    $('.fromWells').each(function(){
        $(this).find('input, textarea').attr("readonly", type != "fromWells");
        $(this).find('select').attr("disabled", type != "fromWells");
        $(this).css("opacity", type == "fromWells"? 1: 0.2);
    });
    
    $('.fromRaw').each(function(){
        $(this).find('input, textarea').attr("readonly", type != "fromRaw");
        $(this).find('select').attr("disabled", type != "fromRaw");
        $(this).css("opacity", type == "fromRaw"? 1: 0.2);
    });
    
    $("#id_blockArgs").val(type);
    show_warning();
};

show_warning = function(){
    var warning = "";
    var type = $('input[name=pipetype]:checked').val();
    
    if(type == "fromRaw"){
        warning = WARNINGS["sigproc"];
    }else{
        if($('#id_do_thumbnail').is(':checked')){
            warning = WARNINGS[$("#id_previousThumbReport :selected").data("pk")];
        }else{
            warning = WARNINGS[$("#id_previousReport :selected").data("pk")];
        }
    }

    if(warning){
        $("#warning").text(warning).show();
    }else{
        $("#warning").hide();
    }
};

$("#id_previousReport, #id_previousThumbReport").change(show_warning);
$("input[name=pipetype]").click(change_pipetype);

$("#id_do_thumbnail").click(function(){
    if (! $(this).is(':checked') ){
        // by default Proton (on-instrument) fullchip runs start from basecalling
        $("#fromWells").prop('checked',true);
    } else {
        $("#fromRaw").prop('checked',true);
    }
    change_pipetype();
});


$("#id_do_base_recal").change(function(){
    var selected_recal_mode = $('select[name="do_base_recal"]').val()
    console.log("at analyze.html.js id_do_base_recal.change selectedValue=", selected_recal_mode);
    
    var checked = true;
    if (selected_recal_mode == "no_recal") {
        checked = false;
    }
    $('.recalib').each(function(){
        $(this).find('input, textarea').attr("readonly", !checked);
        $(this).css("opacity", checked? 1: 0.2);
    });
}).change();

// ***************************** Reference ******************************

// match available BED files to reference
$("#id_reference").change(function(){
    var reference = this.value;
    var select = $("#id_targetRegionBedFile option, #id_hotSpotRegionBedFile option");
    select.filter('[value != ""]').hide();
    select.each(function(i,elem){
        var bedfile_path = elem.value;
        if( bedfile_path.split('/').indexOf(reference)>0 )
            $(this).show();
    })
}).change();

// update barcoded references when reference selection changes
$("#id_reference").change(function(){
    if ($("#useBarcodedReference").length == 1){
        var reference = this.value;
        $('#barcodedReference_container tbody tr').each(function(){
            $(this).find("select[name='reference']").val(reference);
        });
        _save_barcoded_reference = true;
    }
});

// re-Analysis page doesn't support selecting barcoded samples, if user changes Barcode Set - disable barcoded references section
$('#id_barcodeKitName').change(function(){
    var barcodeSet = this.value;
    if ($("#useBarcodedReference").length == 1){
        if (barcodeSet != _barcode_set){
            $('#barcodedReference_container table').hide();
            $('#barcodedReference_container div')
                .append("<div id='unsupported'> Barcoded References are not supported when Barcode Set value is changed. Please use run Edit page to assign samples to new barcodes.</div>");
        } else {
            $('#barcodedReference_container table').show();
            $('#barcodedReference_container div').remove("#unsupported");
        }
    }
});

// Barcoded references
$("#useBarcodedReference").click(function(){
    _save_barcoded_reference = true;
    if ( $(this).is(':checked') ){ $('#barcodedReference_container').show(); }else{$('#barcodedReference_container').hide();}
});

get_barcodedReferences = function(){
    if (_save_barcoded_reference && $('#barcodedReference_container div #unsupported').length==0){
        var barcodedReferences = {};
        $('#barcodedReference_container tbody tr').each(function(){
            var $cells = $(this).children();
            barcodedReferences[$cells.eq(0).text()] = {
                'reference': $cells.find('select').eq(0).val(),
                'nucType': $cells.find('select').eq(1).val()
            };
        });
        $('#id_barcodedReferences').val(JSON.stringify(barcodedReferences));
    }
    $('#barcodedReference_container tbody tr').find('select').removeAttr('name');
}

// ***************************** PLUGINS *********************************

$('#plugins_select_all').click(function(){
    $('input:checkbox[name=plugins]').each(function(){
        if (! $(this).prop('disabled')){
            $(this).prop('checked', true);
            $("#configure_plugin_"+this.value).show();
        }
    });
});

$('#plugins_clear_selection').click(function(){
    $('input:checkbox[name=plugins]').each(function(){
        $(this).prop('checked', false);
        $("#configure_plugin_"+this.value).hide();
    });
});

$('input:checkbox[name=plugins]').click(function() {
    pluginId = this.value
    if (this.checked){
        $("#configure_plugin_"+pluginId).show();
        $("#configure_plugin_"+pluginId).click();
    } else {
        $("#configure_plugin_"+pluginId).hide();
        $("#plugin_config_cancel").click();
    }
});

$(".configure_plugin").click(function(){
// plugin configure button clicked, opens plugin's plan.html in an iframe
    var pluginId = this.getAttribute('data-plugin_pk');
    var iframe = $('#plugin_iframe');
    
    // first check whether an iframe for another plugin is already open and save data
    if (iframe.attr('src'))
        save_plugin_config();

    iframe.attr('data-plugin_pk',pluginId);
    var src = "/configure/plugins/plugin/XXX/configure/plan/".replace('XXX',pluginId);
    $("#plugin_config").show();
    
    // restore saved configuration, if any
    var pluginUserInput = JSON.parse($('#id_pluginsUserInput').val());
    var plugin_json_obj = pluginUserInput[pluginId];
    updateIframe(iframe, src, plugin_json_obj);
});

save_plugin_config = function(){
    var iframe = $('#plugin_iframe');
    var plugin_json_obj = serializeIframe(iframe);
    var pluginId = iframe.attr('data-plugin_pk');
    console.log('save plugin', pluginId + ' configuration', plugin_json_obj);
    
    var pluginUserInput = JSON.parse($('#id_pluginsUserInput').val());
    pluginUserInput[pluginId] = plugin_json_obj;
    $('#id_pluginsUserInput').val(JSON.stringify(pluginUserInput))        
}

$("#plugin_config_save").click(function(){
    save_plugin_config();
    $("#plugin_config").hide();
    $("#plugin_iframe").attr("src", "");
});

$("#plugin_config_cancel").click(function(){
    $("#plugin_config").hide();
    $("#plugin_iframe").attr("src", "");
});
