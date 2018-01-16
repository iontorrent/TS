$('#report_analyze_form').submit(function(e) {
        e.preventDefault();
        return false;
});

$('#submitButton').click(function(e){
    console.log("I have been clicked.");
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
    cmdline_args_display();
};

show_warning = function(){
    var warning = WARNINGS["all"] || "";
    var type = $('input[name=pipetype]:checked').val();
    
    if(type == "fromRaw"){
        warning += WARNINGS["sigproc"] || "";
    }else{
        if($('#id_do_thumbnail').is(':checked')){
            warning += WARNINGS[$("#id_previousThumbReport :selected").data("pk")] || "";
        }else{
            warning += WARNINGS[$("#id_previousReport :selected").data("pk")] || "";
        }
    }

    if(warning){
        $("#warning").html(warning).show();
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
$("[id^=id_reference]").change(function(){
    var reference = this.value;

    var region = $('#' + this.id.replace('reference','targetRegionBedFile'));
    var hotspot = $('#' + this.id.replace('reference','hotSpotRegionBedFile'));
    var select = region.children('option').add(hotspot.children('option'))

    select.filter('[value != ""]').hide();
    select.each(function(i,elem){
        var bedfile_path = elem.value;
        if( bedfile_path.split('/').indexOf(reference)>0 ){
            $(this).show();
        } else {
            $(this).prop('selected', false);
        }
    })
}).change();

// update barcoded references when default reference selection changes
$("#id_reference, #id_targetRegionBedFile, #id_hotSpotRegionBedFile, #useDefaultReference").change(function(){
    if ($("#useDefaultReference").is(':checked')){
        $("#barcodedReference_container tbody select[id*='reference']").val($("#id_reference").val());
        $("#barcodedReference_container tbody select[id*='reference']").change();
        $("#barcodedReference_container tbody select[id*='targetRegionBedFile']").val($("#id_targetRegionBedFile").val());
        $("#barcodedReference_container tbody select[id*='hotSpotRegionBedFile']").val($("#id_hotSpotRegionBedFile").val());
        _save_barcoded_reference = true;
    }
});

$("#useDefaultReference").change(function(){
    var disable = $(this).is(':checked');
    $('#barcodedReference_container tbody').find("select[name='reference']").prop('disabled', disable);
    $('#barcodedReference_container tbody').find("select[name='targetRegionBedFile']").prop('disabled', disable);
    $('#barcodedReference_container tbody').find("select[name='hotSpotRegionBedFile']").prop('disabled', disable);
});

$('#barcodedReference_container table').change(function(){
    _save_barcoded_reference = true;
});

// re-Analysis page doesn't support selecting barcoded samples, if user changes Barcode Set - disable barcoded references section
$('#id_barcodeKitName').change(function(){
    var barcodeSet = this.value;
    if (barcodeSet != _barcode_set){
        $('#barcodedReference_container #barcodedReference_div').hide();
        $('#barcodedReference_container #unsupported').addClass('unsupported').show();
        _save_barcoded_reference = true;
    } else {
        $('#barcodedReference_container #barcodedReference_div').show();
        $('#barcodedReference_container #unsupported').removeClass('unsupported').hide();
    }
});

get_barcodedReferences = function(){
    if (_save_barcoded_reference){
        var barcodedReferences = {};
        if ( !$('#barcodedReference_container #unsupported').hasClass('unsupported') ){
            $('#barcodedReference_div tbody tr').each(function(){
                var $cells = $(this).children();
                barcodedReferences[$cells.eq(0).text()] = {
                    'reference': $cells.find("select[id*='reference']").val(),
                    'targetRegionBedFile': $cells.find("select[id*='targetRegionBedFile']").val(),
                    'hotSpotRegionBedFile': $cells.find("select[id*='hotSpotRegionBedFile']").val()
                };
            });
        } else {
            barcodedReferences['unsupported']= true;
        }
        $('#id_barcodedReferences').val(JSON.stringify(barcodedReferences));
        console.log('barcoded references', barcodedReferences);
    }
    // remove name attribute to avoid submitting extra barcoded data with the form
    $('#barcodedReference_div tbody tr').find('select').removeAttr('name');
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

// ***************************** AnalysisArgs ********************************
cmdline_args_display = function(){
    var default_args = $('[name=args_choice]:checked').val() == "default";
    $('#cmdline_args').each(function(){
        $(this).find('input, textarea').attr("readonly", default_args);
        $(this).find('select').attr("disabled", default_args);
    });
}

$("#analysisargs_select").change(function(){
    var val = $(this).val();
    if (val){
        $('.args').each(function(){
            var name = $(this).attr('name').toLowerCase();
            $(this).val( ANALYSISARGS[val][name] );
        });
    }
});

$('[name=args_choice]').change(function(){
    if (this.value == 'default'){
        $('#id_custom_args').val('False');
        $("#analysisargs_select .best_match").prop('selected',true).change();
    }else{
        $('#id_custom_args').val('True');
        if ($('#cmdline_args').hasClass('out')){
            $('#cmdline_args').removeClass('out').addClass('in');
            $('.showargs').removeClass('icon-plus').addClass('icon-minus');
        }
    }
    cmdline_args_display();
});

$('#cmdline_args textarea').on('change keypress paste', function(){
    $("#analysisargs_select").val('');
});

$('.showargs').click(function(){
    $(this).toggleClass('icon-plus icon-minus');
});
