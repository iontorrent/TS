TB.namespace('TB.plan.wizard');

/**
 * Add selected plugins to review div
 */
TB.plan.wizard.addPluginsToReview = function(plugins, uploaders) {
    plugins = plugins || [];
    uploaders = uploaders || [];
    $("#review_selectedPlugins").append(plugins.join(', '));
    $("#review_export").append(uploaders.join(', '));
};

/**
 * modal_plan_wizard's form submit handler
 */
TB.plan.wizard.submit = function(e) {
    var that = this;
    //workaround: #samples and #notes are not picked up by toObject()
    var samples = $("#samples_workaround").val();
    var notes = $("#notes_workaround").val();

    // validate project names for bad characters and length
    var newProjects = $('#newProjects').val();
    if(newProjects) {
        var newProjects_error = "";
        if (!newProjects.match(/^[a-zA-Z0-9\-_\., ]+$/)) {
            newProjects_error += 'Project names should contain only letters, numbers, dashes, and underscores.';
        }
        var projects = newProjects.split(',');
        for (var i = 0 ; i < projects.length; i++) {
            if (projects[i].replace(/^\s+|\s+$/g,'').length > 64) {
                newProjects_error += ' Project name length should be 64 characters maximum.';
            }
        }
        if (newProjects_error) {
            setTab('#ws-6');
            $('#ws-6_addProjects').effect("highlight", {"color" : "#F20C18"}, 2000);
            $('#newProjects-error').text(newProjects_error);
            return false;
        }
    }

    if ($("#barcodeName option:selected").text()) {
        var text = $("#barcodeName option:selected").text();
        var rows = $('table.ws-8_barcodedSampleTable tr');
        var bcSamples = [];
        $('table.ws-8_barcodedSampleTable tr.'+text+' input').each(function (index, input) {
            if (input.value) {
                bcSamples.push([input.name, input.value]);
            }
        });
        $('input[name=bcSamples_workaround]').val(bcSamples);
    }


    var json = $('#modal_plan_wizard #planTemplateWizard').serializeJSON(),
        url = $('#modal_plan_wizard #planTemplateWizard').attr('action'),
        type = $('#modal_plan_wizard #planTemplateWizard').attr('method');

    // Get IR config values for IR_1 and higher IR versions
    var irConfigList = [];
    var uploaders = $('input:checkbox[name=uploaders]:checked').each(function(index, uploader){
        uploader = $(uploader);
        uploader_name = uploader.val().split('|')[1];

        if (uploader_name === "IonReporterUploader_V1_0") {
            irConfigList.push({
                'Workflow': $('select[id="ir1_irWorkflow"]').val()
            });
        } else if (uploader_name.search("IonReporterUploader") >= 0) {
            var bcKit = $("#barcodeName option:selected").text();
            if (bcKit === "") {
                // non-barcoded + IR
                $('input[id^="sample_irSample_"]').each(function(i, elem){
                    if (elem.value){
                        irConfigList.push({
                        'sample': elem.value,
                        'Workflow': $('select[id="sample_irWorkflow_select_nn"]'.replace('nn',i+1)).val(),
                        'Relation': $('select[id="sample_irRelation_select_nn"]'.replace('nn',i+1)).val(),
                        'RelationRole': $('select[id="sample_irRelationRole_select_nn"]'.replace('nn',i+1)).val(),
                        'setid': $('input[id="sample_irSetId_select_nn"]'.replace('nn',i+1)).val()
                        });
                    }
                });
            } else {
                // barcoded + IR
                $('input[id^="bcSample_sample_BarcodeKit_"]'.replace('BarcodeKit',bcKit)).each(function(i, elem) {
                    if (elem.value) {
                        irConfigList.push({
                        'sample': elem.value,
                        'Workflow': $('select[id="bcSample_irWorkflow_select_BarcodeKit_nn"]'.replace('nn',i+1).replace('BarcodeKit',bcKit)).val(),
                        'Relation': $('select[id="bcSample_irRelation_select_BarcodeKit_nn"]'.replace('nn',i+1).replace('BarcodeKit',bcKit)).val(),
                        'RelationRole': $('select[id="bcSample_irRelationRole_select_BarcodeKit_nn"]'.replace('nn',i+1).replace('BarcodeKit',bcKit)).val(),
                        'setid': $('input[id="bcSample_irSetId_select_BarcodeKit_nn"]'.replace('nn',i+1).replace('BarcodeKit',bcKit)).val(),
                        'barcodeId': $('[id="bcSample_index_BarcodeKit_nn"]'.replace('nn',i+1).replace('BarcodeKit',bcKit)).text()
                        });
                    }
                });
            }
        }
    });
    json.irConfigList = irConfigList;

    // find selected plugins
    var selectedPlugins = [];
    $('input:checkbox[name=plugins]:checked').each(function() {
        var autorun = $(this)[0].getAttribute('data-autorun');
        if (autorun && autorun == 'auto'){
            return true;
        }
        var tokens = $(this).val().split("|");
        var userInput = "";
        if ($("#configure_plugin_"+ tokens[0]).length==1)
            userInput = JSON.parse($("#configure_plugin_"+tokens[0])[0].getAttribute('data-plugin_config'));

        selectedPlugins.push({
            "id": tokens[0],
            "name": tokens[1],
            "version": tokens[2],
            "userInput": userInput
        });
    });

    var selectedUploaders =[];
    $('input:checkbox[name=uploaders]:checked').each(function() {
        var autorun = $(this)[0].getAttribute('data-autorun');
        if (autorun && autorun == 'auto'){
            return true;
        }
        var tokens = $(this).val().split("|");
        selectedUploaders.push({
            "id": tokens[0],
            "name": tokens[1],
            "version": tokens[2]
        });
    });
    json.selectedPlugins = {"planplugins": selectedPlugins, "planuploaders": selectedUploaders};
    console.log(json);

    if (submitUrl) {
        url = submitUrl;
    }

    if ($('#modal_plan_wizard #planTemplateWizard').attr('method') == "POST") {
    }
    json = JSON.stringify(json);

    console.log('transmitting :', type, url, json );

    $.ajax({
        type: type,
        url: url,
        async: false,
        dataType: "json",
        contentType: "application/json",
        data: json
    }).done(function(msg) {
        var json = $.parseJSON(msg);
        if (msg.error) {
            apprise(msg.error);
            //$('#error-messages').empty();
            //$('#error-messages').append('<p class="error">ERROR: ' + msg.error + '</p>');
        } else {
            if ((INTENT == "Plan Run New") || (INTENT == "Plan Run")) {
                $('body #modal_plan_wizard').modal("hide");
                window.location = PLANNED_URL;
            } else {
                $('#modal_plan_wizard').trigger('modal_plan_wizard_done', {});
                $('body #modal_plan_wizard').modal("hide");
            }
        }
    }).fail(function(data) {
        apprise('Error saving Plan!');
    })
    .always(function(data) { });

    //always return false because the POSTing is done to the API.
    return false;
};

TB.plan.wizard.initialize = function() {
    $('#flows').spinner({ min: 0, max: 10000 });
    $('#qcValues_1').spinner({ min: 0, max: 100 });
    $('#qcValues_2').spinner({ min: 1, max: 100 });
    $('#qcValues_3').spinner({ min: 0, max: 100 });

    $('[name=bcIrSetId_select]').spinner({min: 0 });
    $('[name=irSetId_select]').spinner({min: 0 });

    //$('#ws-8').bind('beforeShow', function(e){

    //formdata = $('#modal_plan_wizard #planTemplateWizard').serializeJSON();
    //console.log($('#modal_plan_wizard #planTemplateWizard').serializeJSON());
    //var encodingTemplate = kendo.template($("#reviewWorkflowTemplate").text());
    //var presets = {}; //TODO: load the presets from {% url get_application_product_presets %}
    //$('#modal_plan_wizard #planTemplateWizard #review-workflow').html(encodingTemplate({data:formdata, presets:presets}));
    //});

    $('.workflow-menu li:first').next().addClass('next-tab');
    $('.workflow-menu a').click(function(e) {
        e.preventDefault();
        var $this = $(this), clicked = $this.attr('href');
        setTab(clicked);
        e.preventDefault();
    });

    $('.prev-button').click(function(e) {
        e.preventDefault();
        var $this = $(this), clicked = $this.attr('href');
        setTab(clicked);
        e.preventDefault();
    });

    $('.next-button').click(function(e) {
        e.preventDefault();
        var $this = $(this), clicked = $this.attr('href');
        setTab(clicked);
        e.preventDefault();
    });

    $(".submitPlanRun").click(function(e) {
        e.preventDefault();
        $("#submitIntent").val("savePlan");
        $('#planTemplateWizard').submit();
    });
    $(".submitSaveTemplate").click(function(e) {
        e.preventDefault();
        $("#submitIntent").val("saveTemplate");
        $('#planTemplateWizard').submit();
    });

    $(".submitUpdateTemplate").click(function(e) {
        e.preventDefault();
        $("#submitIntent").val("updateTemplate");
        $('#modal_plan_wizard #planTemplateWizard').submit();
    });

    $(".submitUpdatePlanRun").click(function(e) {
        e.preventDefault();$("#submitIntent").val("updatePlan");
        $('#planTemplateWizard').submit();
    });

    $('#modal_plan_wizard #planTemplateWizard').submit(TB.plan.wizard.submit);

    //both click and change works
    //radio button clicked for run mode
    $("input:radio[name=runMode]").click(function() {
        var runMode = $(this).val();
        var runType = $("input:radio[name=runType]:checked").val();

        console.log("runMode selection changed. runMode=",runMode, " runType=", runType);
        var applProductForRunType = TB.plan.wizard.getApplProduct(runType);
        if (applProductForRunType && runMode === 'pe') {
            $('#librarykitname option:selected', 'select').removeAttr('selected');
            $("#librarykitname option[value='" + applProductForRunType.libKitName + "']").attr('selected', 'selected');
            $('#sequencekitname option:selected', 'select').removeAttr('selected');
            $("#sequencekitname option[value='" + applProductForRunType.seqKitName + "']").attr('selected', 'selected');
        }

        if (runMode == "pe") {
            $("#review_runType").text("Paired-End");

            //PE libKey & adapter default selections
            $("#review_peForwardLibKey").text($("#peForwardLibraryKey").val());
            $("#review_peForward3Adapter").text($("#peForward3primeAdapter").val());
            $("#review_peReverseLibKey").text($("#reverselibrarykey").val());
            $("#review_peReverse3Adapter").text($("#reverse3primeAdapter").val());
            $("#review_peLibAdapter").text($("#pairedEndLibraryAdapterName").val());

            $('.extra_kit_forward_info').slideUp('fast');
            $('.extra_kit_pe_forward_info').slideDown('fast');
            $('.extra_kit_pe_reverse_info').slideDown('fast');

            $(".review_extra_kit_info").slideUp('fast');
            $(".review_extra_pe_kit_info").slideDown('fast');

            $('div.extra_kit_forward_info').hide();
            $('div.review_extra_kit_info').hide();
        } else {
            $("#review_runType").text("Fragment");

            //single libKey & adapter default selections
            $("#review_forwardLibKey").text($("#libraryKey").val());
            $("#review_forward3Adapter").text($("#forward3primeAdapter").val());

            $('.extra_kit_pe_forward_info').slideUp('fast');
            $('.extra_kit_pe_reverse_info').slideUp('fast');
            $('.extra_kit_forward_info').slideDown('fast');

            $(".review_extra_kit_info").slideDown('fast');
            $(".review_extra_pe_kit_info").slideUp('fast');

            $('div.extra_kit_pe_forward_info').hide();
            $('div.extra_kit_pe_reverse_info').hide();
            $('div.review_extra_pe_kit_info').hide();
        }

        console.log('libKitName:',applProductForRunType.libKitName, ' ', 'seqKitName:', applProductForRunType.seqKitName);

        //if user changes the run mode, flowCount will come from application product default.
        //if user selects a specific sequencing kit, flowCount will come from seqKit default

        //flowCount = $("input:hidden[name='"+applProductForRunType.seqKitName+"']").val();
        //console.log("#flows", flowCount);
        //$("#flows").val(flowCount);

        $("#flows").val(applProductForRunType.flowCount);
        console.log("#flows", $("#flows").val());

        $("#review_seqKit").text(applProductForRunType.seqKitName);
        $("#review_libKit").text(applProductForRunType.libKitName);
        $("#review_flowCount").text($("#flows").val());

    });


    //dropdown list selection change for sequencing kit
    $("#sequencekitname").change(function() {
        var value = $(this).val();
        var flowCount = $("input:hidden[name='" + value + "']").val();

        $("#flows").val(flowCount);

        var selectedFlowCount = $("#flows").val();
        $("#review_flowCount").text(selectedFlowCount);

        $("#review_seqKit").text($('#sequencekitname option:selected').val());

    });

    //flow count value change
    $('input[name=flows]').change(function() {
        //$("#flows").change(function() {
        var value = $(this).val();
        //console.log("flow count changed. value=", value);
        $("#review_flowCount").text(value);
    });

    //dropdown list selection change for chip type
    $("#chipType").change(function() {
        if ($("#chipType option:selected").text()) {
            var value = $("#chipType option:selected").text();
            //console.log("chip type changed. value=", value);
            $("#review_chipType").text(value);
        } else {
            $("#review_chipType").text("");
        }
    });


    //dropdown list selection change for template kit
    $("#templatekitname").change(function() {
        var value = $(this).val();
        $("#review_templateKit").text($('#templatekitname option:selected').val());

    });


    //dropdown list selection change for control sequence kit
    $("#controlsequence").change(function() {
        var value = $(this).val();
        $("#review_controlSeq").text($('#controlsequence option:selected').val());
    });

    //dropdown list selection change for forward library key
    $("#libraryKey").change(function() {
        var value = $(this).val();
        $("#review_forwardLibKey").text($('#libraryKey option:selected').val());

    });



    //dropdown list selection change for forward 3' adapter
    $("#forward3primeAdapter").change(function() {
        var value = $(this).val();
        $("#review_forward3Adapter").text($('#forward3primeAdapter option:selected').val());
    });

    //dropdown list selection change for pairedEnd forward library key
    $("#peForwardLibraryKey").change(function() {
        var value = $(this).val();
        $("#review_peForwardLibKey").text($('#peForwardLibraryKey option:selected').val());

    });

    //dropdown list selection change for pairedEnd forward 3' adapter
    $("#peForward3primeAdapter").change(function() {
        var value = $(this).val();
        $("#review_peForward3Adapter").text($('#peForward3primeAdapter option:selected').val());
    });

    //dropdown list selection change for pairedEnd reverse library key
    $("#reverselibrarykey").change(function() {
        var value = $(this).val();
        $("#review_peReverseLibKey").text($('#reverselibrarykey option:selected').val());

    });

    //dropdown list selection change for pairedEnd reverse 3' adapter
    $("#reverse3primeAdapter").change(function() {
        var value = $(this).val();
        $("#review_peReverse3Adapter").text($('#reverse3primeAdapter option:selected').val());
    });

    //dropdown list selection change for pairedEnd library adapter
    $("#pairedEndLibraryAdapterName").change(function() {
        var value = $(this).val();
        $("#review_peLibAdapter").text($('#pairedEndLibraryAdapterName option:selected').val());

    });


    //dropdown list selection change for sample prep kit
    $("#samplePrepKitName").change(function(){
        var value = $(this).val();
        $("#review_samplePrepKitName").text($('#samplePrepKitName option:selected').val());
    });


    //dropdown list selection change for barcode kit
    $("#barcodeName").change(function() {
        if ($("#barcodeName option:selected").text()) {
            var text = $("#barcodeName option:selected").text();

            $("#review_barcodeKit").text(text);
            $("#isBarcodedPlan").val("True");

            var rows = $('table.ws-8_barcodedSampleTable tr');
            console.log('rows=' + rows);
            rows.filter("." + text).show();
            rows.not("." + text).hide();
            rows.filter('.barcodedSampleTableHeader').show();
            //console.log("barcode kit selection changed: barcode selected");
            $.showAndHide_ws8();
        } else {
            //console.log("barcode kit selection changed: no barcode selected");
            $("#review_barcodeKit").text("");
            $("#isBarcodedPlan").val("False");
            $.showAndHide_ws8();
        }
    });

    //user input change for qc values
    $('#qcValues_1').change(function() {
        var value = $(this).val();
        $("#review_qcValues_1").text(value);
    });

    //user input change for qc values
    $('#qcValues_2').change(function() {
        var value = $(this).val();
        $("#review_qcValues_2").text(value);
    });

    //user input change for qc values
    $('#qcValues_3').change(function() {
        var value = $(this).val();
        $("#review_qcValues_3").text(value);
    });

    //dropdown list selection change for genome reference
    $("#library").change(function() {
        var value = $(this).val();
        $("#review_refLib").text($('#library option:selected').val());
    });

    //dropdown list selection change for target regions bed file
    $("#bedfile").change(function() {
        var value = $(this).val();
        $("#review_bedfile").text($('#bedfile option:selected').val());
    });

    //dropdown list selection change for hotspot regions bed file
    $("#regionfile").change(function() {
        var value = $(this).val();
        $("#review_regionfile").text($('#regionfile option:selected').val());
    });

    $(".configure_plugin").click(function(){
        //TODO: use jQuery.data() instead of get/setAttribute. 
        //TODO: use jQuery $('#plugin_iframe')  
        // opens plugin's plan.html in an iframe
        var plugin_pk = this.getAttribute('data-plugin_pk');
        var url = this.getAttribute('href');
        var iframe = document.getElementById('plugin_iframe');
        iframe.src = url;
        iframe.setAttribute('data-plugin_pk',plugin_pk);
        $(iframe).bind('load', function() {});
        $("#plugin_config").show();

        // restore saved configuration, if any
        var plugin_json_obj = JSON.parse($("#configure_plugin_"+plugin_pk)[0].getAttribute('data-plugin_config'));
        $(iframe).one("load", function(){
          if (plugin_json_obj !== null){
              console.log('calling restoreJson', plugin_json_obj);
              $(iframe.contentDocument.forms).restoreJSON(plugin_json_obj);
              iframe.contentWindow.$(':input').trigger('change')
          }
        });
    });

    $("#plugin_config_save").click(function(){
        var plugin_json = $($("#plugin_iframe")[0].contentDocument.forms).serializeJSON();
        plugin_json = JSON.stringify(plugin_json);
        var plugin_pk = $("#plugin_iframe")[0].getAttribute('data-plugin_pk');
        console.log(plugin_pk + ' plugin configuration', plugin_json);
        $("#configure_plugin_"+plugin_pk)[0].setAttribute('data-plugin_config',plugin_json);
        $("#plugin_config").hide();
    });

    $("#plugin_config_cancel").click(function(){
        $("#plugin_config").hide();
    });
    //when plugin selection changes
    $('input:checkbox[name=plugins]').click(function() {
        var autorun = $(this)[0].getAttribute('data-autorun');
        if (autorun) {
            // 2 states for autorun plugins: user selected and autorun selected, currently can't unselect
            if (autorun == 'auto') {
                $(this)[0].style.opacity = 1;
                $(this)[0].setAttribute('data-autorun', 'selected');
                $(this).data('tooltip').enabled = false;
            } else {
                $(this)[0].style.opacity = 0.3;
                $(this)[0].setAttribute('data-autorun', 'auto');
                $(this).data('tooltip').enabled = true;
            }
            $(this)[0].checked = true;
        }

        var pluginId = $(this).val().split("|")[0];
        var pluginName = $(this).val().split("|")[1];

        if ($(this).is(':checked')) {
            $("#review_selectedPlugins").append(pluginName + ",");
            // show plugin configuration
            $("#configure_plugin_"+pluginId).show();
            $("#configure_plugin_"+pluginId).click();
        } else {
            $("#review_selectedPlugins").text($("#review_selectedPlugins").text().replace(pluginName + ",", "").replace(/,,/g, ","));
            // hide configure button
            $("#configure_plugin_"+pluginId).hide();
            $("#plugin_config_cancel").click();
        }
    });


    //when uploader selection changes
    $('input:checkbox[name=uploaders]').click(function() {
        var autorun = $(this)[0].getAttribute('data-autorun');
        if (autorun) {
            // 2 states for autorun plugins: user selected and autorun selected, currently can't unselect
            if (autorun == 'auto') {
                $(this)[0].style.opacity = 1;
                $(this)[0].setAttribute('data-autorun', 'selected');
                $(this).data('tooltip').enabled = false;
            } else {
                $(this)[0].style.opacity = 0.3;
                $(this)[0].setAttribute('data-autorun', 'auto');
                $(this).data('tooltip').enabled = true;
            }
            $(this)[0].checked = true;
        }

        var uploaderName = $(this).val().split("|")[1];
        if ($(this).is(':checked')) {
            if (uploaderName.toLowerCase() == "ionreporteruploader_v1_0") {
                $('.uploader_input').each(function() {
                    var eachUploaderName = $(this).val().split("|")[1];
                    if ((eachUploaderName.toLowerCase() != "ionreporteruploader_v1_0") && (eachUploaderName.toLowerCase().search('ionreporteruploader') >= 0)) {
                        $(this).removeAttr("checked");
                        $("#review_export").text($("#review_export").text().replace(eachUploaderName + ",", "").replace(/,,/g, ","));
                    }
                });
                $("#review_export").append(uploaderName + ",");
                if (INTENT == "EditPlan" || INTENT == "Plan Run" || INTENT == "Plan Run New" || INTENT == "CopyPlan") {
                    $('.ir1_hideable_IRConfig').slideDown();
                    $.showAndHide_ws8();
                }
            } else if (uploaderName.toLowerCase().search('ionreporteruploader') >= 0) {
                $('.uploader_input').each(function() {
                    var eachUploaderName = $(this).val().split("|")[1];
                    if ((eachUploaderName.toLowerCase().search('ionreporteruploader_v1_0') >= 0)) {
                        $(this).removeAttr("checked");
                        $("#review_export").text($("#review_export").text().replace(eachUploaderName + ",", "").replace(/,,/g, ","));
                    }
                });
                $("#review_export").append(uploaderName + ",");
                if (INTENT == "EditPlan" || INTENT == "Plan Run" || INTENT == "Plan Run New" || INTENT == "CopyPlan") {
                    $('.ir1_hideable_IRConfig').slideUp();
                    $.showAndHide_ws8();
                }
            }
        } else {
            $("#review_export").text($("#review_export").text().replace(uploaderName + ",", "").replace(/,,/g, ","));
            //user unchecks IR v1.0
            if (uploaderName.toLowerCase() == "ionreporteruploader_v1_0") {
                $('.ir1_hideable_IRConfig').slideUp();
            } else {
                if (uploaderName.toLowerCase().search('ionreporteruploader') >= 0) {
                    //if only barcode kit is selected
                    if ($("#barcodeName option:selected").text()) {
                        //console.log("uploader selection changed: only barcode kit is selected!!");

                        $('#ws-8_nonBarcodedSamples').slideUp('fast');
                        $('#ws-8_nonBarcodedSamples_irConfig').slideUp('fast');

                        $('div.ws-8_nonBarcodedSamples').hide();
                        $('div.ws-8_nonBarcodedSamples_irConfig').hide();

                        $('.bcSample_hideable_IRConfig').hide();
                        $('#ws-8_barcodedSamples').children('.bcSample_hideable_basic').show();
                    } else {
                        //if neither irUploader nor barcode kit is selected
                        $('#ws-8_nonBarcodedSamples').slideDown('fast');
                        $('#ws-8_nonBarcodedSamples_irConfig').slideUp('fast');
                        $('#ws-8_barcodedSamples').slideUp('fast');

                        //$('div.ws-8_nonBarcodedSamples').show();
                        $('div.ws-8_nonBarcodedSamples_irConfig').hide();
                    }
                }
            }
        }
    });


    //when project selection changes
    $('input:checkbox[name=projects]').click(function() {
        var projectName = $(this).val().split("|")[1];
        if ($(this).is(':checked')) {
            $("#review_projects").append(projectName + ",");
        } else {
            var reviewText = $("#review_projects").text().replace(projectName + ",", "").replace(/,,/g, ",");
            $("#review_projects").text(reviewText);
        }
    });


    //user input change for manually entered projects
    $('#newProjects').bind('input propertychange', function() {
        var value = $(this).val();
        $("#review_projects2").text(value);
    });

    //projects filtering
    $("#projects_search_text").change(function(e) {
        $("input[name='projects']").parent().show();
        if ($(this).val()) {
            $("input[name=projects]").parent().not(":contains(" + $(this).val() + ")").hide();
        }
    });


    //both click and change works
    //radio button clicked for application type
    $("input:radio[name=runType]").click(function() {
        //TODO: Replace using JSON version of planTemplateData
        var runType = $(this).val();

        applProduct = TB.plan.wizard.getApplProduct(runType);

        if (applProduct) {
            if (INTENT == "New") {
                $('#modal_plan_wizard-title').text(INTENT + " " + applProduct.runTypeDescription + " Plan");
            }

            //The selector 'input[name=field]:eq(1)' means find all input elements with a name attribute equal to 'field' and
            //pick the second one (eq numbers from 0)
            if (applProduct.isDefaultPairedEnd == "True") {
                $("input:radio[name=runMode]:eq(1)").attr("checked", "checked");
            } else {
                $("input:radio[name=runMode]:eq(0)").attr("checked", "checked");
            }

            runMode = $("input:radio[name=runMode]").val();
            console.log("at application selection change. runMode=", runMode);

            if (runMode == "pe") {
                $("#review_runType").text("Paired-End");
            } else {
                $("#review_runType").text("Fragment");
            }

            if (applProduct.variantFrequency == "None" || applProduct.variantFrequency == "&quot;") {
                //alert("no variant frequency to set");
                $("input:radio[name='variantfrequency']:checked").val("");
            } else {
                $("input:radio[name='variantfrequency']:checked").val(applProduct.variantFrequency);
            }
            $('#librarykitname option:selected', 'select').removeAttr('selected');
            $("#librarykitname option[value='" + applProduct.libKitName + "']").attr('selected', 'selected');
            $('#sequencekitname option:selected', 'select').removeAttr('selected');
            $("#sequencekitname option[value='" + applProduct.seqKitName + "']").attr('selected', 'selected');

            //console.log("#flows", applProduct.flowCount);
            $("#flows").val(applProduct.flowCount);
            //console.log("#flows", $("#flows").val());
            $('#library option:selected', 'select').removeAttr('selected');
            $("#library option[value=" + applProduct.reference + "]").attr('selected', 'selected');
            $('#bedfile option:selected', 'select').removeAttr('selected');
            $("#bedfile option[value=" + applProduct.targetBedFile + "]").attr('selected', 'selected');
            $('#regionfile option:selected', 'select').removeAttr('selected');
            $("#regionfile option[value=" + applProduct.hotSpotBedFile + "]").attr('selected', 'selected');
            $('#chipType option:selected', 'select').removeAttr('selected');
            $("#chipType option[value='" + applProduct.chipType + "']").attr('selected', 'selected');

            $("#review_application").text(applProduct.runTypeDescription);

            console.log("application selection changed. libKit=", applProduct.libKitName, " seqKit=", applProduct.seqKitName);

            $("#review_libKit").text(applProduct.libKitName);
            $("#review_seqKit").text(applProduct.seqKitName);
            $("#review_flowCount").text(applProduct.flowCount);

            if (applProduct.chipType == "None" || applProduct.chipType == "&quot;") {
                $("#review_chipType").text("");
            } else {
                $("#review_chipType").text("Ion " + applProduct.chipType + "&trade; Chip");
            }

            $("#review_templateKit").text(applProduct.templateKitName);
            $("#review_controlSeq").text(applProduct.controlSeqName);
        }
    });


    //workaround: plan sample input change
    $("#samples").change(function() {
        var value = $(this).val();
        $("#samples_workaround").val(value);
    });

    //workaround: plan template name input change
    $("#notes").change(function() {
        var value = $(this).val();
        $("#notes_workaround").val(value);
    });



    $('.review_plan_inline').toggle(function() {
        $('.review_plan_info').slideDown();
        $(this).html('Review Run Plan &nbsp;&ndash;');
    }, function() {
        $('.review_plan_info').slideUp('fast');
        $(this).html('Review Run Plan &nbsp;+');
    });



    $('.add_project_inline').click(function() {
        $('.ws-6_addProjects').slideDown();
    });



    $('.extra_kit_inline').toggle(function() {
        //alert("extra button toggled 1");
        var runMode = $("input:radio[name=runMode]:checked").val();
        if (runMode == 'single') {
            $('.extra_kit_pe_forward_info').slideUp('fast');
            $('.extra_kit_pe_reverse_info').slideUp('fast');
            $('.extra_kit_forward_info').slideDown('fast');
        } else {
            $('.extra_kit_forward_info').slideUp('fast');
            $('.extra_kit_pe_forward_info').slideDown('fast');
            $('.extra_kit_pe_reverse_info').slideDown('fast');
        }
        $(this).html('Details &nbsp;&ndash;');

    }, function() {
        //alert("extra button toggled 2");
        $('.extra_kit_forward_info').slideUp('fast');
        $('.extra_kit_pe_forward_info').slideUp('fast');
        $('.extra_kit_pe_reverse_info').slideUp('fast');

        $(this).html('Details &nbsp;+');
    });



    //201207109-wip
    $(".goSaveAsPlanRun").click(function() {
        console.log("I AM AT .goSaveAsPlanRun... TBC");
    });

    $('#modal_plan_wizard').on('hidden', function() {
        $('body #modal_plan_wizard').remove();
    });



    //non-barcoded sample dropdown list selection change for ionReporter configuration
    $(".irWorkflow_select").change(function() {        
        var value = $(this).val();
        console.log('.irWorkflow_select change event handler called with', value);

        $(this).parent().next().find('select').find('option').remove();
        if (value === null || value === "") {
            $(this).parent().next().find('select').attr('disabled', 'disabled');
            //console.debug($(this).parent().nextAll().find('select').attr('name'));

            $(this).parent().nextAll().find('select').find('option').remove();
            $(this).parent().nextAll().find('select').find('option').attr('disabled', 'disabled');
            $(this).parent().nextAll().find('input').val('');
            $(this).parent().nextAll().find('input').attr('disabled', 'disabled');
        } else {
            $(this).parent().next().find('select').removeAttr('disabled');

            var option = '<option value=""></option>';
            $(this).parent().next().find('select').append(option);
            var _that = $(this);
            var relationChoices = $.getIRConfigRelation(value);
            $.each(relationChoices, function(i, relation){
                var option = '<option value="' + relation + '">' + relation + '</option>';
                _that.parent().next().find('select').append(option);
            });
        }
    });



    //non-barcoded sample dropdown list selection change for ionReporter configuration
    $(".irRelation_select").change(function() {
        var value = $(this).val();
        //  console.log(value);
        relationRoleChoices = $.getIRConfigRelationRole(value);
        isToDisableRelationRole = $.isToDisableIRConfigRelationRole(value);
        isToDisableSetId = $.isToDisableIRConfigSetId(value);

        if (isToDisableSetId === true) {
            $(this).parent().nextAll().find('input').val('');
            $(this).parent().nextAll().find('input').attr('disabled', 'disabled');
        } else {
            $(this).parent().next().nextAll().find('input').removeAttr('disabled');
        }

        $(this).parent().nextAll().find('select').find('option').remove();
        if (isToDisableRelationRole === true) {
            $(this).parent().nextAll().find('select').attr('disabled', 'disabled');
        } else {
            $(this).parent().nextAll().find('select').removeAttr('disabled');

            var option = '<option value=""></option>';
            $(this).parent().nextAll().find('select').append(option);
            var _that = $(this);
            $.each(relationRoleChoices, function(i, relation) {
                var option = '<option value="' + relation + '">' + relation + '</option>';
                _that.parent().nextAll().find('select').append(option);
            });
        }
    });


    //non-barcoded sample dropdown list selection change for ionReporter configuration
    $(".irRelationRole_select").change(function() {
        var value = $(this).val();
        //  console.log(value);
        isToDisable = $.isToDisableIRConfigSetId(value);

        if (isToDisable === true) {
            $(this).parent().nextAll().find('input').val('');
            $(this).parent().nextAll().find('input').attr('disabled', 'disabled');
        } else {
            $(this).parent().nextAll().find('input').removeAttr('disabled');
        }
    });

    $(function() {
        var source = '#modal_plan_wizard';
        $(document).bind('modal_confirm_plugin_refresh_done', function(e){
            console.log('!!!!!!!!!!!!!!!!here');
            //e.plugininfo contains the JSON equivalent to models.Plugin.info()
            $.refreshPluginDone(e.plugininfo);
        });
        $(source + " .refresh-uploader-information").click(function(e) {
            e.preventDefault();
            url = $(this).attr('href');
            $('body #modal_confirm_plugin_refresh').remove();
            $.get(url, function(data) {
                $('body').append(data);
                $("#modal_confirm_plugin_refresh").data('source', source);
                $("#modal_confirm_plugin_refresh").modal("show");
                return false;
            }).done(function(data) {
                console.log("success:", url);
            }).fail(function(data) {
                $('#error-messages').empty();
                $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
                console.log("error:", data);
            }).always(function(data) {/*console.log("complete:", data);*/
            });
        });
    });


    $(document).ready(function () {
        /** TS-4640: IE6,7,8 long text within fixed width <select> is clipped, set width:auto temporarily */
        TB.utilities.browser.selectAutoExpand();

        if (INTENT == "New" || INTENT == "Plan Run New") {
            $(".review_extra_kit_info").slideDown('fast');
            $(".review_extra_pe_kit_info").slideUp('fast');
            $('div.review_extra_pe_kit_info').hide();

            if (selectedApplProductData) {
                if (selectedApplProductData.isDefaultPairedEnd == "True") {
                    $("#review_peForwardLibKey").text($("#peForwardLibraryKey").val());
                    $("#review_peForward3Adapter").text($("#peForward3primeAdapter").val());
                    $("#review_peReverseLibKey").text($("#reverselibrarykey").val());
                    $("#review_peReverse3Adapter").text($("#reverse3primeAdapter").val());
                    $("#review_peLibAdapter").text($("#pairedEndLibraryAdapterName").val());

                    $(".review_extra_kit_info").slideUp('fast');
                    $(".review_extra_pe_kit_info").slideDown('fast');
                    $('div.review_extra_kit_info').hide();
                } else {
                    $("#review_forwardLibKey").text($("#libraryKey").val());
                    $("#review_forward3Adapter").text($("#forward3primeAdapter").val());

                    $(".review_extra_kit_info").slideDown('fast');
                    $(".review_extra_pe_kit_info").slideUp('fast');
                    $('div.review_extra_pe_kit_info').hide();
                }
            } else {
                //alert("NO selectedApplProductData at documentReady")
            }
        } else {
            if (selectedPlanTemplate && selectedPlanTemplate.runMode === "pe") {
                //if we no PE lib kits are active, peForwardLibKeys will be none and we'll hide the pe run mode
                if (planTemplateDataplanTemplateData.peForwardLibKeys === "None") {
                    apprise("Paired-end is not officially supported. Please activate paired-end kits in your database before proceeding with paired-end plan/template creation or edit.");
                }

                $(".review_extra_kit_info").slideUp('fast');
                $(".review_extra_pe_kit_info").slideDown('fast');
                $('div.review_extra_kit_info').hide();
            } else {
                $(".review_extra_kit_info").slideDown('fast');
                $(".review_extra_pe_kit_info").slideUp('fast');
                $('div.review_extra_pe_kit_info').hide();
            }
        }



        if (INTENT == "EditPlan" || INTENT == "Plan Run" || INTENT == "CopyPlan" ) {
        //init with previous sample and notes info
            if (selectedPlanTemplate) {
                $("#samples_workaround").val(selectedPlanTemplate.sampleDisplayedName);
                $("#notes_workaround").val(selectedPlanTemplate.notes);
            }
            if (selectedPlanTemplate && selectedPlanTemplate.barcodeId !== "") {
                var selectedBarcodeKit = selectedPlanTemplate.barcodeId;
            
                if (!jQuery.isEmptyObject(selectedPlanTemplate.barcodedSamples)) {
                    var bcSamples = selectedPlanTemplate.barcodedSamples;
                    
					          for (var sampleName in bcSamples) {
						            for (var i=0; i<bcSamples[sampleName]['barcodes'].length; i++){
							              var fieldName = "bcKey|" + selectedBarcodeKit + "|" + bcSamples[sampleName]['barcodes'][i];
							              $("input[name='"+fieldName+"']").val(sampleName); 
						            }
					          }
                }

                var rows = $('table.ws-8_barcodedSampleTable tr');
                var rowsToShow = rows.filter("."+selectedBarcodeKit+"");
                var rowsToHide = rows.not("."+selectedBarcodeKit+"");
                rowsToShow.show();
                rowsToHide.hide();
                rows.filter('.barcodedSampleTableHeader').show();
                $('div.ws-8_nonBarcodedSamples').hide();
            }
            else {
                $('div.ws-8_nonBarcodedSamples').show();
            }
        }

        // add selected plugins to review
        TB.plan.wizard.addPluginsToReview(planTemplateData.selectedPlugins, planTemplateData.selectedUploaders);

        // attach saved userInput to configurable plugins
        $(".configure_plugin").each(function(){
            var plugin_pk = this.getAttribute('data-plugin_pk');
            var userInput = planTemplateData.pluginUserInput[plugin_pk];
            if (userInput)
                this.setAttribute('data-plugin_config',userInput);
        });
        

        if (INTENT == "EditPlan" || INTENT == "Plan Run" || INTENT == "Plan Run New" || INTENT == "CopyPlan") {
            //hide irConfig columns if irReporter is not pre-selected
            if (isIR_v1_selected) {
                //console.log("document.ready going to SHOW ws-7 ir1_hideable_IRConfig");
                $('.ir1_hideable_IRConfig').slideDown();
            } else {
                //console.log("document.ready going to HIDE ws-7 ir1_hideable_IRConfig");
                $('.ir1_hideable_IRConfig').slideUp();
            }

            $.showAndHide_ws8();
            $.addIR1FormFields();
            $.addIRFormFields();

            // load previously saved IR selections
            var irUserSelection = planTemplateData.irConfigSaved;
            var irVersion = planTemplateData.irConfigSaved_version;
            var obj = jQuery.parseJSON(irUserSelection);
            if (obj === null){
                return;
            }
            // IR version 1.0
            if (irVersion < 1.2) {
                $('select[id="ir1_irWorkflow"]').val(obj[0].Workflow);
                return;
            }

            if (selectedPlanTemplate && selectedPlanTemplate.barcodeId === "") {
                // non-barcoded case
                obj = obj[0];
                $('input[id="sample_irSample_1"]').val(obj.sample);
                $.setIR_select('select[id="sample_irWorkflow_select_1"]', obj.Workflow);
                $.setIR_select('select[id="sample_irRelation_select_1"]', obj.Relation);
                $.setIR_select('select[id="sample_irRelationRole_select_1"]', obj.RelationRole);
                $('input[id="sample_irSetId_select_1"]').val(obj.setid.split('__')[0]);
            } else {
                //barcoded case
                var count = 0;
                $('input[id^="bcSample_sample_BarcodeKit_"]'.replace('BarcodeKit', selectedPlanTemplate.barcodeId)).each(function(i, elem) {
                    if (elem.value) {
                        $.setIR_select('select[id="bcSample_irWorkflow_select_BarcodeKit_nn"]'.replace('nn', i + 1).replace('BarcodeKit', selectedPlanTemplate.barcodeId), obj[count].Workflow);
                        $.setIR_select('select[id="bcSample_irRelation_select_BarcodeKit_nn"]'.replace('nn', i + 1).replace('BarcodeKit', selectedPlanTemplate.barcodeId), obj[count].Relation);
                        $.setIR_select('select[id="bcSample_irRelationRole_select_BarcodeKit_nn"]'.replace('nn', i + 1).replace('BarcodeKit', selectedPlanTemplate.barcodeId), obj[count].RelationRole);
                        $('input[id="bcSample_irSetId_select_BarcodeKit_nn"]'.replace('nn', i + 1).replace('BarcodeKit', selectedPlanTemplate.barcodeId)).val(obj[count].setid.split('__')[0]);
                        count++;
                    }
                });
            }
        }
    });

};

(function($) {
    $.irConfigSelection_1 = null; //JSON set at modal load time
    $.irConfigSelection = null; //JSON set at modal load time

    $.getIRConfigSelection_1 = function(){
        return $.irConfigSelection_1;
    };

    $.getIRConfigSelection = function() {
        return $.irConfigSelection;
    };

    $.getIRConfigApplication = function(theSelectedWorkflow) {
        var selectedApplication = '';
        if (theSelectedWorkflow === null || theSelectedWorkflow === '') {
            return selectedApplication;
        }
        var obj = $.getIRConfigSelection();
        if (obj !== null) {
            $.each(obj["column-map"], function() {
                //console.log(key + ' ' + value);
                var application = this.ApplicationType || '';
                var workflow = this.Workflow || '';
                if (workflow == theSelectedWorkflow) {
                    selectedApplication = application;
                }
            });
        } else {
            console.log("obj is null at getIRConfigApplication...");
        }
        return selectedApplication;
    };

    $.getIRConfigRelation = function(theSelectedWorkflow) {
        var selectedApplication = $.getIRConfigApplication(theSelectedWorkflow);

        var selectedValidValues = [];
        if (selectedApplication === null || selectedApplication === '') {
            return selectedValidValues;
        }

        var obj = $.getIRConfigSelection();
        var validValues = [];

        if (obj !== null) {
            $.each(obj.columns, function() {
                var field = this.Name || '';
                var values = this.Values || [];
                if (field === "RelationshipType") {
                    validValues = validValues.concat(values);
                }
            });
            $.each(obj.restrictionRules, function() {
                var _for = this.For || null;
                var _valid = this.Valid || null;
                if (_for && _valid) {
                    if (_for.Name === "ApplicationType" && _for.Value === selectedApplication) {
                        var values = _valid.Values || [];
                        validValues = [].concat(_valid.Values);
                    }
                }
            });
            selectedValidValues = validValues;
        } else {
            console.log("obj is null at getIRConfigRelation...");
        }
        return selectedValidValues;
    };

    $.getIRConfigRelationRole = function(theSelectedRelation) {
        var selectedValidValues = [];
        if (theSelectedRelation === null || theSelectedRelation === '') {
            return selectedValidValues;
        }

        var obj = $.getIRConfigSelection();
        var validValues = [];

        if (obj !== null) {
            $.each(obj.columns, function() {
                var field = this.Name || '';
                var values = this.Values || [];
                if (field === "Relation") {
                    validValues = validValues.concat(values);
                }
            });
            $.each(obj.restrictionRules, function() {
                var _for = this.For || null;
                var _valid = this.Valid || null;
                if (_for && _valid) {
                    if (_for.Name === "RelationshipType" && _for.Value === theSelectedRelation) {
                        var values = _valid.Values || [];
                        validValues = [].concat(values);
                    }
                }
            });
            selectedValidValues = validValues;
        } else {
            //console.log("obj is null at getIRConfigRelationRole....");
        }
        return selectedValidValues;
    };

    $.isToDisableIRConfigRelationRole = function(theSelectedRelation) {
        return $.isToDisableIRConfigAttribute(theSelectedRelation, 'Relation');
    };

    $.isToDisableIRConfigSetId = function(theSelectedRelation) {
        return $.isToDisableIRConfigAttribute(theSelectedRelation, 'SetID');
    };

    $.isToDisableIRConfigAttribute = function(theSelectedRelation, theDisabledName) {
        var isToDisable = false;
        if (theSelectedRelation === null || theSelectedRelation === '' || theDisabledName === null || theDisabledName === '') {
            isToDisable = true;
            return isToDisable;
        }

        var obj = $.getIRConfigSelection();

        var isToDisableName = '';
        var forName = '';
        var forValue = '';

        if (obj !== null) {
            $.each(obj.restrictionRules, function() {
                var _for = this.For || null;
                var _disabled = this.Disabled || null;
                if (_for && _disabled) {
                    if (_disabled.Name === theDisabledName && _for.Value === theSelectedRelation) {
                        isToDisable = true;
                    }
                }
            });
        } else {
            //console.log("obj is null at isToDisableIRConfigAttribute....");
        }
        return isToDisable;
    };

    $.showAndHide_ws8 = function() {
        if ($("#barcodeName option:selected").text()) {
            //if both barcode kit and IonReporter both selected
            //var uploaders = $(“#review_export”).text().toLowerCase();
            if (($('#review_export').text().toLowerCase().search('ionreporteruploader_v1_0') < 0) && ($('#review_export').text().toLowerCase().search('ionreporteruploader') >= 0)) {
                $('#ws-8-barcoded-refresh-uploader-information').removeClass('hide');
                //console.log("showAndHide_ws8: both barcodeKit and uploader are selected");
                $('#ws-8_nonBarcodedSamples').slideUp('fast');
                $('#ws-8_nonBarcodedSamples_irConfig').slideUp('fast');

                $('#ws-8_barcodedSamples').children('.bcSample_hideable_IRConfig').slideDown('fast');
                //need-this
                $('#ws-8_barcodedSamples').slideDown('fast');
                //cause recursion issue
                //$('div.ws-8_barcodedSamples').show();
                $("#ws-8_barcodedSampleTable tr").each(function() {
                    $(this).find(".bcSample_hideable_IRConfig").css("display", "");
                });

                $('div.ws-8_nonBarcodedSamples').hide();
                $('div.ws-8_nonBarcodedSamples_irConfig').hide();
            } else {
                $('#ws-8-barcoded-refresh-uploader-information').addClass('hide');
                //console.log("showAndHide_ws8: only barcodeKit is selected AND NOT (IR v.10 not selected + IR v1.2 selected) ");
                //if only barcode kit is selected
                $('#ws-8_nonBarcodedSamples').slideUp('fast');
                $('#ws-8_nonBarcodedSamples_irConfig').slideUp('fast');
                //$('#ws-8_barcodedSamples').slideDown('fast');
                $('div.ws-8_nonBarcodedSamples').hide();
                $('div.ws-8_nonBarcodedSamples_irConfig').hide();
                $('#ws-8_barcodedSamples').slideDown('fast');

                $("#ws-8_barcodedSampleTable tr").each(function() {
                    $(this).find(".bcSample_hideable_IRConfig").css("display", "none");
                });
            }
        } else {
            //if only IonReporter is selected
            if (($('#review_export').text().toLowerCase().search('ionreporteruploader_v1_0') < 0) && ($('#review_export').text().toLowerCase().search('ionreporteruploader') >= 0)) {
                $('#ws-8-barcoded-refresh-uploader-information').addClass('hide');
                //console.log("showAndHide_ws8: only IR v1.0 not selected + IR v1.2 selected");
                $('#ws-8_nonBarcodedSamples').slideUp('fast');
                $('#ws-8_nonBarcodedSamples_irConfig').slideDown('fast');

                $('#ws-8_barcodedSamples').slideUp('fast');

                $('div.ws-8_nonBarcodedSamples').hide();
                $('ws-8_barcodedSamples').hide();
            } else {
                //console.log("showAndHide_ws8: neither barcode kit nor IonReporter is selected");

                //if neither barcode kit nor IonReporter is selected
                $('#ws-8_nonBarcodedSamples').slideDown('fast');
                $('#ws-8_nonBarcodedSamples_irConfig').slideUp('fast');
                $('#ws-8_barcodedSamples').slideUp('fast');

                $('div.ws-8_barcodedSamples').hide();
                $('div.ws-8_nonBarcodedSamples_irConfig').hide();
                //$('div.ws-8_nonBarcodedSamples').show();
            }
        }
    };

    $.addIR1FormFields = function() {
        console.log('called addIR1FormFields');

        var obj = $.getIRConfigSelection_1();
        var columnCounter = 0;

        if (obj !== null && obj.columns != null) {
            $.each(obj.columns, function() {
                var field = this.Name || '';
                var values = this.Values || [];
                if (field === "Workflow") {
                    var selectedWorkflow = $('[name=ir1_irWorkflow_select] :selected').val();
                    $('[name=ir1_irWorkflow_select]').empty().append('<option value=""></option>');
                    $.each(values, function(i, value){
                        var option = '<option value="' + value + '">' + value + '</option>';
                        $('[name=ir1_irWorkflow_select]').append(option);
                    });
                    $('[name=ir1_irWorkflow_select] option').filter(function(){return $(this).text() == selectedWorkflow; }).attr('selected', true);
                }
            });
        }
    };

    $.addIRFormFields = function() {
        console.log('called addIRFormFields');
        var obj = $.getIRConfigSelection();
        var columnCounter = 0;
        
        if (obj !== null && obj.columns != null) {
            $.each(obj.columns, function() {
                var field = this.Name || '';
                var values = this.Values || [];
                if (field === "Workflow") {
                    $("select[name='irWorkflow_select']").empty().append('<option value=""></option>');
                    $("[name='bcIrWorkflow_select']").empty().append('<option value=""></option>');
                    $.each(values, function(i, value){
                        var option = '<option value="' + value + '">' + value + '</option>';
                        $("select[name='irWorkflow_select']").append(option);
                        $("[name='bcIrWorkflow_select']").append(option);
                    });
                }
            });
        }
    };

    /**
     * IR version 1.2 or higher
     */
    $.setIR_select = function (element, value) {
        if ($(element) !== null && $(element).length > 0 && value != null) {
            if ($(element)[0].options.length > 1){
                $(element).val(value);
                $(element).change();
            }
            else{
                // display saved IRconfig values even if workflows didn't load
                $(element).append('<option value="' + value + '">' + value + '</option>');
                $(element).val(value);
            }
        }
    };

    $.refreshPluginDone = function(plugininfo) {
        console.log('called $.refreshPluginDone');
        plugininfo = jQuery.parseJSON(JSON.stringify(plugininfo));
        // console.log(plugininfo);
        if (plugininfo === null) {
            console.log('$.refreshPluginDone bailing');
            return;
        }
        console.log(plugininfo.name);
        if (plugininfo.name.toLowerCase().search('ionreporteruploader_v1_0') === 0) {
            console.log('$.refreshPluginDone updating IR1 form fields');
            $.irConfigSelection_1 = plugininfo.config;
            $.addIR1FormFields(plugininfo.config);
        } else if (plugininfo.name.toLowerCase().search('ionreporteruploader_v') === 0) {
            console.log('$.refreshPluginDone updating IR form fields');
            console.log("Found [name='irWorkflow_select']:" + $("[name='irWorkflow_select']").exists());
            $.irConfigSelection = plugininfo.config;
            $.addIRFormFields(plugininfo.config);
        }
    };

    // restore serialized form
    $.fn.restoreJSON = function(data) {
        var els = $(this).find(':input').get();
        if(data && typeof data == 'object') {
            $.each(els, function() {
                if (this.name && data[this.name]) {
                    if(this.type == 'checkbox' || this.type == 'radio') {
                        $(this).attr("checked", (data[this.name] == $(this).val()));
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

})(jQuery);
