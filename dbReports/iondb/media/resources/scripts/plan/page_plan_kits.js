
$(document).ready(function () {
    //there are 8 ways flow count can be automatically set based on the
    // 1 - sequencing kit
    // 2 - templating kit
    // 3 - templating size
    // 4 - read length
    // 5 - chipType
    // 6 - applProduct for the application and target technique
    // 7 - applProduct for the application, target technique and sequencing instrument
    // 8 - categorized applProduct for the application, target technique and categories
    var categorizedApplProductInUse = null;
    var isInit = true;

    $("form").submit(function(){
        $("select[name=barcodeId]").prop('disabled', false);
        $('.advanced').each(function(){ $(this).prop('disabled',false); });
    });

    $("#base_recalibrate").each( function (){
        var i =0;
        var sel = this;
        for(i = 0; i < sel.length;i++){
            if (sel.options[i].text == "Default Calibration") {
                sel.options[i].title = "A random sample of reads (up to 100,000 by default) is aligned and used to determine calibration parameters, which are then applied to the rest of the run";
            }
            else if (sel.options[i].text == "Enable Calibration Standard") {
                sel.options[i].title = "Select Enable Calibration Standard when the experiment does not include a reference BAM file.  Only choose this option if you have added Calibration Standards with the library for AmpMix preparation.";
            }
        }
       }
    );

    function init_protocol_n_readLength_visibility() {
        var templateKit = templateKits[$("#templateKit").val()];
        //console.log("templateKit=", templateKit);
        if (templateKit) {
            var categories = templateKit.categories;
            if (categories.toLowerCase().indexOf("multiplereadlength") >= 0) {
                $('.library_read_length_info').hide();
                $('.read_length_info').show();
            }
            else if (categories.toLowerCase().indexOf("supportlibraryreadlength") >= 0) {
                $('.library_read_length_info').show();
                $('.read_length_info').hide();
            }
            else {
                $('.library_read_length_info').hide();
                $('.read_length_info').hide();
           }
           handleTemplateKitSelectionForProtocol();
           update_templating_size_warning()
        }
    }

    function handleTemplateKitSelectionForLibraryReadLength() {
        $('.library_read_length_info').hide();
        var templateKit = templateKits[$("#templateKit").val()];

        if (templateKit && templateKit.categories.toLowerCase().indexOf("supportlibraryreadlength") >= 0) {
            updateLibraryReadLength();
            $('.library_read_length_info').show();
        }
    }

    function updateLibraryReadLength(){
        var defaultLibraryReadLength = 0;
        var templateKit = templateKits[$("#templateKit").val()];

        if (templateKit && templateKit.categories.toLowerCase().indexOf("supportlibraryreadlength") >= 0) {
            defaultLibraryReadLength = templateKit.libraryReadLength;

            var sequenceKit = sequencingKits[$("#sequenceKit").val()];
            if (sequenceKit && sequenceKit.categories.toLowerCase().indexOf("readlengthderivablefromflows") >= 0){

                // set Library Read Length from Seq Kit flows instead
                var seqKitFlowCount = sequenceKit.flowCount;
                if (categorizedApplProductInUse && sequenceKit.value === categorizedApplProducts[categorizedApplProductInUse].defaultSequencingKit) {
                    seqKitFlowCount = categorizedApplProducts[categorizedApplProductInUse].defaultFlowCount;
                }
                if (seqKitFlowCount > 0) defaultLibraryReadLength = seqKitFlowCount / 2.2 | 0;
            } else {
                setFlowCountByLibraryReadLength(defaultLibraryReadLength);
            }
        }
        $('input[name = "libraryReadLength"]').val(defaultLibraryReadLength);
        updateSummaryPanel("#selectedLibraryReadLength", $('input[name = "libraryReadLength"]').val());
    }

    //for some templating kits, templating size cannot be used to drive UI behavior or db persistence.  Need to use read length instead
    function handleTemplateKitSelectionForReadLength() {
        $('input[name = "readLength"]').prop('checked', false);
        $('.read_length_info').hide();

        var defaultReadLength = "";
        var templateKit = templateKits[$("#templateKit").val()];

        if (templateKit) {
            var categories = templateKit.categories;

            if (categories.toLowerCase().indexOf("multiplereadlength") >= 0) {
                // define available Read length radio buttons
                var allowed_readLengths = "200;400";
                if (allowed_readLengths) {
                    $('input[name = "readLength"]').parent().hide();

                    jQuery.each(allowed_readLengths.split(";"), function(index, item) {
                        $('input[name = "readLength"][value="' + item + '"]').parent().show();
                        if (index == 0) {
                            $('input[name = "readLength"][value="' + item + '"]').prop('checked', true);
                            defaultReadLength = item;
                        }
                    });
                    $('.read_length_info').show();
                    updateSummaryPanel("#selectedLibraryReadLength", defaultReadLength);
                }
            }
        }
    }

    function handleTemplateKitSelectionForAutoSeqKitSelection() {
        var templateKit = templateKits[$("#templateKit").val()];

        if (templateKit) {
            var categories = templateKit.categories;
            var categories_lower = categories.toLowerCase();

            if (categories_lower.indexOf("s5") >= 0) {
                //if sequencing kit's categories is same or is subset of templating kit's categories, we found our entry
                $.each(categories_to_seq_kits_map , function(key, value) {
                    if (categories_lower.indexOf(key) >= 0) {
                        //alert( key + ": " + value );
                        seqKitName = value;
                        $("#sequenceKit").val(seqKitName);
                        handleSequencingKitSelection(seqKitName);
                    }
                });
            }
        }
    }

    function handleTemplateKitSelectionForProtocol() {
        categorizedApplProductInUse = null;
        var templateKit = templateKits[$("#templateKit").val()]

        if (templateKit && ($('input[name="templatekitType"]:checked').val() == "IonChef")) {
            var kitCategories = templateKit.categories;

            if (kitCategories.toLowerCase().indexOf("sampleprepprotocol") >= 0) {
                var categorizedApplProduct = getCategorizedApplProduct(kitCategories);
                if (categorizedApplProduct) {
                    update_samplePrepProtocol_select(kitCategories);
                    if (!_isEditRun && !isInit){
                        //try to auto-select the appropriate samplePrepProtocol
                        handleAutoSelectionByKitCategories(categorizedApplProduct, kitCategories);
                    }
                    return;
                }
                else{
                    update_samplePrepProtocol_select(kitCategories, "noCategorizedApplProduct");
                    return;
                }

            }
        }
        $('#samplePrepProtocol').empty().change();
    }

    function update_samplePrepProtocol_select(kitCategories, noCategorizedApplProduct){

    	//be able to filter by seq_instruments as well
        var filters = get_filters("");

        filters['categories'] = kitCategories;

        // TS-14664 remove empty templating protocol choice for myeloid plan
        var includeEmpty = true;
        if (_planCategories && _planCategories.indexOf('chef_myeloid_protocol') >= 0){
            includeEmpty = false;
        }

        var noCategorizedApplProduct = arguments[1];
        if (noCategorizedApplProduct && kitCategories.indexOf("pcr200_400bp") >= 0){
            filters['excludeProtocol'] = "true";
            includeEmpty = false;
        }

        filter_select_dropdown_multi_tokens(samplePrepProtocols, filters, "#samplePrepProtocol", includeEmpty);
    }

    function handleAutoSelectionByKitCategories(categorizedApplProduct, kitCategories) {

        if (categorizedApplProduct) {
            categorizedApplProductInUse = categorizedApplProduct;
            var categorizedDefaults = categorizedApplProducts[categorizedApplProduct];

            //do not auto-select protocol but user should be able to select it in the dropdown
            //setAdvancedSettingsSelection("samplePrepProtocol", categorizedDefaults.defaultSamplePrepProtocol);

            if (categorizedDefaults.defaultFlowOrder && $("#flowOrder").val() === "") {
                setAdvancedSettingsSelection("flowOrder", categorizedDefaults.defaultFlowOrder);
            }

            if (categorizedDefaults.defaultSequencingKit) {
                $("#sequenceKit").val(categorizedDefaults.defaultSequencingKit);
                $("#sequenceKit").change();

                if (categorizedDefaults.defaultFlowCount) {
                    updateFlows(categorizedDefaults.defaultFlowCount);
                    set_samplePrepProtocol_from_category_rules();
                }
            }
        }
        else {
            categorizedApplProductInUse = null;
            if (kitCategories){
                update_samplePrepProtocol_select(kitCategories);
            }
            setAdvancedSettingsSelection("samplePrepProtocol", "");
        }
    }

    function handleTemplateKitTypeChange(selectedTemplateKitType, selectedTemplateKit) {
        $('input[name="templatekitType"][value='+ selectedTemplateKitType +']').prop('checked', true);
        $("#templateKit_comment").hide();
        $("#templateKit").show();

        // update dropdown options
        update_TemplateKit_select();
        update_SequencingKit_select()

        if (selectedTemplateKit) {
            $("#templateKit").val(selectedTemplateKit);
            $("#templateKit").change();
        }
    }

    function handleSeqKitSelectionForFlowOrder(selectedSeqKitName) {
        var defaultFlowOrder = "";

        if (categorizedApplProductInUse && categorizedApplProducts[categorizedApplProductInUse].defaultSequencingKit == selectedSeqKitName && categorizedApplProducts[categorizedApplProductInUse].defaultFlowOrder) {
            defaultFlowOrder = categorizedApplProducts[categorizedApplProductInUse].defaultFlowOrder;
        }
        else {
            var sequenceKit = sequencingKits[selectedSeqKitName];
            if (sequenceKit && sequenceKit.defaultFlowOrder != null) {
                defaultFlowOrder = sequenceKit.defaultFlowOrder;
            } else {
                defaultFlowOrder = "";
            }
        }
        setAdvancedSettingsSelection("flowOrder", defaultFlowOrder);
    }

    function handleSequencingKitSelection(selectedSequenceKitName) {
        if (!_isEditRun && !isInit) {
            setFlowCountBySelectedKits()
            updateLibraryReadLength();
            handleSeqKitSelectionForFlowOrder(selectedSequenceKitName);
        }

        updateSummaryPanel("#selectedSequenceKit", seqKitNameToDesc[selectedSequenceKitName]);
    }

    function updateSummaryPanel(el, value){
        if (typeof value == 'undefined') value = '';
        if (el=="#selectedLibraryReadLength" && value<=0) value = "--";
        $(el).html(value);
    }

    function updateFlows(value) {
        if (!_isEditRun && !isInit && value > 0){
            $('input[name = "flows"]').val(value);
            updateSummaryPanel("#selectedFlows", value);
        }
    }

    function setFlowCountBySelectedKits(){

        if (_isEditRun || isInit) return;

        var templateKit = templateKits[$("#templateKit").val()];
        var sequenceKit = sequencingKits[$("#sequenceKit").val()];
        var templateKitFlowCount = templateKit ? templateKit.flowCount : 0;
        var sequenceKitFlowCount = sequenceKit ? sequenceKit.flowCount : 0;

        var flowCount = 0;
        if (categorizedApplProductInUse && sequenceKit.value === categorizedApplProducts[categorizedApplProductInUse].defaultSequencingKit && categorizedApplProducts[categorizedApplProductInUse].defaultFlowCount) {
            flowCount = categorizedApplProducts[categorizedApplProductInUse].defaultFlowCount;
        }
        else {
            if (!templateKit){
                flowCount = sequenceKitFlowCount;
            } else if (!sequenceKit){
                flowCount = templateKitFlowCount;
            } else {
                // both kits selected: use templateKit flows if flowoverridable, otherwise use sequenceKit flows
                var categories = sequenceKit.categories;
                if ((categories.toLowerCase().indexOf("flowoverridable") >= 0) && (templateKitFlowCount > 0)){
                    flowCount = templateKitFlowCount;
                } else {
                    flowCount = sequenceKitFlowCount;
                }
            }
        }
        updateFlows(flowCount);
    }

    function setFlowCountByLibraryReadLength(libraryReadLength) {
        if (categorizedApplProductInUse) return;

        var sequenceKit = sequencingKits[$("#sequenceKit").val()];
        if (sequenceKit) {
            var categories = sequenceKit.categories;
            if (categories.toLowerCase().indexOf("flowsderivablefromreadlength") >= 0){
                updateFlows((libraryReadLength * 2.2) | 0);
            }
        }
    }

    // Set number of flows based on category in selected Template Kit
    function set_default_flows_from_category_rules(){
        var templateKit = templateKits[$("#templateKit").val()];
        if (templateKit && templateKit.categories){
            var flowCount = null;
            // values_selected fields must correspond to rules defined in KitInfo._category_flowCount_rules
            var values_selected = {
                'samplePrepProtocol': $('select#samplePrepProtocol option:checked').val(),
                'readLength': $('[name=readLength]:checked').val(),
                'chipType': $('#chipType').val(),
            }

            $.each(defaultFlowsFromCategoryRules, function(i, rule){
                if (templateKit.categories.toLowerCase().indexOf(rule.category.toLowerCase()) >= 0){
                    for (field in values_selected) {
                        if (field in rule && values_selected[field] == rule[field]){
                            flowCount = rule.flowCount;
                        }
                    }
                }
            });

            //even for categorizedApplProductInUse, rule still applies for templating/flowCount relationship
            /*
            if (categorizedApplProductInUse) {
                var categorizedDefaults = categorizedApplProducts[categorizedApplProductInUse];

                if (categorizedDefaults.defaultTemplateKit && categorizedDefaults.defaultTemplateKit === templateKit.value &&
                    categorizedDefaults.defaultSequencingKit && categorizedDefaults.defaultSequencingKit === $("#sequenceKit").val()) {
                    //it is already set, no more changes needed
                    return;
                }
            }
            */

            if (flowCount) updateFlows(flowCount);
        }
    }

    // Set samplePrep protocol as defined in the business rule based on number of flows selected
    function set_samplePrepProtocol_from_category_rules() {
        var templateKit = templateKits[$("#templateKit").val()];
        if (templateKit && templateKit.categories){
            // values_selected fields must correspond to rules defined in KitInfo._category_flowCount_rules
            var values_selected = {
                'flowCount': $("#flows").val()
            }

            $.each(defaultFlowsFromCategoryRules, function(i, rule){
                if (templateKit.categories.toLowerCase().indexOf(rule.category.toLowerCase()) >= 0){
                    for (field in values_selected) {
                        if (field in rule && values_selected[field] == rule[field]){
                            samplePrepProtocol = rule.samplePrepProtocol;
                            setAdvancedSettingsSelection("samplePrepProtocol", samplePrepProtocol);
                        }
                    }
                }
            });
        }
    }

    function update_barcodeKit_dropdown() {
        $('input[name = "isBarcodeKitRequired"]').val("false");
        var value_list = barcode_kits_all_list;
        var libraryKit = libraryKits[$("#libraryKitType").val()];
        var templateKit = templateKits[$("#templateKit").val()];

        if (templateKit && categorizedApplProductInUse){
            var categorizedDefaults = categorizedApplProducts[categorizedApplProductInUse];
            if (categorizedDefaults.isBarcodeKitSelectionRequired == "True"){
                $('#barcodeKitLabel').text("Barcode Set (required)");
                $('input[name = "isBarcodeKitRequired"]').val("true");
            }
        }
        if (libraryKit) {
            isBarcodeKitRequired = $("#isBarcodeKitRequired").val() || "";
            if (!isBarcodeKitRequired || isBarcodeKitRequired == "false"){
                if ("{{helper.getApplProduct.isBarcodeKitSelectionRequired }}" == "True"){
                    $('#barcodeKitLabel').text("Barcode Set (required)");
                    $('input[name = "isBarcodeKitRequired"]').val("true");
                }
                else if (libraryKit.categories.toLowerCase().indexOf("bcrequired") >= 0) {
                    $('#barcodeKitLabel').text("Barcode Set (required)");
                    $('input[name = "isBarcodeKitRequired"]').val("true");
                }
                else{
                    $('#barcodeKitLabel').text("Barcode Set (optional)");
                    $('input[name = "isBarcodeKitRequired"]').val("false");
                }

            }
            if (libraryKit.categories.toLowerCase().indexOf("bcshowsubset") >= 0){
                value_list = barcode_kits_subset_list;
            }
        }

        var $selects = $("#barcodeId");
        var selected_barcodeId = $("#barcodeId").val();
        $selects.empty();
        $selects.append($("<option></option>"));

        //loop through the barcode kits
        $.each(value_list, function(i) {
            var value = value_list[i];

            var $opt = $("<option></option>");
            if (value == selected_barcodeId) {
                $opt.attr("selected","selected");
            }
            $opt.attr('value', value);
            $opt.text(value);
            $selects.append($opt);
        });
    }

    function getApplProduct() {
        var selectedInstrumentType = $("#instrumentType").val();
        var applProduct = applProductToInstrumentType[selectedInstrumentType.toLowerCase()];
        console.log(">>> selectedInstrumentType=", selectedInstrumentType, "; applProduct=", applProduct);
        return applProduct;
    }

    function getCategorizedApplProduct(kitCategories) {
        var tokens = kitCategories.split(";");
        for (var index in tokens) {
            var category = tokens[index].toLowerCase();

            //console.log("getCategorizedApplProduct() tokens=", category, "; applProductToCategories=", applProductToCategories);
            if (category in applProductToCategories) {
                var applProduct = applProductToCategories[category];

                //applProduct must match instrument type as well
                var selectedInstrumentType = $("#instrumentType").val();

                //instrumentType may not be set when creating plan from template and jumping to the kits chevron right away
                if (!selectedInstrumentType && $("#chipType").val()){
                    var chip = chips[$("#chipType").val()];
                    $('#instrumentType').val(chip ? chip['seq_instrument'] : "");
                }
                selectedInstrumentType = $("#instrumentType").val();

                if (selectedInstrumentType && selectedInstrumentType.toLowerCase() === categorizedApplProducts[applProduct].instrumentType.toLowerCase()) {
                    return applProduct;
                }
                else {
                    return null;
                }
            }
        }
        return null;
    }

    function handleAutoSelectionByApplProduct(selectedInstrumentType) {
        //get the applProduct for selected instrumentType
        var applProduct = getApplProduct();
        if (!_isEditRun && !isInit && applProduct) {
            var defaults = applProductDefaults[applProduct];

            selectedLibKitValue = $("#libraryKitType").val();
            if (!selectedLibKitValue && defaults.defaultLibraryKit){
                $('#libraryKitType option[value="' + defaults.defaultLibraryKit + '"]').attr('selected', 'selected');
                $('#libraryKitType').change();

            }

            selectedTemplKit = $("#templateKit").val();
            if (!selectedTemplKit){
                // select default Template Kit Type and trigger change to update Kit dropdown options and selected value
                var defaultTemplateKitType = defaults.defaultTemplateKitType || $('input[name="templatekitType"]:checked').val();
                $('input[name="templatekitType"][value='+ defaultTemplateKitType +']').prop('checked', true);
                $('input[name="templatekitType"][value='+ defaultTemplateKitType +']').change();
            }

            selectedSeqKit = $("#sequenceKit").val();
            if (!selectedSeqKit && defaults.defaultSequencingKit) {
                $('#sequenceKit option[value="'+defaults.defaultSequencingKit+'"]').attr('selected', 'selected');
                $('#sequenceKit').change();
            }
        }
    }


    function handleTemplateKitSelectionForThreePrimeadapter(){
        var templateKit = templateKits[$("#templateKit").val()];

        if (templateKit && templateKit.defaultThreePrimeAdapter) {
            var selectedThreePrimeAdapter = $('[name=forward3primeAdapter]').val();
            if (selectedThreePrimeAdapter != templateKit.defaultThreePrimeAdapter){
                // save previously selected value
                $('[name=forward3primeAdapter]').data('previous_selection',selectedThreePrimeAdapter);
                setAdvancedSettingsSelection('forward3primeAdapter', templateKit.defaultThreePrimeAdapter);
            }
        } else {
            // restore 3' adapter selection if it was auto set based on kit default but now the kit changed again
            var previous_selection = $('[name=forward3primeAdapter]').data('previous_selection');
            if (selectedThreePrimeAdapter != previous_selection){
                setAdvancedSettingsSelection('forward3primeAdapter', previous_selection);
            }
        }
    }

    $("#sequenceKit").on('change', function(){
        handleSequencingKitSelection($(this).val());
    });

    $("#samplePreparationKit").change(function()  {
        updateSummaryPanel("#selectedSamplePreparationKit", $(this).val());
    });


    $("#libraryKitType").change(function()  {
        //update barcode kit selection list on library kit selection changed
        var libKitName = $(this).val();
        //console.log("at libraryKitType.change() libKitName=", libKitName);

        updateSummaryPanel("#selectedLibraryKitType", libKitNameToDesc[libKitName]);

        categorizedApplProductInUse = null;
        var libraryKit = libraryKits[libKitName];

        //if user traverses from chevron to chevron, we should not trigger auto-selection when user retuns to the Kits chevron
        if (libraryKit && !isInit) {
            if (libraryKit.categories.toLowerCase().indexOf("sampleprepprotocol") >= 0) {
                var kitCategories = libraryKit.categories;
                var categorizedApplProduct = getCategorizedApplProduct(kitCategories);

                //if user now selects a library kit with no filter category, need to reset kit dropdowns
                if (!_isEditRun) {
                    update_TemplateKit_select();
                    update_SequencingKit_select();
                    if (categorizedApplProduct) {
                        categorizedApplProductInUse = categorizedApplProduct;
                        var categorizedDefaults = categorizedApplProducts[categorizedApplProduct];

                        // select default Template Kit Type and trigger change to update Kit dropdown options and selected value
                        var defaultTemplateKitType = categorizedDefaults.defaultTemplateKitType;
                        if (defaultTemplateKitType) {
                            handleTemplateKitTypeChange(defaultTemplateKitType, categorizedDefaults.defaultTemplateKit);
                        }
                    }
                }
            }
            else if (libraryKit.categories.toLowerCase().indexOf("filter_") >= 0) {
                var kitCategories = libraryKit.categories;

                if (!_isEditRun) {
                    update_TemplateKit_select_by_category_filter(kitCategories);
                    update_SequencingKit_select_by_category_filter(kitCategories);
                }
            }
            else {
                //if user now selects a library kit with no filter category, need to reset kit dropdowns
                if (!_isEditRun) {
                    update_TemplateKit_select();
                    update_SequencingKit_select();
                }
                //if categorizedApplProduct exists, need to reset samplePrepProtocol value
                if (categorizedApplProduct) {
                    setAdvancedSettingsSelection("samplePrepProtocol", "");
                }
            }
        }
        update_barcodeKit_dropdown()

    }).change();


    $('input[name="templatekitType"]').change(function () {
        $("#templateKit_comment").hide();
        $("#templateKit").show();

        // update dropdown options
        update_TemplateKit_select();
        update_SequencingKit_select()

        // retrieve previously selected kit or default if any
        var templateKit = $(this).data('templatekitname') || '';
        if (!templateKit || $("#templateKit option[value='"+ templateKit +"']").length == 0){
            var defaults = applProductDefaults[getApplProduct()];
            if (defaults){
                templateKit = ($(this).val() == 'OneTouch') ? defaults.defaultTemplateKit : defaults.defaultIonChefPrepKit;
            }
        }
        // default to auto-selecting if only one db entry is available
        var entryCount = $("#templateKit option").length;
        if (!templateKit || (entryCount > 0 && entryCount <= 2)) {
             templateKit = $("#templateKit option").eq(entryCount - 1).val();
        }

        $("#templateKit").val(templateKit).change();
    });


    $("#templateKit").change(function()  {
        categorizedApplProductInUse = null;

        var selectedVal = $(this).find("option:selected").val();

        // save kit value to restore if type selection changes
        if (selectedVal) $('input[name="templatekitType"]:checked').data('templatekitname', selectedVal);

        setFlowCountBySelectedKits();
        handleTemplateKitSelectionForLibraryReadLength();
        handleTemplateKitSelectionForReadLength();
        handleTemplateKitSelectionForAutoSeqKitSelection();
        handleTemplateKitSelectionForProtocol();
        set_default_flows_from_category_rules();
        handleTemplateKitSelectionForThreePrimeadapter();
        update_barcodeKit_dropdown();
        update_templating_size_warning();
        updateSummaryPanel("#selectedTemplatingKitName", templateKitNameToDesc[selectedVal]);
    });


    $("#libraryKey").change(function() {
        updateSummaryPanel("#selectedLibraryKey", libraryKeySeqToNameAndSeq[$(this).val()]);
    });


    $("#forward3primeAdapter").change(function() {
        var value = $(this).val();

        updateSummaryPanel("#selected3PrimeAdapter", threePrimeAdapterSeqToNameAndSeq[value]);
    });

    $("#flowOrder").change(function() {
        var value = $(this).val();

        var displayedValue = flowOrderToDescAndFlowOrder[value];
        if (displayedValue) {
            updateSummaryPanel("#selectedFlowOrder", displayedValue);
        }
        else {
            updateSummaryPanel("#selectedFlowOrder", "Use Instrument Default");
        }
    });


    $("#samplePrepProtocol").change(function() {
        var value = $(this).val();

        var displayedValue = samplePrepProtocolToDisplayedValue[value];
        if (displayedValue) {
            updateSummaryPanel("#selectedSamplePrepProtocol", displayedValue);
        }
        else {
            updateSummaryPanel("#selectedSamplePrepProtocol", "Use Instrument Default");
        }
        set_default_flows_from_category_rules()
    });

    $("#base_recalibrate").change(function() {
        var value = $(this).val();
        var displayedValue = baseRecalibrationModesValueToName[value];
        updateSummaryPanel("#selectedBaseCalibrationMode", displayedValue);
    });

    var flowsSpinner = $("#flows").spinner({min: 1, max: 2000});
    if (_isEditRun){
        flowsSpinner.spinner("disable").prop('disabled',false);
    } else {
        $("#flows").on("spinstop spinchange", function(event, ui){
            updateSummaryPanel("#selectedFlows", flowsSpinner.spinner("value"));
        });
    }


    var readLengthSpinner = $("#libraryReadLength").spinner({min: 0, max: 1000});
    if (_isEditRun){
        readLengthSpinner.spinner("disable").prop('disabled',false);
    } else {
        $("#libraryReadLength").on("spinstop spinchange", function(event, ui){
            var value = readLengthSpinner.spinner("value");
            updateSummaryPanel("#selectedLibraryReadLength", value);
            setFlowCountByLibraryReadLength(value);
        });
    }

    $('input[type=radio][name=readLength]').change(function() {
        updateSummaryPanel("#selectedLibraryReadLength", this.value);

        // TS-11339,  TS-11340 set number of flows based on read length selection
        set_default_flows_from_category_rules();
    });

    $("#controlsequence").change(function()  {
        updateSummaryPanel("#selectedControlSequence", $(this).val());
    });

    $("#chipType").change(function()  {
        if (!$('#instrumentType').val()){
            var chip = chips[$(this).val()];
            $('#instrumentType').val(chip ? chip['seq_instrument'] : "");
            $('#instrumentType').change();
        } else {
            update_LibraryKit_select();
            update_TemplateKit_select();
            update_SequencingKit_select();
        }
        set_default_flows_from_category_rules();
        show_chiptype_warning($(this).val());
        updateSummaryPanel("#selectedChipType", chipNameToDisplayName[$(this).val()]);
    });

    $("#instrumentType").change(function()  {
        update_ChipType_select();
        update_LibraryKit_select();
        update_TemplateKit_select();
        update_SequencingKit_select();
        //toggleFlowOrder();

        handleAutoSelectionByApplProduct($(this).val());
        updateSummaryPanel("#selectedInstrumentName",$(this).find('option:selected').html());
    });

    $("#barcodeId").change(function()  {
        updateSummaryPanel("#selectedBarcode", $(this).val());
    });

    init_protocol_n_readLength_visibility();


    $('#isDuplicateReads').change(function(){
        if($(this).is(':checked')) {
            $('#selectedMarkAsPcrDuplicates').text('True');
        } else {
            $('#selectedMarkAsPcrDuplicates').text('False');
        }

    });

    $('#realign').change(function(){
        if($(this).is(':checked')) {
            $('#selectedEnableRealignment').text('True');
        } else {
            $('#selectedEnableRealignment').text('False');
        }

    });

    // generate filtered dropdowns
    if ($('#chipType').val()){
        var chip = chips[$('#chipType').val()];
        $('#instrumentType').val(chip ? chip['seq_instrument'] : "");
        show_chiptype_warning($('#chipType').val());
    }
    update_ChipType_select();
    update_LibraryKit_select();
    update_TemplateKit_select();
    update_SequencingKit_select();

    // Advanced Settings
    $('[name=advancedSettingsChoice]').change(function(){
        if (isCustomKitSettings()){
            if (!$(".hideable_advanced_settings_section").is(':visible')){
                $(".showhide").click();
            }
            $('.advanced').prop('disabled', false);
        } else {
            $('.advanced').each(function(){
                for (id in _defaultAdvancedSettings){
                    setAdvancedSettingsSelection(id, _defaultAdvancedSettings[id]);
                }
                $(this).siblings('p.alert').hide();
                $(this).prop('disabled', true);
            });
        }
    });

    $('.advanced').on('store_advanced_settings', function(){
        // this runs on page load
        $(this).data('previousVal', $(this).val() || "");
        if (isCustomKitSettings()){
            recommend_advanced_settings(this);
        }
    }).change(function(){
        var previous = $(this).data('previousVal');
        var current = $(this).val() || "";

        if (isCustomKitSettings()){
            recommend_advanced_settings(this);
        } else {
            if ( (previous || current) && (previous != current) ){
                $('#show_updated').show();
                setTimeout(function(){ $('#show_updated').hide(); }, 4000);
            }
        }
        $(this).data('previousVal', current);
    }).trigger('store_advanced_settings');

    function recommend_advanced_settings(el){
        var name = $(el).attr('name');
        if (name in _defaultAdvancedSettings){
            var current = $(el).val() || "";
            var default_value = _defaultAdvancedSettings[name];
            if (default_value === null) default_value = "";
    
            if ( default_value != current ){
                var label = $(el).siblings('label').text();
                var text = "Recommended " + $.trim(label) + ": ";
    
                if ( $(el).find('option[value='+ default_value +']').length > 0 ){
                    text += $(el).find('option[value='+ default_value +']').text();
                } else {
                    text += default_value;
                }
                $(el).siblings('p.alert').text(text).show();
            } else {
                $(el).siblings('p.alert').hide();
            }
        }
    }
    
    function setAdvancedSettingsSelection(id, value) {
        _defaultAdvancedSettings[id] = value || "";
        if (!isCustomKitSettings()){
            $("#"+id).val(value);
        }
        $("#"+id).change();
    }
    
    isInit = false;
});

// ******************* Filters ************************* //

function filter_select_dropdown(data, filters, selectId){
    // re-create dropdown options after applying filters
    // filter logic is: value == filter OR !value
    // multiple values must use ";" as separator, e.g. "RNA;AMPS_RNA"
    var filtered_options = $.map(data, function(option){
        for (filterKey in filters){
            if (option[filterKey] && option[filterKey] != "None"){
                var match_values = option[filterKey].split(';');
                if (match_values.indexOf(filters[filterKey]) < 0){
                    //console.log('filtered', selectId, filters[filterKey], match_values, option.display)
                    return;
                }
            }
        }
        return ({ "index": option.index, "value": option.value, "display": option.display });
    });
    // make sure to keep original option order
    filtered_options.sort(function(a,b){ return a.index > b.index? 1: -1;})
    
    create_filter_select_dropdown(filtered_options, selectId);
}

function filter_select_dropdown_multi_tokens(data, filters, selectId, includeEmpty){
    // compare if any of the tokens in filters[filterKey] is found in match_values

    // re-create dropdown options after applying filters
    // filter logic is: value == filter OR !value
    // multiple values must use ";" as separator, e.g. "RNA;AMPS_RNA"
    var filtered_options = $.map(data, function(option){
        for (filterKey in filters){
            if (option[filterKey] && option[filterKey] != "None"){
                var match_values = option[filterKey].split(';');
                //var filter_tokens = filters[filterKey].split(';');
                var isFound = false;
                for (match_value in match_values) {
                    if (filters[filterKey].toLowerCase().indexOf(match_values[match_value].toLowerCase()) >= 0) {
                        //console.log("Include only templatingSize related protocol(pcr200_400bp));
                        if (('excludeProtocol' in filters ) && !(match_values[match_value].indexOf("pcr") >= 0)) {
                            return;
                        }
                        //console.log('FOUND! multi_tokens filtered - selectId=', selectId, "; filterKey=", filterKey, "; filters[key]=", filters[filterKey], "; match_value=", match_value, "; matchValues=", match_values, "; display=", option.display)
                        isFound = true;
                    }
                }
                if (!isFound) {
                    return;
                }
            }
        }
        return ({ "index": option.index, "value": option.value, "display": option.display });
    });
    // make sure to keep original option order
    filtered_options.sort(function(a,b){ return a.index > b.index? 1: -1;})

    create_filter_select_dropdown(filtered_options, selectId, includeEmpty);
}

function create_filter_select_dropdown(filtered_options, selectId, includeEmpty){
    var includeEmpty = (includeEmpty === undefined) ? true : includeEmpty;

    // create dropdown options
    var $select = $(selectId);
    var selectedValue = $select.val();
    $select.empty();
    if (includeEmpty) $select.append($("<option></option>"));

    $.each(filtered_options, function(i, obj){
        $select.append(
            $('<option>', {
                value: obj.value,
                html: obj.display,
                selected: obj.value == selectedValue
            })
        );
    });
    // if previously selected option no longer available, trigger onchange
    if ($select.val() != selectedValue){
        console.log(selectId, selectedValue, 'not available after filtering');
        $select.change();
    }
}

function update_LibraryKit_select(){
    var filters = get_filters("");
    filter_select_dropdown(libraryKits, filters, "#libraryKitType");
}

function getSamplePrep_filter_option(){
    templatekitType_value = $('input[name="templatekitType"]:checked').val();
    if (templatekitType_value == "IA"){
        SamplePrep_filter_option = "IA";
    }
    else{
        SamplePrep_filter_option = templatekitType_value == "IonChef"? "IC" : "OT";
    }
    return SamplePrep_filter_option;
}

function update_TemplateKit_select(){
    var filters = get_filters("");
    filters['samplePrep_instruments'] = getSamplePrep_filter_option();

    filter_select_dropdown(templateKits, filters, "#templateKit");
}

function update_SequencingKit_select(){
    var filters = get_filters("");
    filters['samplePrep_instruments'] = getSamplePrep_filter_option();

    filter_select_dropdown(sequencingKits, filters, "#sequenceKit");
}

function update_ChipType_select(){
    var filters = {};
    var instrumentType = $('#instrumentType').val();
    if (instrumentType)
        filters = {'seq_instrument': instrumentType };
    filter_select_dropdown(chips, filters, "#chipType");
}

function update_LibraryKit_select_by_category_filter(categoryFilter){
    var filters = get_filters(categoryFilter);
    filter_select_dropdown(libraryKits, filters, "#libraryKitType");
}

function update_TemplateKit_select_by_category_filter(categoryFilter){
    var filters = get_filters(categoryFilter);
    filters['samplePrep_instruments'] = $('input[name="templatekitType"]:checked').val() == "IonChef"? "IC" : "OT",

    filter_select_dropdown(templateKits, filters, "#templateKit");
}

function update_SequencingKit_select_by_category_filter(categoryFilter){
    var filters = get_filters(categoryFilter);
    filters['samplePrep_instruments'] = $('input[name="templatekitType"]:checked').val() == "IonChef"? "IC" : "OT",

    filter_select_dropdown(sequencingKits, filters, "#sequenceKit");
}

function get_filters(categoryFilter) {
    var filters = {
        'runType': _runType
    };

    if (categoryFilter) {
        filters['categories'] = categoryFilter
    }
    var instrumentType = $('#instrumentType').val();
    if (instrumentType){
        filters['seq_instruments'] = instrumentType;
    }
    var chipType = $('#chipType').val();
    if (chips[chipType]){
        filters['chipTypes'] = chipType;
    }

    return filters;
 }

// Show and hide advanced settings
$(".showhide").click(function (event) {
    event.preventDefault();
    $(this).toggleClass('icon-minus icon-plus');
    $(".hideable_advanced_settings_section").toggle();
    window.localStorage.setItem("plan-advanced-settings", $(this).hasClass("icon-minus"));
});
if (window.localStorage.getItem("plan-advanced-settings") == "true") {
    $(".showhide").click();
}

function isCustomKitSettings(){
    return $('[name=advancedSettingsChoice]:checked').val() == 'custom';
}

function show_chiptype_warning(chipType){
    var warning = chipNameToWarning[chipType];
    if (warning){
        $("#chipTypeWarning").html(warning).show();
        $("#chipType").css('border-color','red');
    } else {
        $("#chipTypeWarning").empty().hide();
        $("#chipType").css('border-color','')
    }
}

function update_templating_size_warning(){
    $("#templatingProtocolAlert").empty().hide();
    var templateKit = templateKits[$("#templateKit").val()];
    if (templateKit){
        var kitCategories = templateKit.categories;
        if (kitCategories.indexOf("pcr200_400bp") >= 0) {
            warning = "Templating Size <i class='icon-info-sign' rel='tooltip' title='Warning! " +
                "Templating Size is obsolete. Click Advanced Settings -> Customize -> select appropriate Templating Protocol'></i>";
            $("#templatingProtocolAlert").html(warning).show();
        }
    }
}
