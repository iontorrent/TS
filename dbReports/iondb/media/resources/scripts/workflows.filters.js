(function ($, global) {

    var FilterSlider = global.SliderWidget;
    var FilterSelect = global.SelectWidget;

    function FilterDialog(dialogEl) {
        this.dialogEl = dialogEl;
        this.configurationWidget = null;
        this.activeSettingName = null;
    }

    FilterDialog.prototype.init = function (dialogEl) {
    	this.dialogEl = dialogEl
    	if(this._init) return;
    	
        this._parseOptions();
        this._addEventHandlers();

        // Need to trigger change event immediately to fill in the configuration opts.
        this.dialogEl.find("#settingDatumSelector").trigger("change");
    	this._init = true;
    };

    FilterDialog.prototype._parseOptions = function () {
        var self = this;
        var filterOptionEls = this.dialogEl.find("#settingDatumSelector option");

        // self.filterOptions = [];
        $.each(filterOptionEls, function (index, optionRaw) {
            var optionEl = $(optionRaw);
            var options = optionEl.data('options'); // jquery automatically decodes the JSON
            options['label'] = optionEl.text();
            optionEl.data('optionIndex', index);
            // self.filterOptions.push(options);
        });
    };

    FilterDialog.prototype._addEventHandlers = function () {
        var self = this;
        this.dialogEl.on("change", "#settingDatumSelector", $.proxy(this.onFilterTypeChange, this));

        this.dialogEl.on("click", ".btn-primary", function (clickEvent) {
            clickEvent.preventDefault();
            self.dialogEl.find("form").submit();
        });
        this.dialogEl.on("submit", "form", $.proxy(this.onFormSubmit, this));
    };

    FilterDialog.prototype.onFormSubmit = function (submitEvent) {
        submitEvent.preventDefault();

        var settingTemplateHtml = $("#filter-setting-template").text(),
            settingTemplate = $(settingTemplateHtml);

        settingTemplate.find(".setting-name").text(this.activeSettingName);
        settingTemplate.find(".setting-value").text(this.configurationWidget.getSettingValue());

        settingTemplate.appendTo("#filtersChain");
        $('#addFilterDialog').modal('hide');
    };

    FilterDialog.prototype.onFilterTypeChange = function (changeEvent) {
        var filterTypeSelector = $(changeEvent.target),
            filterTypeSelected = $(filterTypeSelector.children()[filterTypeSelector[0].selectedIndex]),
            options = filterTypeSelected.data('options');

        if (this.configurationWidget) {
            this.configurationWidget.destroy();
            this.configurationWidget = null;
        }
        this.activeSettingName = filterTypeSelected.text();
        var destinationEl = $("#addFilterDialog #filterInstanceCreateFields");
        console.log('destinationEl', destinationEl);
        if (options['type'] == "slider") {
            this.configurationWidget = new FilterSlider(destinationEl, options);
        } else if (options['type'] == "select") {
            this.configurationWidget = new FilterSelect(destinationEl, options);
        }
    };

    global.FilterDialog = FilterDialog;
/*
    $(document).ready(function () {
        var addFilterDialog = $("#addFilterDialog");

        var filterDialogWidget = new FilterDialog(addFilterDialog);
        filterDialogWidget.init();

        $("#createNewFilterChain").on("click", function (clickEvent) {
            clickEvent.preventDefault();
            $(".workflow-settings-editor-inactive").removeClass("workflow-settings-editor-inactive");
            $("#filtersChain").sortable().sortable("enable")
                .disableSelection();
        });
        $("#settingsSaveBtn").on("click", function (clickEvent) {
            clickEvent.preventDefault();
            $(".workflow-settings-editor").addClass("workflow-settings-editor-inactive");

            $("#filtersChain").sortable("disable")
                .enableSelection();
        });

        $(".settings-chain").on("click", ".icon-trash", function (clickEvent) {
            clickEvent.preventDefault();
            $(clickEvent.target).closest("li").remove();
        });

    });
*/
}(jQuery, window));
