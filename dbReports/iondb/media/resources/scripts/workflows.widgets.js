(function ($, global) {
    var destroyWidget = function () {
        this.targetEl.empty();
        this.targetEl.off(this.eventNamespace);
    };

    function SliderWidget(targetEl, passedConfig) {
        var defaults = {'defaultValue': 1}; // a default value is required or the slider breaks
        var config = $.extend({}, defaults, passedConfig);
        this.targetEl = targetEl;

        var sliderTemplateHtml = $("#slider-template").text();
        var slider = $("<div></div>").html(sliderTemplateHtml);
        var sliderInput = slider.find("input");

        slider.find(".slider-label-wrapper").text(config['label']);
        slider.find(".slider-unit").text(config['unit']);
        slider.appendTo(this.targetEl);
        sliderInput.val(config['defaultValue']);
        SliderSupport.makeSlider(sliderInput, {
            'step': config['step'],
            'max': config['max'],
            'round': config['round']
        });
    }

    SliderWidget.prototype.destroy = destroyWidget;
    SliderWidget.prototype.eventNamespace = ".filterSlider";
    SliderWidget.prototype.getSettingValue = function () {
        var sliderInput = this.targetEl.find("input");
        return sliderInput.val();
    };

    function SelectWidget(targetEl, passedConfig) {
        var defaults = {};
        this.config = $.extend({}, defaults, passedConfig);
        this.targetEl = targetEl;

        var selectWrapper = $("<div class='select-wrapper'></div>");
        if (this.config['multiple']) {
            $.each(this.config['options'], function (i, optionText) {
                var optionLabel = $("<label class='checkbox'></label>").text(optionText);
                $("<input type='checkbox' />").prependTo(optionLabel).data('optionText', optionText);
                optionLabel.appendTo(selectWrapper);
            });
        } else {
            var selectElement = $("<select></select>");
            $.each(this.config['options'], function (i, optionText) {
                var optionEl = $("<option></option>").text(optionText);
                optionEl.appendTo(selectElement);
            });
            selectElement.appendTo(selectWrapper);
        }

        selectWrapper.appendTo(this.targetEl);
    }

    SelectWidget.prototype.destroy = destroyWidget;
    SelectWidget.prototype.eventNamespace = ".filterSelect";
    SelectWidget.prototype.getSettingValue = function () {
        if (this.config['multiple']) {
            return this._getSettingValueMultiple();
        } else {
            return this._getSettingValueSingle()
        }
    };
    SelectWidget.prototype._getSettingValueMultiple = function() {
        var checkedInputs = this.targetEl.find(".select-wrapper input:checked"),
            values = [];
        $.each(checkedInputs, function () {
            values.push($(this).closest("label").text());
        });
        if (values.length > 2) {
            var lastValue = values.pop();
            return values.join(", ") + ", and " + lastValue;
        } else {
            return values.join(", ");
        }
    };
    SelectWidget.prototype._getSettingValueSingle = function () {
        var selectInput = this.targetEl.find("select");
        return selectInput.val();
    };

    // Export global objects
    global.SelectWidget = SelectWidget;
    global.SliderWidget = SliderWidget;

}(jQuery, window));