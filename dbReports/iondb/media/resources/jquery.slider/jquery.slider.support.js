var SliderSupport = {};

(function () {
    var createUpdateInputFunc = function (input) {
        return function (value) {
            $(input).val(value);
        }
    };

    var createUpdateSliderFunc = function (slider) {
        return function (inputEvent) {
            $(slider).slider("value", $(inputEvent.target).val());
        };
    };

    SliderSupport.makeSlider = function makeSlider(sliderInput, passedConfig) {
        var sliderAltTextInput = $(document.createElement("input")).addClass("slider-text-input");
        var defaults = {
            "max": 100,
            "step": 1,
            "round": 0
        };
        var config = $.extend({}, defaults, passedConfig);

        sliderInput.slider({
            from: 0,
            to: config['max'],
            step: config['step'],
            smooth: true,
            round: config['round'],
            skin: "blue",
            onstatechange: createUpdateInputFunc(sliderAltTextInput)
        });
        sliderAltTextInput.insertAfter(sliderInput.closest(".slider-wrapper"));
        sliderAltTextInput.val(sliderInput.val());
        sliderAltTextInput.on("input", createUpdateSliderFunc(sliderInput));
    };
}());
