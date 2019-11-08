(function ($, $_datepicker, undefined) {
    /* jQuery extend now ignores nulls! */
    function extendRemove(target, props) {
        $.extend(target, props);
        for (var name in props)
            if (props[name] == null || props[name] == undefined)
                target[name] = props[name];
        return target;
    }

    function DateRangePickerSettings() {
        this._language = ''; //default language
        this.regional = []; // Available regional settings, indexed by language code
        var now = new Date(),
            today = Date.parse('today');
        this.regional[''] = {
            dateFormat: 'M d yy',
            presetRanges: [
                {text: 'Today', dateStart: today, dateEnd: today},
                {text: 'Last 7 Days', dateStart: 'today-7days', dateEnd: today},
                {text: 'Last 30 Days', dateStart: 'today-30days', dateEnd: today},
                {text: 'Last 60 Days', dateStart: 'today-60days', dateEnd: today},
                {text: 'Last 90 Days', dateStart: 'today-90days', dateEnd: today}
            ],
            //presetRanges: array of objects for each menu preset.
            //Each obj must have text, dateStart, dateEnd. dateStart, dateEnd accept date.js string or a function which returns a date object
            presets: {
                dateRange: 'Date Range',
                allDatesBefore: 'Older than Date',
                allDatesAfter: 'Newer than Date'
            },
            rangeStartTitle: 'Start date',
            rangeEndTitle: 'End date',
            nextLinkText: 'Next',
            prevLinkText: 'Prev',
            doneButtonText: 'Done',
            earliestDate: Date.parse('1/1/2010'),
        };
        this._defaults = {
            earliestDate: Date.parse('1/1/2010'),
            latestDate: today,
            constrainDates: false,
            rangeSplitter: ' - ', //string to use between dates in single input
            dateFormat: $_datepicker.ISO_8601, // Available formats: http://docs.jquery.com/UI/Datepicker/%24.datepicker.formatDate
            closeOnSelect: true, //if a complete selection is made, close the menu
            arrows: false,
            appendTo: 'body',
            onClose: false,
            onOpen: false,
            onChange: false,
            datepickerOptions: null //object containing native UI datepicker API options};

        };
        $.extend(this._defaults, this.regional[this._language]);
    }

    $.extend(DateRangePickerSettings.prototype, {
        get: function (language) {
            language = language || this._language;
            language = language.replace(/_/g, '-');
            return this._defaults;
        },

        /* Override the default settings for all instances of the date range picker.
           @param  settings  object - the new settings to use as defaults (anonymous object)
           @return the manager object */
        setDefaults: function (settings) {
            extendRemove(this._defaults, settings || {});
            return this;
        },
    });

    $.DateRangePickerSettings = new DateRangePickerSettings(); //singleton instance
})(jQuery, $.datepicker);