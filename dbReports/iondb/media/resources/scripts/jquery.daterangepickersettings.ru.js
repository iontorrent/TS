jQuery(function ($) {
    var now = new Date(),
        today = Date.parse('today');

    $.DateRangePickerSettings.regional['ru'] = {
        dateFormat: 'yy M d',
        presetRanges: [
            {text: 'Cегодня', dateStart: today, dateEnd: today},
            {text: 'Последние 7 дней', dateStart: 'today-7days', dateEnd: today},
            {text: 'Последние 30 дней', dateStart: 'today-30days', dateEnd: today},
            {text: 'Последние 60 дней', dateStart: 'today-60days', dateEnd: today},
            {text: 'Последние 90 дней', dateStart: 'today-90days', dateEnd: today}
        ],
        presets: {
            dateRange: 'Диапазон дат',
            allDatesBefore: 'Старее даты',
            allDatesAfter: 'Новее, чем дата'
        },
        rangeStartTitle: 'Дата начала',
        rangeEndTitle: 'Дата окончания',
        nextLinkText: 'следующий',
        prevLinkText: 'предшествующий',
        doneButtonText: 'Готово',
    };
    $.DateRangePickerSettings.setDefaults($.DateRangePickerSettings.regional['ru']);
});