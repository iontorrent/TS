jQuery(function ($) {
    var now = new Date(),
        today = Date.parse('today');

    $.DateRangePickerSettings.regional['zh-CN'] = {
        dateFormat: 'yy M d',
        presetRanges: [
            {text: '今天', dateStart: today, dateEnd: today},
            {text: '过去7天', dateStart: 'today-7days', dateEnd: today},
            {text: '过去30天', dateStart: 'today-30days', dateEnd: today},
            {text: '过去60天', dateStart: 'today-60days', dateEnd: today},
            {text: '过去90天', dateStart: 'today-90days', dateEnd: today}
        ],
        presets: {
            dateRange: '日期范围',
            allDatesBefore: '比日期早',
            allDatesAfter: '比日期更新'
        },
        rangeStartTitle: '开始日期',
        rangeEndTitle: '结束日期',
        nextLinkText: '下一个',
        prevLinkText: '上一页',
        doneButtonText: '完成',
    };
    $.DateRangePickerSettings.setDefaults($.DateRangePickerSettings.regional['zh-CN']);
});