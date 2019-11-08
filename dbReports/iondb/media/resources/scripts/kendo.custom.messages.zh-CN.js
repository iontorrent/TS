 (function ($, undefined) {
    /* Pager messages */
    if (kendo.ui.Pager) {
        kendo.ui.Pager.prototype.options.messages =
            $.extend(true, kendo.ui.Pager.prototype.options.messages,{
                "display": "{0} - {1} 条　共 {2} 条数据",
                "empty": "无相关数据",
                "page": "转到第",
                "of": "页　共 {0} 页",
                "itemsPerPage": "条每页",
                "first": "首页",
                "previous": "上一页",
                "next": "下一页",
                "last": "末页",
                "refresh": "刷新"
            });
    }
 })(window.kendo.jQuery);