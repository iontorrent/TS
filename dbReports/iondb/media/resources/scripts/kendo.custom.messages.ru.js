(function ($, undefined) {
    /* Pager messages */

    if (kendo.ui.Pager) {
        kendo.ui.Pager.prototype.options.messages =
            $.extend(true, kendo.ui.Pager.prototype.options.messages, {
                "display": "Отображены записи {0} - {1} из {2}",
                "empty": "Нет записей для отображения",
                "page": "Страница",
                "of": "из {0}",
                "itemsPerPage": "элементов на странице",
                "first": "Вернуться на первую страницу",
                "previous": "Перейти на предыдущую страницу",
                "next": "Перейдите на следующую страницу",
                "last": "К последней странице",
                "refresh": "Обновить",
            });
    }
})(window.kendo.jQuery);