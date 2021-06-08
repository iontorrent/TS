//=============================================
//MultiSelect - A user extension of KendoUI DropDownList widget.
(function ($) {
    // shorten references to variables
    var kendo = window.kendo,
        ui = kendo.ui,
        DropDownList = ui.DropDownList,
        keys = kendo.keys,
        SELECT = "select",
        SELECTIONCHANGED = "selectionChanged",
        SELECTED = "k-state-selected",
        HIGHLIGHTED = "k-state-active",
        CHECKBOX = "custom-check-item",
        SELECTALLITEM = "custom-multiselect-selectAll-item",
        MULTISELECTPOPUP = "custom-multiselect-popup",
        EMPTYSELECTION = "custom-multiselect-summary-empty";
    var lineTemplate = '<input type="checkbox" name="#= {1} #" value="#= {0} #" class="' + CHECKBOX + '" />' +
        '<span data-value="#= {0} #">#= {1} #</span>';
    var MultiSelectBox = DropDownList.extend({
        init: function (element, options) {
            options.template = kendo.template(kendo.format(lineTemplate, options.dataValueField, options.dataTextField));
            // base call to widget initialization
            DropDownList.fn.init.call(this, element, options);
        },
        options: {
            name: "MultiSelectBox",
            index: -1,
            showSelectAll: true,
            preSummaryCount: 1,  // number of items to show before summarising
            emptySelectionLabel: ''  // what to show when no items are selected
        },
        events: [
            SELECTIONCHANGED
        ],
        refresh: function () {
            // base call
            DropDownList.fn.refresh.call(this);
            this._updateSummary();
            $(this.popup.element).addClass(MULTISELECTPOPUP);
        },
        current: function (candidate) {
            return this._current;
        },
        open: function () {
            var that = this;
            this._removeSelectAllItem();
            this._addSelectAllItem();
            if (!that.ul[0].firstChild) {
                that._open = true;
                if (!that._request) {
                    that.dataSource.fetch();
                }
            } else {
                that.popup.open();
                that._scroll(that._current);
            }
        },
        close: function () {
            this._removeSelectAllItem();
            this._current = null;
            this._highlightCurrent();
            this._raiseSelectionChanged();
            DropDownList.fn.close.call(this);
        },
        _raiseSelectionChanged: function () {
            var currentValue = this.value();
            var currentValues = currentValue.length > 0 ? currentValue.split(",") : [];
            if (this._oldValue) {
                var hasChanged = !($(this._oldValue).not(currentValues).length == 0 && $(currentValues).not(this._oldValue).length == 0);
                if (hasChanged) {
                    this.trigger(SELECTIONCHANGED, {newValue: currentValues, oldValue: this._oldValue});
                }
            } else if (currentValue.length > 0) {
                this.trigger(SELECTIONCHANGED, {newValue: currentValues, oldValue: this._oldValue});
            }
            this._oldValue = currentValues;
        },
        _addSelectAllItem: function () {
            if (!this.options.showSelectAll) return;
            var firstListItem = this.ul.children('li:first');
            if (firstListItem.length > 0) {
                this.selectAllListItem = $('<li tabindex="-1" role="option" unselectable="on" class="k-item ' + SELECTALLITEM + '"></li>').insertBefore(firstListItem);
                // fake a data object to use for the template binding below
                var selectAllData = {};
                selectAllData[this.options.dataValueField] = '*';
                selectAllData[this.options.dataTextField] = 'All';
                this.selectAllListItem.html(this.options.template(selectAllData));
                this._updateSelectAllItem();
                this._makeUnselectable(); // required for IE8
            }
        },
        _removeSelectAllItem: function () {
            if (this.selectAllListItem) {
                this.selectAllListItem.remove();
            }
            this.selectAllListItem = null;
        },
        _focus: function (li) {
            if (this.popup.visible() && li && this.trigger(SELECT, {item: li})) {
                this.close();
                return;
            }
            this.select(li);
        },
        _highlightCurrent: function () {
            $('li', this.ul).removeClass(HIGHLIGHTED);
            $(this._current).addClass(HIGHLIGHTED);
        },
        _keydown: function (e) {
            // currently ignore Home and End keys
            // can be added later
            if (e.keyCode == kendo.keys.HOME ||
                e.keyCode == kendo.keys.END) {
                e.preventDefault();
                return;
            }
            DropDownList.fn._keydown.call(this, e);
        },
        _move: function (e) {
            var that = this,
                key = e.keyCode,
                ul = that.ul[0],
                down = key === keys.DOWN,
                pressed;
            if (key === keys.UP || down) {
                if (down) {
                    if (!that.popup.visible()) {
                        that.toggle(down);
                    }
                    if (!that._current) {
                        that._current = ul.firstChild;
                    } else {
                        that._current = ($(that._current)[0].nextSibling || that._current);
                    }
                } else {
                    //up
                    // only if anything is highlighted
                    if (that._current) {
                        that._current = ($(that._current)[0].previousSibling || ul.firstChild);
                    }
                }
                if (that._current) {
                    that._scroll(that._current);
                }
                that._highlightCurrent();
                e.preventDefault();
                pressed = true;
            } else {
                pressed = DropDownList.fn._move.call(this, e);
            }
            return pressed;
        },
        selectAll: function () {
            var unselectedItems = this._getUnselectedListItems();
            this._selectItems(unselectedItems);
            // todo: raise custom event
        },
        unselectAll: function () {
            var selectedItems = this._getSelectedListItems();
            this._selectItems(selectedItems);  // will invert the selection
            // todo: raise custom event
        },
        _selectItems: function (listItems) {
            var that = this;
            $.each(listItems, function (i, item) {
                var idx = ui.List.inArray(item, that.ul[0]);
                that.select(idx);  // select OR unselect
            });
        },
        _select: function (li) {
            var that = this,
                current = that._current,
                data = that._data(),
                value,
                text,
                idx;
            li = that._get(li);
            if (li && li[0]) {
                idx = ui.List.inArray(li[0], that.ul[0]);
                // if (li[0].innerText === "Upload Only") {
                //     $(this._getAllValueListItems()).removeClass(SELECTED);
                // } else {
                //     $(this.ul.children("li")[1]).removeClass(SELECTED);
                // }

                if (idx > -1) {
                    if (li.hasClass(SELECTED)) {
                        li.removeClass(SELECTED);
                        that._uncheckItem(li);
                        if (this.selectAllListItem && li[0] === this.selectAllListItem[0]) {
                            this.unselectAll();
                        }
                    } else {
                        li.addClass(SELECTED);
                        that._checkItem(li);
                        if (this.selectAllListItem && li[0] === this.selectAllListItem[0]) {
                            this.selectAll();
                        }
                    }
                    if (this._open) {
                        that._current(li);
                        that._highlightCurrent();
                    }
                    var selecteditems = this._getSelectedListItems();
                    value = [];
                    text = [];
                    $.each(selecteditems, function (indx, item) {
                        var obj = $(item).children("span").first();
                        value.push(obj.attr("data-value"));
                        text.push(obj.text());
                    });
                    that._updateSummary(text);
                    that._updateSelectAllItem();
                    that._accessor(value, idx);
                    // todo: raise change event (add support for selectedIndex) if required
                }
            }
        },
        _getAllValueListItems: function () {
            if (this.selectAllListItem) {
                return this.ul.children("li").not(this.selectAllListItem[0]);
            } else {
                return this.ul.children("li");
            }
        },
        _getSelectedListItems: function () {
            return this._getAllValueListItems().filter("." + SELECTED);
        },
        _getUnselectedListItems: function () {
            return this._getAllValueListItems().filter(":not(." + SELECTED + ")");
        },
        _getSelectedItemsText: function () {
            var text = [];
            var selecteditems = this._getSelectedListItems();
            $.each(selecteditems, function (indx, item) {
                var obj = $(item).children("span").first();
                text.push(obj.text());
            });
            return text;
        },
        _updateSelectAllItem: function () {
            if (!this.selectAllListItem) return;
            // are all items selected?
            if (this._getAllValueListItems().length == this._getSelectedListItems().length) {
                this._checkItem(this.selectAllListItem);
                this.selectAllListItem.addClass(SELECTED);
            } else {
                this._uncheckItem(this.selectAllListItem);
                this.selectAllListItem.removeClass(SELECTED);
            }
        },
        _updateSummary: function (itemsText) {
            if (!itemsText) {
                itemsText = this._getSelectedItemsText();
            }
            if (itemsText.length == 0) {
                this._inputWrapper.addClass(EMPTYSELECTION);
                this.text(this.options.emptySelectionLabel);
                return;
            } else {
                this._inputWrapper.removeClass(EMPTYSELECTION);
            }
            if (itemsText.length <= this.options.preSummaryCount) {
                this.text(itemsText.join(", "));
            } else {
                this.text(itemsText.length + ' selected');
            }
        },
        _checkItem: function (itemContainer) {
            if (!itemContainer) return;
            itemContainer.children("input").attr("checked", "checked");
        },
        _uncheckItem: function (itemContainer) {
            if (!itemContainer) return;
            itemContainer.children("input").removeAttr("checked");
        },
        _isItemChecked: function (itemContainer) {
            return itemContainer.children("input:checked").length > 0;
        },
        value: function (value) {
            var that = this,
                idx,
                valuesList = [];
            if (value !== undefined) {
                if (!$.isArray(value)) {
                    valuesList.push(value);
                    this._oldValue = valuesList; // to allow for selectionChanged event
                } else {
                    valuesList = value;
                    this._oldValue = value; // to allow for selectionChanged event
                }
                // clear all selections first
                $(that.ul[0]).children("li").removeClass(SELECTED);
                $("input", that.ul[0]).removeAttr("checked");
                $.each(valuesList, function (indx, item) {
                    if (item !== null) {
                        item = item.toString();
                    }
                    that._valueCalled = true;
                    if (item && that._valueOnFetch(item)) {
                        return;
                    }
                    idx = that._index(item);
                    if (idx > -1) {
                        that.select(idx);
                    }
                });
            } else {
                return that._accessor();
            }
        }
    });
    ui.plugin(MultiSelectBox);
})(jQuery);