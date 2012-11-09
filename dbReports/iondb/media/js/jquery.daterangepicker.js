// jquery.daterangepicker plugin 1.0
// May be freely distributed under the MIT and GPL licenses
// Copyright (c) 2011 Raymond Julin
// Copyright (c) 2010 Filament Group, Inc
//
// Forked off daterangepicker.jQuery.js
// by Scott Jehl, scott@filamentgroup.com (http://www.filamentgroup.com/lab/update_date_range_picker_with_jquery_ui/)
// Dependencies: jquery, jquery UI datepicker, date.js, jQuery UI CSS Framework

(function ($) {
    var daterangepicker = function(settings) {
        var rangeInput = $(this), now = new Date(), today = Date.parse('today');
        var dateValueExpander = function(value) {
            switch (typeof value) {
                case 'string':
                    return Date.parse(value);
                case 'function':
                    return value();
            }
            return value;
        };

        //defaults
        var options = jQuery.extend({
            presetRanges: [
                {text: 'Today', dateStart: today, dateEnd: today},
                {text: 'Last 7 days', dateStart: 'today-7days', dateEnd: today},
                {text: 'This month',
                    dateStart: new Date(now.getFullYear(), now.getMonth(), 1),
                    dateEnd: 'today'
                }
            ],
            //presetRanges: array of objects for each menu preset.
            //Each obj must have text, dateStart, dateEnd. dateStart, dateEnd accept date.js string or a function which returns a date object
            presets: {
                dateRange: 'Date Range'
            },
            rangeStartTitle: 'Start date',
            rangeEndTitle: 'End date',
            nextLinkText: 'Next',
            prevLinkText: 'Prev',
            doneButtonText: 'Done',
            earliestDate: false,
            latestDate: false,
            constrainDates: false,
            rangeSplitter: ' - ', //string to use between dates in single input
            dateFormat: $.datepicker.ISO_8601, // Available formats: http://docs.jquery.com/UI/Datepicker/%24.datepicker.formatDate
            closeOnSelect: true, //if a complete selection is made, close the menu
            arrows: false,
            appendTo: 'body',
            onClose: false,
            onOpen: false,
            onChange: false,
            datepickerOptions: null //object containing native UI datepicker API options
        }, settings);



        //custom datepicker options, extended by options
        var datepickerOptions = {
            onSelect: function(dateText, inst) {
                    $(this).trigger('constrainOtherPicker');

                    var start = rp.find('.range-start').datepicker('getDate');
                    var end = rp.find('.range-end').datepicker('getDate');
                    var rangeA = fDate(start);
                    var rangeB = fDate(end);

                    if(rp.find('.ui-daterangepicker-specificDate').is('.ui-state-active')){
                                rangeB = rangeA;
                            }

                    //send back to input or inputs
                    if(rangeInput.length === 2){
                        rangeInput.eq(0).val(rangeA);
                        rangeInput.eq(1).val(rangeB);
                    }
                    else{
                        rangeInput.val((rangeA !== rangeB) ? rangeA + options.rangeSplitter + rangeB : rangeA);
                    }

                    rangeInput.data('daterange', {
                        start : start,
                        end : end
                    });
                    //if closeOnSelect is true
                    if (options.closeOnSelect)
                    {
                        if (!rp.find('li.ui-state-active').is('.ui-daterangepicker-dateRange') && !rp.is(':animated'))
                        {
                            hideRP();
                        }
                    }
                    rangeInput.trigger('change');
                },
                defaultDate: +0
        };

        //change event fires both when a calendar is updated or a change event on the input is triggered
        if (options.onChange) {
            rangeInput.bind('change', options.onChange);
        }

        //datepicker options from options
        options.datepickerOptions = (settings) ? jQuery.extend(datepickerOptions, settings.datepickerOptions) : datepickerOptions;

        //Capture Dates from input(s)
        var inputDateA, inputDateB = today;
        var inputDateAtemp, inputDateBtemp;
        if(rangeInput.size() === 2){
            inputDateAtemp = Date.parse( rangeInput.eq(0).val() );
            inputDateBtemp = Date.parse( rangeInput.eq(1).val() );
            if(inputDateAtemp === null){inputDateAtemp = inputDateBtemp;}
            if(inputDateBtemp === null){inputDateBtemp = inputDateAtemp;}
        }
        else {
            var rangeInputVals = rangeInput.val().split(options.rangeSplitter);
            inputDateAtemp = Date.parse( rangeInputVals[0] );
            inputDateBtemp = Date.parse( rangeInputVals[1] );
            if(inputDateBtemp === null){inputDateBtemp = inputDateAtemp;} //if one date, set both
        }
        if(inputDateAtemp !== null){inputDateA = inputDateAtemp;}
        if(inputDateBtemp !== null){inputDateB = inputDateBtemp;}


        //build picker and
        var rp = $('<div class="ui-daterangepicker ui-widget ui-helper-clearfix ui-widget-content ui-corner-all"></div>');
        var rpPresets = (function(){
            var ul = $('<ul class="ui-widget-content"></ul>').appendTo(rp);
            jQuery.each(options.presetRanges,function(){
                this.dateStart = dateValueExpander(this.dateStart);
                this.dateEnd = dateValueExpander(this.dateEnd);
                $('<li class="ui-daterangepicker-'+ this.text.replace(/ /g, '') +' ui-corner-all"><a href="#">'+ this.text +'</a></li>')
                    .data('dateStart', this.dateStart)
                    .data('dateEnd', this.dateEnd)
                    .appendTo(ul);
            });
            var x=0;
            jQuery.each(options.presets, function(key, value) {
                $('<li class="ui-daterangepicker-'+ key +' preset_'+ x +' ui-helper-clearfix ui-corner-all"><span class="ui-icon ui-icon-triangle-1-e"></span><a href="#">'+ value +'</a></li>')
                .appendTo(ul);
                x++;
            });

            ul.find('li').hover(
                    function(){
                        $(this).addClass('ui-state-hover');
                    },
                    function(){
                        $(this).removeClass('ui-state-hover');
                    })
                .click(function(){
                    rp.find('.ui-state-active').removeClass('ui-state-active');
                    $(this).addClass('ui-state-active');
                    clickActions($(this),rp, rpPickers, doneBtn);
                    return false;
                });
            return ul;
        }());

        //function to format a date string
        function fDate(date){
            return (date !== null && date.getDate())
                ? jQuery.datepicker.formatDate( options.dateFormat, date ) : '';
        }


        jQuery.fn.restoreDateFromData = function(){
            var node = $(this);
            if (node.data('saveDate')) {
                node.datepicker('setDate', node.data('saveDate')).removeData('saveDate');
            }
            return this;
        };
        jQuery.fn.saveDateToData = function(){
            var node = $(this);
            if (!node.data('saveDate')) {
                node.data('saveDate', node.datepicker('getDate') );
            }
            return this;
        };

        //show, hide, or toggle rangepicker
        function showRP(){
            if(rp.data('state') === 'closed'){
                positionRP();
                rp.fadeIn(300).data('state', 'open');
                if (options.onOpen) options.onOpen();
            }
        }
        function hideRP(){
            if(rp.data('state') === 'open'){
                rp.fadeOut(300).data('state', 'closed');
                if (options.onClose) options.onClose();
            }
        }
        function toggleRP(){
            if( rp.data('state') === 'open' ){ hideRP(); }
            else { showRP(); }
        }
        function positionRP(){
            var relEl = riContain || rangeInput; //if arrows, use parent for offsets
            var riOffset = relEl.offset(),
                side = 'left',
                val = riOffset.left,
                offRight = $(window).width() - val - relEl.outerWidth();

            if(val > offRight){
                side = 'right';
                val = offRight;
            }

            rp.parent().css(side, val).css('top', riOffset.top + relEl.outerHeight());
        }



        //preset menu click events
        function clickActions(el, rp, rpPickers, doneButton){

            var rangeStart = rp.find('.range-start'), rangeEnd = rp.find('.range-end');
            doneButton.hide();
            if (el.is('.ui-daterangepicker-specificDate')){
                //Specific Date (show the "start" calendar)
                rpPickers.show();
                rp.find('.title-start').text( options.presets.specificDate );
                rangeStart.restoreDateFromData().css('opacity',1).show(400);
                rangeEnd.restoreDateFromData().css('opacity',0).hide(400);
                setTimeout(function(){doneButton.fadeIn();}, 400);
            }
            else if (el.is('.ui-daterangepicker-allDatesBefore')){
                //All dates before specific date (show the "end" calendar and set the "start" calendar to the earliest date)
                rpPickers.show();
                rp.find('.title-end').text( options.presets.allDatesBefore );
                rangeStart.saveDateToData().css('opacity',0).hide(400);
                rangeEnd.restoreDateFromData().css('opacity',1).show(400);

                if (options.earliestDate) rangeStart.datepicker('setDate', options.earliestDate);

                setTimeout(function(){doneButton.fadeIn();}, 400);
            }
            else if(el.is('.ui-daterangepicker-allDatesAfter')){
                //All dates after specific date (show the "start" calendar and set the "end" calendar to the latest date)
                rpPickers.show();
                rp.find('.title-start').text( options.presets.allDatesAfter );
                rangeStart.restoreDateFromData().css('opacity',1).show(400);
                rangeEnd.saveDateToData().css('opacity',0).hide(400);

                if (options.latestDate) rangeEnd.datepicker('setDate', options.latestDate);
                setTimeout(function(){doneButton.fadeIn();}, 400);
            }
            else if(el.is('.ui-daterangepicker-dateRange')){
                //Specific Date range (show both calendars)
                rpPickers.show();
                rp.find('.title-start').text(options.rangeStartTitle);
                rp.find('.title-end').text(options.rangeEndTitle);
                rangeStart.restoreDateFromData().datepicker('refresh').css('opacity',1).show(400);
                rangeEnd.restoreDateFromData().datepicker('refresh').css('opacity',1).show(400);
                setTimeout(function(){doneButton.fadeIn();}, 400);
            }
            else {
                //custom date range specified in the options (no calendars shown)
                rp.find('.range-start, .range-end').css('opacity',0).hide(400, function(){
                    rpPickers.hide();
                });

                rangeStart.datepicker('setDate', el.data('dateStart'));
                rangeEnd.datepicker('setDate', el.data('dateEnd'));

                // This actually triggers a close on the dialog as well as marks the selected dates
                rp.find('.ui-datepicker-current-day a').trigger('click');
            }

            return false;
        }


        //picker divs
        var rpPickers = $('<div class="ranges ui-widget-header ui-corner-all ui-helper-clearfix"><div class="range-start"><span class="title-start">Start Date</span></div><div class="range-end"><span class="title-end">End Date</span></div></div>').appendTo(rp);
        rpPickers.find('.range-start, .range-end')
            .datepicker(options.datepickerOptions);


        rpPickers.find('.range-start').datepicker('setDate', inputDateA);
        rpPickers.find('.range-end').datepicker('setDate', inputDateB);

        rpPickers.find('.range-start, .range-end')
            .bind('constrainOtherPicker', function(){
                if(options.constrainDates){
                    //constrain dates
                    if($(this).is('.range-start')){
                        rp.find('.range-end').datepicker( "option", "minDate", $(this).datepicker('getDate'));
                    }
                    else{
                        rp.find('.range-start').datepicker( "option", "maxDate", $(this).datepicker('getDate'));
                    }
                }
            })
            .trigger('constrainOtherPicker');

        var doneBtn = $('<button class="btnDone ui-state-default ui-corner-all">'+ options.doneButtonText +'</button>')
        .click(function(){
            rp.find('.ui-datepicker-current-day').trigger('click');
            hideRP();
        })
        .hover(
                function(){
                    $(this).addClass('ui-state-hover');
                },
                function(){
                    $(this).removeClass('ui-state-hover');
                }
        )
        .appendTo(rpPickers);




        //inputs toggle rangepicker visibility
        $(this).click(function(){
            toggleRP();
            return false;
        });
        //hide em all
        rpPickers.hide().find('.range-start, .range-end, .btnDone').hide();

        rp.data('state', 'closed');

        //Fixed for jQuery UI 1.8.7 - Calendars are hidden otherwise!
        rpPickers.find('.ui-datepicker').css("display","block");

        //inject rp
        $(options.appendTo).append(rp);

        //wrap and position
        rp.wrap('<div class="ui-daterangepickercontain"></div>');

        //add arrows (only available on one input)
        if(options.arrows && rangeInput.size()===1){
            var prevLink = $('<a href="#" class="ui-daterangepicker-prev ui-corner-all" title="'+ options.prevLinkText +'"><span class="ui-icon ui-icon-circle-triangle-w">'+ options.prevLinkText +'</span></a>');
            var nextLink = $('<a href="#" class="ui-daterangepicker-next ui-corner-all" title="'+ options.nextLinkText +'"><span class="ui-icon ui-icon-circle-triangle-e">'+ options.nextLinkText +'</span></a>');

            $(this)
            .addClass('ui-rangepicker-input ui-widget-content')
            .wrap('<div class="ui-daterangepicker-arrows ui-widget ui-widget-header ui-helper-clearfix ui-corner-all"></div>')
            .before( prevLink )
            .before( nextLink )
            .parent().find('a').click(function(){
                var dateA = rpPickers.find('.range-start').datepicker('getDate');
                var dateB = rpPickers.find('.range-end').datepicker('getDate');
                var diff = Math.abs( new TimeSpan(dateA - dateB).getTotalMilliseconds() ) + 86400000; //difference plus one day
                if($(this).is('.ui-daterangepicker-prev')){ diff = -diff; }

                rpPickers.find('.range-start, .range-end ').each(function(){
                        var thisDate = $(this).datepicker( "getDate");
                        if(thisDate === null){return false;}
                        $(this).datepicker( "setDate", thisDate.add({milliseconds: diff}) ).find('.ui-datepicker-current-day').trigger('click');
                });
                return false;
            })
            .hover(
                function(){
                    $(this).addClass('ui-state-hover');
                },
                function(){
                    $(this).removeClass('ui-state-hover');
                });

            var riContain = rangeInput.parent();
        }


        $(document).click(function(){
            if (rp.is(':visible')) {
                hideRP();
            }
        });

        rp.click(function(){return false;}).hide();
        return this;
    };

    jQuery.fn.daterangepicker = daterangepicker;
}(jQuery));
