(function ($) {
    function SlickGridPager(dataView, grid, tools, $container) {
        var $status;
        var _toolsFunction = tools;

        function init() {
            dataView.onPagingInfoChanged.subscribe(function (e, pagingInfo) {
                updatePager(pagingInfo);
            });

            constructPagerUI();
            updatePager(dataView.getPagingInfo());
        }

        function getNavState() {
            var cannotLeaveEditMode = !Slick.GlobalEditorLock.commitCurrentEdit();
            var pagingInfo = dataView.getPagingInfo();
            var lastPage = Math.ceil(pagingInfo.totalRows / pagingInfo.pageSize) - 1;
            if (lastPage < 0) {
                lastPage = 0;
            }

            return {
                canGotoFirst: !cannotLeaveEditMode && pagingInfo.pageSize != 0 && pagingInfo.pageNum > 0,
                canGotoLast: !cannotLeaveEditMode && pagingInfo.pageSize != 0 && pagingInfo.pageNum != lastPage,
                canGotoPrev: !cannotLeaveEditMode && pagingInfo.pageSize != 0 && pagingInfo.pageNum > 0,
                canGotoNext: !cannotLeaveEditMode && pagingInfo.pageSize != 0 && pagingInfo.pageNum < lastPage,
                pagingInfo: pagingInfo,
                lastPage: lastPage
            }
        }

        function setPageSize(n) {
            dataView.setRefreshHints({
                isFilterUnchanged: true
            });
            dataView.setPagingOptions({pageSize: n});
        }

        function gotoFirst() {
            if (getNavState().canGotoFirst) {
                dataView.setPagingOptions({pageNum: 0});
                $("#checkAll").attr("src", "lifegrid/images/checkbox_empty.png");
            }
        }

        function gotoLast() {
            var state = getNavState();
            if (state.canGotoLast) {
                dataView.setPagingOptions({pageNum: state.lastPage});
                $("#checkAll").attr("src", "lifegrid/images/checkbox_empty.png");
            }
        }

        function gotoPrev() {
            var state = getNavState();
            if (state.canGotoPrev) {
                dataView.setPagingOptions({pageNum: state.pagingInfo.pageNum - 1});
                $("#checkAll").attr("src", "lifegrid/images/checkbox_empty.png");
            }
        }

        function gotoNext() {
            var state = getNavState();
            if (state.canGotoNext) {
                dataView.setPagingOptions({pageNum: state.pagingInfo.pageNum + 1});
                $("#checkAll").attr("src", "lifegrid/images/checkbox_empty.png");
            }
        }

        function constructPagerUI() {
            $container.empty();
            $('<span class="btn" id="export">Export Selected</span>').appendTo($container);

            if ($.QueryString["debug"]) {
                $('<span class="btn" id="suspect" style="margin-left: 10px;">Add selected to Diagnostic Export</span>').appendTo($container);
            }

            //var bulbID = $container.attr('id')+'-settings-expanded';
            var $nav = $("<span class='slick-pager-nav' />").appendTo($container);
            //var $settings = $("<span class='slick-pager-settings' />").appendTo($container);
            $status = $("<span class='slick-pager-status' />").appendTo($container);

            //$settings
            //    .append("<span id='"+bulbID+"' style='display:none'>Show: <a data='-1'>Auto</a><a data=25>25</a><a data=50>50</a><a data=100>100</a><a data=0>All</a></span>");

            var vp = grid.getViewport();
            setPageSize(20);

            /*
             $settings.find("a[data]").click(function (e) {
             var pagesize = $(e.target).attr("data");
             if (pagesize != undefined) {
             if (pagesize == -1) {
             var vp = grid.getViewport();
             setPageSize(vp.bottom - vp.top);
             } else {
             setPageSize(parseInt(pagesize));
             }
             }
             });
             */

            var icon_prefix = "<span class='ui-state-default ui-corner-all ui-icon-container'><span class='ui-icon ";
            var icon_suffix = "' /></span>";

            if (_toolsFunction) {
                $(icon_prefix + "ui-icon-wrench" + icon_suffix)
                    .click(_toolsFunction)
                    .attr('title', 'Export Selected')
                    .appendTo($nav);
            }

            /*
             $(icon_prefix + "ui-icon-lightbulb" + icon_suffix)
             .click(function () {
             $("#"+bulbID).toggle()
             })
             .appendTo($settings);
             */

            $(icon_prefix + "ui-icon-seek-first" + icon_suffix)
                .click(gotoFirst)
                .appendTo($nav);

            $(icon_prefix + "ui-icon-seek-prev" + icon_suffix)
                .click(gotoPrev)
                .appendTo($nav);

            $(icon_prefix + "ui-icon-seek-next" + icon_suffix)
                .click(gotoNext)
                .appendTo($nav);

            $(icon_prefix + "ui-icon-seek-end" + icon_suffix)
                .click(gotoLast)
                .appendTo($nav);

            $container.find(".ui-icon-container")
                .hover(function () {
                    $(this).toggleClass("ui-state-hover");
                });

            $container.children().wrapAll("<div class='slick-pager' />");

            if ($.QueryString["debug"]) {
                var table = '<div class="grid-header" id="toInspect" style="margin-top:35px; padding: 5px; width: 99%;"><h3><i class="icon-zoom-in"></i> Variants to inspect</h3>';
                table += '<table class="table" id="inspectTable">';
                table += '<thead id="inspectHead" style="display: none;">';
                table += '<tr> <th>Position</th> <th>Reference</th> <th>Variant</th> <th>Expected Variant</th> <th>Remove</th></tr>';
                table += '</thead>';
                table += '<tbody id="inspectBody"></tbody></table> <div id="manualInspectAdd" class="btn">Add Manually</div>';
                table += '<div id="exportInspect" class="btn btn-primary" style="margin-left: 10px;">Export</div>';
                table += '<div id="inspectOutput" style="padding-top: 10px;"></div> </div>';
                $container.parent().append(table);
            }


        }

        function updatePager(pagingInfo) {
            var state = getNavState();

            $container.find(".slick-pager-nav span").removeClass("ui-state-disabled");
            if (!state.canGotoFirst) {
                $container.find(".ui-icon-seek-first").addClass("ui-state-disabled");
            }
            if (!state.canGotoLast) {
                $container.find(".ui-icon-seek-end").addClass("ui-state-disabled");
            }
            if (!state.canGotoNext) {
                $container.find(".ui-icon-seek-next").addClass("ui-state-disabled");
            }
            if (!state.canGotoPrev) {
                $container.find(".ui-icon-seek-prev").addClass("ui-state-disabled");
            }

            if (pagingInfo.pageSize == 0) {
                $status.text("Showing " + pagingInfo.totalRows + " rows");
            } else {
                $status.text("Showing page " + (pagingInfo.pageNum + 1) + " of " + (state.lastPage + 1));
            }
        }

        init();

        //$(".slick-pager-settings").find(".ui-icon-lightbulb").attr('title','Toggle paging options.');
    }

    // Slick.Controls.Pager
    $.extend(true, window, { Slick: { Controls: { Pager: SlickGridPager }}});
})(jQuery);
