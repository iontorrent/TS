"use strict";
var dashboardApp = {
    // Fetch refs to static elements on the page
    elements: {
        refreshButtonActive: $("#refresh-button-active"),
        refreshButtonInactive: $("#refresh-button-inactive"),

        summaryFragmentContainer: $("#summary-fragment-container"),
        runsFragmentContainer: $("#runs-fragment-container"),
        instrumentsFragmentContainer: $("#instruments-fragment-container"),

        timeSpanSelectContainer: $("#time-span-container"),
        timeSpanSelect: $("#time-span"),
    },
    // Mutable state
    timeSpan: window.localStorage.getItem("dashboard-time-span") || "24hours",
    allOptionShown: false,
    pendingRefreshTimeout: null,
    pendingRefreshRequest: null,
    autoRefreshEnabled: window.localStorage.getItem("dashboard-auto-refresh") || true,
    refreshSeconds: 30,

    init: function () {
        this.setupTimeSpanSelector();
        this.setupAutoRefreshButton();
        this.setupSummaryFragment();
        this.setupInstrumentsFragment();
        // Trigger the fist refresh
        this.refreshRunsTable();
    },
    clearPendingRefresh: function () {
        // Need to kill the pending timeout
        if (this.pendingRefreshTimeout) {
            window.clearTimeout(this.pendingRefreshTimeout);
        }
        // Need to kill the pending request if it is started
        if (this.pendingRefreshRequest) {
            this.pendingRefreshRequest.abort();
        }
    },
    setupAutoRefreshButton: function () {
        //Set the initial state of the button
        if (this.autoRefreshEnabled) {
            this.elements.refreshButtonActive.show();
        } else {
            this.elements.refreshButtonInactive.show();
        }
        // When the "Stop Refresh" button is clicked
        this.elements.refreshButtonActive.click(function () {
            this.autoRefreshEnabled = false;
            window.localStorage.setItem("dashboard-auto-refresh", this.autoRefreshEnabled);
            // Need to stop any pending timeout
            this.clearPendingRefresh();
            this.elements.refreshButtonActive.hide();
            this.elements.refreshButtonInactive.show();
        }.bind(this));
        // When the "Auto Refresh" button is clicked
        this.elements.refreshButtonInactive.click(function () {
            this.autoRefreshEnabled = true;
            window.localStorage.setItem("dashboard-auto-refresh", this.autoRefreshEnabled);
            this.refreshRunsTable();
            this.elements.refreshButtonInactive.hide();
            this.elements.refreshButtonActive.show();
        }.bind(this));
    },
    setupTimeSpanSelector: function () {
        // Set select to value from local storage
        this.elements.timeSpanSelect.val(this.timeSpan);
        // When select is changed, save it and fetch the new table
        this.elements.timeSpanSelect.change(function (event) {
            this.timeSpan = event.target.value;
            // We never want to save the hidden __all__ option
            if (this.timeSpan !== "__all__") {
                window.localStorage.setItem("dashboard-time-span", this.timeSpan);
            }
            // We want to kill the pending refresh when changing the time span
            this.clearPendingRefresh();
            this.refreshRunsTable(true);
        }.bind(this));
        // Hidden all option used for internal debugging
        // Hold shift when clicking to enable the hidden menu
        this.elements.timeSpanSelectContainer.on("click", function (event) {
            if (!this.allOptionShown && event.shiftKey) {
                this.elements.timeSpanSelect.append(
                    $("<option/>", {text: "All", value: "__all__"})
                );
                this.allOptionShown = true;
                this.elements.timeSpanSelect.selectpicker('refresh');
            }
        }.bind(this));
    },
    setupSummaryFragment: function () {
    },
    setupRunsFragment: function () {
        // Samples popover needs to be reinitialized after each html write
        $(".samples").popover({
            html: true,
            trigger: 'hover',
            content: function () {
                return $($(this).data('select')).html();
            }
        });
    },
    setupInstrumentsFragment: function () {
        // Instruments popover needs to be reinitialized after each html write
        $(".instrument-alarms").popover({
            html: true,
            trigger: 'hover',
            content: function () {
                return $(this).data('alarms');
            }
        });
    },
    refreshRunsTable: function (showLoading) {
        // Clear any tooltips TS-16648
        $(".run-type-icon").tooltip('destroy');
        // When refreshing the runs list, don't clear the html first
        if (showLoading) {
            this.elements.runsFragmentContainer.html("<span class='muted'>Loading...</span>")
        }
        // Store the reference to the ajax call to be able to abort it
        this.pendingRefreshRequest = $.ajax({
            url: "/home/fragments?time_span=" + this.timeSpan
        }).done(function (fragments) {
            this.elements.summaryFragmentContainer.html(fragments["summary"]);
            this.elements.runsFragmentContainer.html(fragments["runs"]);
            this.elements.instrumentsFragmentContainer.html(fragments["instruments"]);
            // Run anything that needs to be redone each load
            this.setupSummaryFragment();
            this.setupRunsFragment();
            this.setupInstrumentsFragment();
        }.bind(this)).always(function () {
            // If auto refresh, kick off the next poll
            if (this.autoRefreshEnabled) {
                this.pendingRefreshTimeout = setTimeout(function () {
                    this.refreshRunsTable();
                }.bind(this), this.refreshSeconds * 1000);
            }
        }.bind(this));
    }
};

dashboardApp.init();
