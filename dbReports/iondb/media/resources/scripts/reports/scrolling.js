(function ($) {
    $.reportScrollingManager = function () {
        var plugin = this;

        // Find elements
        var elements = {
            body: $("body"),
            sideNav: $("#report-side-nav"),
            topNav: $("#top-nav"),
            topNavContainer: $("#top-nav-container")
        };

        var navContainerTop = null;
        var initialRefresh = true;

        function refresh() {
            // Refresh scroll spy
            elements.body.scrollspy('refresh');

            // Get the new position for the affix offset
            navContainerTop = elements.topNavContainer.offset().top;

            // Set the width of the side nav
            // The nav's containing block becomes the window when position:fixed
            // The width now needs to be computed in js to make the width a fraction of the window minus padding
            elements.sideNav.css("width", ($(window).width() - 40) * 0.1452991452991453);

            // Scroll to plugin if there is a plugin anchor in the url
            // needs to be here after everything is resized
            if (initialRefresh) {
                initialRefresh = false;
                if (window.location.hash) {
                    var targetElement = $(window.location.hash);
                    if (targetElement.length) {
                        console.log("Scrolling to plugin: ", window.location.hash);
                        window.scrollTo(0, targetElement.offset().top - 30);
                        setTimeout(function () {
                            refresh()
                        }, 0);
                    }
                }
            }
        }

        function init() {
            // Get the init position for the affix offset
            navContainerTop = elements.topNavContainer.offset().top;

            // Setup scroll spy for highlighting the side nav bar
            elements.body.scrollspy({target: "#report-side-nav", offset: 50});

            // Setup affix for keeping the side nav bar at the top of the page
            elements.sideNav.affix({
                offset: {
                    top: function () {
                        return navContainerTop
                    }
                }
            });

            // Setup affix for keeping the top nav bar at the top of the page
            elements.topNav.affix({
                offset: {
                    top: function () {
                        return navContainerTop
                    }
                }
            });

            // Bind events
            $(window).on("ion.pluginsRefreshed", refresh);
            $(window).on("ion.messageAdded", refresh);
            $(window).on("ion.messageRemoved", refresh);

        }

        init();

        return this;
    };
}(jQuery));