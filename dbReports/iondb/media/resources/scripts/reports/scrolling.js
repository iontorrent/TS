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
        var previousElementId = null;

        function refresh() {
            // Refresh scroll spy
            elements.body.scrollspy('refresh');
            $('[data-spy="scroll"]').each(function () {
                $(this).scrollspy('refresh');
            });
            // Get the new position for the affix offset
            navContainerTop = elements.topNavContainer.offset().top;

            // Set the width of the side nav
            // The nav's containing block becomes the window when position:fixed
            // The width now needs to be computed in js to make the width a fraction of the window minus padding
            elements.sideNav.css("width", ($(window).width() - 40) * 0.1452991452991453);
        }

        function storePrevious() {
            previousElementId = null;
            var activeAnchor = $("#report-side-nav").find("li.active > a");
            if (activeAnchor) {
                previousElementId = activeAnchor.attr("href");
            }
            console.log("Storing previous element id: ", previousElementId);
        }

        function scrollPrevious() {
            if (initialRefresh) {
                // Scroll to plugin if there is a plugin anchor in the url
                // needs to be here after everything is resized
                initialRefresh = false;
                if (window.location.hash) {
                    var targetElement = $(window.location.hash);
                    if (targetElement.length) {
                        console.log("Scrolling to section in url: ", window.location.hash);
                        setTimeout(function () {
                            refresh();
                            window.scrollTo(0, targetElement.offset().top);
                            $(window).trigger("scroll");
                        }, 0);
                    }
                }
            } else if (previousElementId) {
                refresh();
                // Scroll to a plugin after refreshing plugins
                console.log("Scrolling to previous id: ", previousElementId);
                refresh();
                setTimeout(function () {
                    window.scrollTo(0, $(previousElementId).offset().top);
                    previousElementId = null;
                }, 100);
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
            $(window).on("ion.pluginsReload", storePrevious);
            $(window).on("ion.pluginsReloaded", scrollPrevious);

            $(window).on("ion.pluginsRefreshed", refresh);
            $(window).on("ion.messageAdded", refresh);
            $(window).on("ion.messageRemoved", refresh);

        }

        init();

        return this;
    };
}(jQuery));