$(function () {
    $(".expandable li a").click(function () {
        var curClass = $(this).parent().attr("class");
        var toRemove = (curClass == "closed") ? "closed" : "expanded";
        var toAdd = (curClass == "closed") ? "expanded" : "closed";
        $(this).parent().switchClass(toRemove, toAdd, 500);
        return false;
    });

    $("a.sectiontitle").click(function () {
        $(this).toggler();
        return false;
    });

    $("img.over")
        .bind('mouseover', function () {
            this.src = this.src.replace(/^(.+)-off(\.[a-z]+)$/, "$1-on$2");
        })
        .bind('mouseout', function () {
            this.src = this.src.replace(/^(.+)-on(\.[a-z]+)$/, "$1-off$2");
        })
        .each(function () {
            this.preloaded = new Image;
            this.preloaded.src = this.src.replace(/^(.+)(\.[a-z]+)$/, "$1-on$2");
        }
    );

    //message hider
    $("a.hide").click(function () {
        $(this).parents("div.section").fadeOut(600);
        return false;
    });

    $("a.zoomable").prettyPhoto();
    $("a.showlarge").prettyPhoto();
    var zoomIcon = '<span class="zoom"><img src="/site_media/plugin_template_styling/images/zoom.gif" height="20" width="21" alt="Zoom"/></span>';
    $("a.zoomable").append(zoomIcon);

    //build navigation
    if ($("div#nav ul li").length == false) {
        $.each($("a[name]"), function (anchor) {
            //append to navigation
            var name = $(this).attr("name");
            var title = $(this).attr("title");
            var section = '<li><a href="#' + name + '" class="navitem">' + title + '</a></li>';
            $("div#nav ul").append(section);
        });
        //finally, append the clearing listitem
        $("div#nav ul").append('<li class="clearing"></li>');

        //now, hook up the click action for each nav item
        $(".links ul li a").click(function () {
            //find the named anchor, and expand it if it isn't already
            var anchor = $(this).attr("href");
            anchor = anchor.replace("#", "");

            //toggle if sectioncontainer is hidden
            var sec = $("a[name='" + anchor + "']").parent();
            if (sec.attr("class").indexOf("collapsed") > -1) {
                sec.find("a.sectiontitle").toggler();
            }
        });
    } else {
        //hook up the click action for each nav item
        $(".links ul li a").click(function () {
            //find the named anchor, and expand it if it isn't already
            var anchor = $(this).attr("href");
            anchor = anchor.replace("#", "");

            //toggle if sectioncontainer is hidden
            var sec = $("a[name='" + anchor + "']").parent();
            if (sec.attr("class").indexOf("collapsed") > -1) {
                sec.find("a.sectiontitle").toggler();
            }
        });
    }
});

(function ($) {
    jQuery.fn.toggler = function () {
        var current = $(this).find("h2").css("backgroundImage");
        var sectionType = (current.indexOf("sectionError") > -1) ? "Error" : "";
        sectionType = (sectionType == "" && current.indexOf("sectionWarning") > -1) ? "Warning" : sectionType;
        sectionType = (sectionType == "" && current.indexOf("sectionSuccess") > -1) ? "Success" : sectionType;
        var a = $(this);
        if (current.indexOf("Minus") > -1) {
            $(this).parent().addClass("collapsed", 500, function () {
                a.find("h2").css("backgroundImage", "url(/site_media/plugin_template_styling/images/section" + sectionType + "Plus.png)");
            });
        } else {
            $(this).parent().removeClass("collapsed", 500, function () {
                a.find("h2").css("backgroundImage", "url(/site_media/plugin_template_styling/images/section" + sectionType + "Minus.png)");
            });
        }
    }
})(jQuery);
