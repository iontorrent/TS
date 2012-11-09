$(document).ready(function() {
	$("#configure").mouseover(function(){
    	$("#gear").removeClass().addClass("gear-active");
    }).mouseout(function(){
		$("#gear").removeClass().addClass("gear-inactive");    			
    }); 
});

/**
* @param {string|Object} dialogPanel jQuery selector for the panel
* @param {string=} cancelButtonSelector jQuery selector for the cancel button.
* The panel is used as the root element.
* @param width the width in px of the dialog, or "auto". default is 800.
*/
function createDialog(dialogPanel, cancelButtonSelector, width) {
	width = width || 800;
	$(dialogPanel).dialog({
		autoOpen: false,
		modal: true,
		width: width,
		open: function(event, ui) {
			$('.ui-dialog-titlebar-close').html('<span>CLOSE X</span>');
		}
	});
	if (cancelButtonSelector != null) {
		$(dialogPanel).dialog().find(cancelButtonSelector).click(function(e) {
			$(dialogPanel).dialog('close');
			return false;
		});
	}
} 

$(document).ready(function(){
	if ($(".dynamic-navbar").length <= 0) return
	
	//$(".dynamic-navbar").empty();
	$.each($("*[dynamic-navbar-section]"), function(element) {
		var href = $.trim($(this).attr("href")) 
			|| $.trim($(this).attr("dynamic-navbar-section"));
		var title =  $.trim($(this).attr("dynamic-navbar-section-title"))
			|| $.trim($(this).text());
		var section = '<li><a href="' + href + '" class="navitem">' + title + '</a></li>';
		$(".dynamic-navbar").append(section);
	});
});

$(document).ready(function(){
    var $win = $(window)
      , $nav = $('.page-nav')
      , navTop = $('.page-nav').length && $('.page-nav').offset().top + 10
      , isFixed = 0

    // processScroll()

    // hack sad times - holdover until rewrite for 2.1
    $nav.on('click', function () {
      if (!isFixed) setTimeout(function () {  $win.scrollTop($win.scrollTop() - 180) }, 10)
    })

    // $win.on('scroll', processScroll)
    // 
    // function processScroll() {
    //   var i, scrollTop = $win.scrollTop()
    //   if (scrollTop >= navTop && !isFixed) {
    //     isFixed = 1
    //     $nav.addClass('page-nav-fixed')
    //   } else if (scrollTop <= navTop && isFixed) {
    //     isFixed = 0
    //     $nav.removeClass('page-nav-fixed')
    //   }
    // }
});

function initTooltip(that) {
	if ($.fn.tooltip){
	    $(that).tooltip({
	      selector: "*[rel=tooltip]",
	      placement: "bottom"
	    });
   	}
}
function hideTooltip(that) {
	that.find('[rel="tooltip"]').each(function(i, elem) {
		$(elem).tooltip('hide');
	});
}
function initDropdownToggle(){
   	if ($.fn.dropdown) {
   		$('.dropdown-toggle').dropdown();
   	}
}

$(document).ready(function(){
	//Enables body to be scanned for any tooltips occurrences
	initTooltip($('body'));
	
	initDropdownToggle();
});

function getParameterByName(name) {
  name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
  var regexS = "[\\?&]" + name + "=([^&#]*)";
  var regex = new RegExp(regexS);
  var results = regex.exec(window.location.search);
  if(results == null)
    return "";
  else
    return decodeURIComponent(results[1].replace(/\+/g, " "));
}


(function($) {
	$.fn.exists = function () {
		/*
		 * helper function to test if the result of a selector finds anything
		 * See http://stackoverflow.com/a/920322
		 */
    	return this.length !== 0;
	}
	$.fn.attr = function(name, value) {
		/**
		 * Overrides the native jQuery attr function to handle a special case of browsers not returning 
		 * the exact value contained within <a>'s href attribute. The issue is noticed by IE6/7 browsers, Opera, plus others 
		 * See http://stackoverflow.com/a/2343009, http://msdn.microsoft.com/en-us/library/cc848861(VS.85).aspx
		 */
		var str = jQuery.access( this, jQuery.attr, name, value, arguments.length > 1 );
		if (name == "href" && value == undefined && str !== undefined) {
			var base = window.location.href.substring(0, window.location.href.lastIndexOf("/") + 1);
			str = str.replace(base, "");
		}
		return str;
	};
	$.fn.serializeJSON = function() {
		var json = {};
		jQuery.map($(this).serializeArray(), function(n, i) {
			if (json[n['name']]) {
				if (!$.isArray(json[n['name']])) {
					json[n['name']] = $.makeArray(json[n['name']]);
				}
				json[n['name']].push(n['value']);
			} else {
				json[n['name']] = n['value'];
			}
		});
		return json;
	};
	var _oldShow = $.fn.show;

    $.fn.show = function(speed, oldCallback) {
        return $(this).each(function() {
                var
                        obj         = $(this),
                        newCallback = function() {
                                if ($.isFunction(oldCallback)) {
                                        oldCallback.apply(obj);
                                }

                                obj.trigger('afterShow');
                        };

                // you can trigger a before show if you want
                obj.trigger('beforeShow');

                // now use the old function to show the element passing the new callback
                _oldShow.apply(obj, [speed, newCallback]);
        });
    };
    
    if (!String.prototype.startsWith) {
        String.prototype.startsWith = function (str) {
            return !this.indexOf(str);
        };
    }
    
})(jQuery); 

function buildParameterMap(options) {
	map = {
		offset: (options.page - 1) * options.pageSize,
		limit: options.pageSize
	};
	if (options.filter) {
		// parse filter options
		for (var i in options.filter.filters) {
			var filter = options.filter.filters[i];
			if (filter.value)
				map[filter.field + filter.operator] = filter.value;
		}
	}	
	if (options.sort != null && options.sort.length != 0)
		map.order_by = format_order_by(options.sort[0]);
		
	return map;
}
function format_order_by(sortObj) {
	return (sortObj.dir == 'desc') ? '-' + sortObj.field : sortObj.field; 
}

function refreshKendoGrid(gridId) {
	var grid = $(gridId).data("kendoGrid");
	var totalPagesBefore = grid.dataSource.totalPages();
	var pageBeforeRefresh = grid.dataSource.page();
	console.log('refreshKendoGrid: Before :: totalPages: ', totalPagesBefore, 'page: ', pageBeforeRefresh);
	//reload grid's datasource
	grid.dataSource.read(); 
	console.log('refreshKendoGrid: After :: totalPages: ', grid.dataSource.totalPages(), 'page: ', grid.dataSource.page());
	if (pageBeforeRefresh > grid.dataSource.totalPages()) {
		grid.dataSource.page(grid.dataSource.totalPages());
	}
	// refreshes the grid
	grid.refresh();
}	
