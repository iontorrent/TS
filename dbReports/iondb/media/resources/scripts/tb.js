/**
 * 
 */
;(function($, undefined) {
    var tb = window.tb = window.tb || {};
    
    tb.toString = function(value) {
        return (value !== undefined && value !== null) ? value : "";
    };
    
    tb.prettyPrintRunName = function(str) {
    	/*
	    	console.log(tb.prettyPrintRunName(undefined));
			console.log(tb.prettyPrintRunName(null));
			console.log(tb.prettyPrintRunName(''));
			console.log(tb.prettyPrintRunName('a'));
			console.log(tb.prettyPrintRunName('R_2012_08_08_13_20_22_e37e66d143'));
    	 */
    	return str && str.replace(/R_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_/,'');
    };
    
    
    tb.properties = function(object) {
    	var keys = [];
		for (key in object) {
		  keys.push(key);
		}
		return keys;
		// var numberOfKeys = keys.length;
    };
    tb.RunModes = [
    	{
		runMode:'pe',
    	description: 'Paired-End',
    	shortDescription: 'PE'
    	}
    	,{
		runMode:'single',
    	description: 'Forward',
    	shortDescription: 'F'
    	}
    ];
    tb.runModeShortDescription = function(runMode) {
    	for(i in tb.RunModes) {
    		if (runMode == tb.RunModes[i].runMode)
    			return tb.RunModes[i].shortDescription
    	}
    	return ''
    }
    tb.runModeDescription = function(runMode) {
    	for(i in tb.RunModes) {
    		if (runMode == tb.RunModes[i].runMode)
    			return tb.RunModes[i].description
    	}
    	return ''
    }
    
})();

/**
 * 
 */
!function ($) {
	var tb = window.tb;
    tb.toggleContent = function() {
    	$('[data-toggle-content^=toggle]').toggle(
			function(e){
				e.preventDefault()
				var that = $(this)
				$('#' + that.data('toggle-content')).slideDown()
				that.find('i').removeClass().addClass('icon-minus')
			} 
			, function(e){
				e.preventDefault()
				var that = $(this)
				$('#' + that.data('toggle-content')).slideUp('fast')
				that.find('i').removeClass().addClass('icon-plus')
			}
			// , function(e){
				// alert('here')
			// }
		);
    }	
	$(function () {
		var tb = window.tb;
	  	tb.toggleContent();
		tb._RunTypes = null;
	    
	    tb.runTypeDescription = function(runType) {
	    	var helper = function() {
		    	if (tb._RunTypes) {
		    		for (i in tb._RunTypes) {
		    			if (tb._RunTypes[i] && tb._RunTypes[i].runType == runType) {
		    				return tb._RunTypes[i].description;
		    			}
		    		}
		    	}
		    	return '';
	    	}
	    	if (tb._RunTypes == 'undefined' || tb._RunTypes == null ) {
	    		$.ajax({
	    			url:'/rundb/api/v1/runtype/',
	    			dataType: 'json',
	    			async: false,
	    			success: function(data) {
	    				if (data) {
	    					tb._RunTypes = data && data.objects;
	    				}
    				}
    			});
	    	}
    		return helper();
	    	
	    };			
		tb._RunModes = null;
	    
	    tb.runTypeDescription = function(runType) {
	    	var helper = function() {
		    	if (tb._RunTypes) {
		    		for (i in tb._RunTypes) {
		    			if (tb._RunTypes[i] && tb._RunTypes[i].runType == runType) {
		    				return tb._RunTypes[i].description;
		    			}
		    		}
		    	}
		    	return '';
	    	}
	    	if (tb._RunTypes == 'undefined' || tb._RunTypes == null ) {
	    		$.ajax({
	    			url:'/rundb/api/v1/runtype/',
	    			dataType: 'json',
	    			async: false,
	    			success: function(data) {
	    				if (data) {
	    					tb._RunTypes = data && data.objects;
	    				}
    				}
    			});
	    	}
    		return helper();
	    	
	    };		
		
	});
	
   
}(window.jQuery);
/**
 * 
 */
(function($) {  
	  
})(jQuery); 

(function($) {
    $.QueryString = (function(a) {
        if (a == "") return {};
        var b = {};
        for (var i = 0; i < a.length; ++i)
        {
            var p=a[i].split('=');
            if (p.length != 2) continue;
            b[p[0]] = decodeURIComponent(p[1].replace(/\+/g, " "));
        }
        return b;
    })(window.location.search.substr(1).split('&'))
})(jQuery);