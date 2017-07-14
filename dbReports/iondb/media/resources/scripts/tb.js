/**
 * Create the TB global namespace variable.  This will allow us to define various functionality
 * without polluting the global namespace.
 */
(function(window) {
   "use strict";
   var TB = {
       /**
         * Ensures that a chain of objects exist for namespacing purposes.
         *
         * @param {string} dotDelimited Dot delimited string that specifies the namespace
         * @return The last object in the namespace chain
         */
       namespace: function(dotDelimited) {
           var parts = dotDelimited.split('.'),
               current = window,
               part;
           for(var i = 0; i < parts.length; i++) {
               part = parts[i];
               // Add object to the chain if not already there
               if(current[part] === undefined) {
                   current[part] = {};
               }
               // Advance down the chain
               current = current[part];
           }
           // Return the last namespace chain object
           return current;
       }
   };
 
   // Add the top level object as a global
   window.TB = TB;
}(this));

TB.namespace('TB.utilities.browser');
TB.utilities.browser.selectAutoExpand = function() {
    /** TS-4640: IE6,7,8 long text within fixed width <select> is clipped, set width:auto temporarily */
    if ($.browser.msie && $.browser.version < 9) {
        $('select').bind('focus mouseover', function() {
            $(this).addClass('expand').removeClass('clicked');
        }).bind('click', function() {
            if ($(this).hasClass('clicked')) {
                $(this).blur();
            }
            $(this).toggleClass('clicked');
        }).bind('mouseout', function() {
            if (!$(this).hasClass('clicked')) {
                $(this).removeClass('expand');
            }
        }).bind('blur', function() {
            $(this).removeClass('expand clicked');
        });
    }
};

TB.namespace('TB.utilities.form');
TB.utilities.form.focus = function() {
    $(':input:enabled:visible:first').focus();
};
$(function () {
    TB.utilities.form.focus();
});
/**
 * 
 */
;(function($, undefined) {
    TB.toString = function(value) {
        return (value !== undefined && value !== null) ? value : "";
    };
    
    TB.prettyPrintRunName = function(str) {
        /*
            console.log(TB.prettyPrintRunName(undefined));
            console.log(TB.prettyPrintRunName(null));
            console.log(TB.prettyPrintRunName(''));
            console.log(TB.prettyPrintRunName('a'));
            console.log(TB.prettyPrintRunName('R_2012_08_08_13_20_22_e37e66d143'));
         */
        return str && str.replace(/R_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_/,'');
    };
    
    
    TB.properties = function(object) {
        var keys = [];
        for (key in object) {
          keys.push(key);
        }
        return keys;
    };
    TB.RunModes = [
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
    TB.runModeShortDescription = function(runMode) {
        for(i in TB.RunModes) {
            if (runMode == TB.RunModes[i].runMode)
                return TB.RunModes[i].shortDescription
        }
        return ''
    }
    TB.runModeDescription = function(runMode) {
        for(i in TB.RunModes) {
            if (runMode == TB.RunModes[i].runMode)
                return TB.RunModes[i].description
        }
        return ''
    }
    
})();

/**
 * 
 */
!function ($) {
    TB.toggleContent = function() {
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
        TB.toggleContent();
        TB._RunTypes = null;
        TB._ApplicationGroups = null;
        
        TB.runTypeDescription = function(runType) {
            var helper = function() {
                if (TB._RunTypes) {
                    for (i in TB._RunTypes) {
                        if (TB._RunTypes[i] && TB._RunTypes[i].runType == runType) {
                            return TB._RunTypes[i].description;
                        }
                    }
                }
                return '';
            }
            if (TB._RunTypes == 'undefined' || TB._RunTypes == null ) {
                $.ajax({
                    url:'/rundb/api/v1/runtype/',
                    dataType: 'json',
                    async: false,
                    success: function(data) {
                        if (data) {
                            TB._RunTypes = data && data.objects;
                        }
                    }
                });
            }
            return helper();
            
        };

        TB.runTypeApplicationDescription = function(runType, application, applicationCategoryName) {
            var helper = function() {
                if (TB._RunTypes && TB._ApplicationGroups) {
                    for (i in TB._ApplicationGroups) {
                        if (TB._ApplicationGroups[i] && (TB._ApplicationGroups[i].name == application || TB._ApplicationGroups[i].description == application)) {
                        	if (TB._ApplicationGroups[i].applications.length <= 1) {
                        	    if (applicationCategoryName) {                   	    
                            		return applicationCategoryName + " | " + TB._ApplicationGroups[i].description;
                            	} else {
                            		return TB._ApplicationGroups[i].description;
                            	}
                            }
                            else {
                            	if (applicationCategoryName) { 
                            		return applicationCategoryName + " | " + TB.runTypeDescription(runType);
                            	} else {
                            		return TB.runTypeDescription(runType);
                            	}
                            }
                        }
                    }
                }
                return '';
            }
            if (TB._RunTypes == 'undefined' || TB._RunTypes == null ) {
                $.ajax({
                    url:'/rundb/api/v1/runtype/',
                    dataType: 'json',
                    async: false,
                    success: function(data) {
                        if (data) {
                            TB._RunTypes = data && data.objects;
                        }
                    }
                });
            }           
            if (TB._ApplicationGroups == 'undefined' || TB._ApplicationGroups == null ) {
                $.ajax({
                    url:'/rundb/api/v1/applicationgroup/',
                    dataType: 'json',
                    async: false,
                    success: function(data) {
                        if (data) {
                            TB._ApplicationGroups = data && data.objects;
                        }
                    }
                });
            }
            return helper();
            
        };
                   
        TB._RunModes = null;
        
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