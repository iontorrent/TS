$(document).ready(function() {
	kendo.notify = function(kendoGridObject){
		kendoGridObjectName = "#"+kendoGridObject.element.attr('id')
		kendoGridObject.bind("dataBound", function(e){
			_kendoGridObject = e.sender
			kendoGridObjectName = "#"+e.sender.element.attr('id')
			if (_kendoGridObject){
				if( _kendoGridObject.options.scrollable){
					// TS-5226: dropdown not visible if there are less than 3 entries in the scrollable table
					var rowCount = $(_kendoGridObject.tbody).find('tr[role=row]:visible').length;
					if (rowCount > 3) {
						$(_kendoGridObject.tbody).find('tr[role=row]:last').prev('tr[role=row]').andSelf().find('.btn-group').addClass('dropup');
						
						// Set popovers to display below 
						var popovers = $(_kendoGridObject.tbody).find('tr[role=row]:first').next('tr[role=row]').andSelf().find('span[rel="popover"]');
						$.each(popovers, function(i, elem) {
							if ($(elem).data('popover')) 
								$(elem).data('popover').options.placement = 'bottom';
						});
						popovers = null;
						// Set popovers to display above 
						popovers = $(_kendoGridObject.tbody).find('tr[role=row]:last').prev('tr[role=row]').andSelf().find('span[rel="popover"]');
						$.each(popovers, function(i, elem) {
							if ($(elem).data('popover')) 
								$(elem).data('popover').options.placement = 'top';
						});
					}
					
				}
				/*				
				 * TS-4938: dropdowns not visible within any k-grid due to k-grid td overflow being hidden 
				 * td:last-child CSS selector does not work in IE <9 therefore we have to manually set overflow:visible 
				 * any k-grid instances containing a .dropdown-menu class 
				 */
				$(_kendoGridObject.tbody).find('.dropdown-menu').parents('td').css({overflow:'visible'});
			}
		});
	};
}); 
