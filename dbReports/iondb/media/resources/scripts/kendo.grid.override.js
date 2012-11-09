$(document).ready(function() {
	kendo.notify = function(kendoGridObject){
		kendoGridObjectName = "#"+kendoGridObject.element.attr('id')
		console.log("a Kendo Grid instance has been initialized/constructed "+kendoGridObjectName);
		
		kendoGridObject.bind("dataBound", function(e){
			_kendoGridObject = e.sender
			kendoGridObjectName = "#"+e.sender.element.attr('id')
			console.log("bound to #" + kendoGridObjectName + " dataBound event");
			if (_kendoGridObject){
				if( _kendoGridObject.options.scrollable){
					console.log("Grid is scrollable");
					console.log(_kendoGridObject)
					// TS-5226: dropdown not visible if there are less than 3 entries in the scrollable table
					var rowCount = $(_kendoGridObject.tbody).find('tr:visible').length;
					if (rowCount > 2) {
						$(_kendoGridObject.tbody).find('tr:last').prev('tr').andSelf().find('.btn-group').addClass('dropup');
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
