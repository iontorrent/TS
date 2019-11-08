(function($) {
	$.fn.strength = function(strength, warning, percentage, caption, suffix) {
		/*
		 * strength : 
		 * 	0-19 all bars gray
		 * 	20 - 39 bar1 red others gray
		 *  40 - 59 bar1,bar2 read others gray
		 *  60 - 79 bar1,2,3 blue others gray
		 *  80 - 99 bar1,2,3,4 blue others gray
		 *  100 all bars blue
		 * warning: 
		 *  0-39 warning above bar1
		 *  40 - 59 warning above bar2
		 *  60 - 79 warning above bar3
		 *  80 - 99 warning above bar4
		 *  100 warning above bar5
		 */
		// console.debug('in stength()');
		// I thought JavaScript variable hoisting was just an academic problem
		// until I worked on this code.
		var stren, str, i, j;
		// console.debug('argument:', strength);
		if (typeof warning == 'undefined' || warning == null) {
			warning = 0;
		} else {
			warning = parseInt(warning);
		}
		var signalCaptionContainer = $('<div class="signal-caption"></div>');
		if ( typeof percentage == 'undefined' || percentage == null) {
			percentage = '?'
		} else {
			percentage = parseFloat(percentage)
		}
		if (typeof suffix == 'undefined' || suffix == null) {
			suffix = gettext('signal.js.strength.percentage.suffix') || ' %';
		}
		signalCaptionContainer.append("<p class='percentage'>" + percentage + suffix + '</p>');
		if (typeof caption == 'undefined' || caption == null) {
			caption = '';
		}
		signalCaptionContainer.append("<p class='caption'>" + caption + '</p>');
		
		$(this).each(function(i, elem) {
			$(elem).empty();
			$(elem).addClass('signal');
			var signalContainer = $('<div></div>');
			$(elem).append(signalContainer);
			$(signalContainer).addClass('signal-strength');
			
			if (typeof strength == 'undefined' || strength == null) {
				stren = 0;
				strength = 0;
			} else {
				strength = parseInt(strength);
				stren = Math.ceil(strength / 20);
				if (stren > 5) stren = 5;
				stren = stren * 20;
			}
			if (strength >= warning) {
				warning = 0;
			} else {
				$(signalContainer).addClass('signal-failqc');
			}
			$(signalContainer).addClass('strength' + stren );
			str = '';
			j = Math.ceil(warning / 20);
			// console.debug(i,j);
			for (i = 1 ; i <= 5 ; i++) {
				str += '<span class="bar' + i + '">';
				if (warning && j == i) {
					str += '<span class="signal-warning"></span>';
				}
				str += '</span>';
			}
			$(signalContainer).append(str)
			if (typeof percentage !== 'undefined' || typeof caption !== 'undefined') {
				$(elem).append(signalCaptionContainer);
			}
		});
		return this;
	};
})(jQuery);
