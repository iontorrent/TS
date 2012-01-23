 /*
 * TipTipX 1.0
 * Copyright 2010 Tomasz Szymczyszyn
 *
 * Based on TipTip
 * Copyright 2010 Drew Wilson
 * www.drewwilson.com
 * code.drewwilson.com/entry/tiptip-jquery-plugin
 *
 * This plug-in is dual licensed under the MIT and GPL licenses:
 * http://www.opensource.org/licenses/mit-license.php
 * http://www.gnu.org/licenses/gpl.html
 */

(function($){
  function TipTipX(elements, options) {
	 	this.opts = $.extend({}, TipTipX.defaults, options);
    this.elements = elements;
  
    var tag = this.opts.tag;

	 	// Setup tip tip elements and render them to the DOM
	 	if ($("#tiptip_holder-" + tag).length <= 0) {
	 		this.tiptip_holder = $('<div class="tiptip_holder" id="tiptip_holder-' + tag + '" style="max-width:'+ this.opts.maxWidth +';"></div>');
			this.tiptip_content = $('<div class="tiptip_content" id="tiptip_content-' + tag + '"></div>');
			this.tiptip_arrow = $('<div class="tiptip_arrow" id="tiptip_arrow-' + tag + '"></div>');
			$("body").append(this.tiptip_holder.html(this.tiptip_content).prepend(this.tiptip_arrow.html('<div class="tiptip_arrow_inner" id="tiptip_arrow_inner-' + tag + '"></div>')));
		} else {
			this.tiptip_holder = $("#tiptip_holder-" + tag);
			this.tiptip_content = $("#tiptip_content-" + tag);
			this.tiptip_arrow = $("#tiptip_arrow-" + tag);
		};
  };

  TipTipX.defaults = {
      tag: "default",
			activation: "hover",
			keepAlive: false,
			maxWidth: "600px",
			edgeOffset: 8,
			defaultPosition: "bottom",
			delay: 400,
			fadeIn: 200,
			fadeOut: 200,
			attribute: "title",
			content: false, // HTML or String to fill TipTipX with
		  enter: function(){},
		  exit: function(){}
  };

  TipTipX.prototype.showNow = function() {
    var that = this;
    var activate_all = function() {
      $(that.elements).each(function() {
  			var org_elem = $(this);
        if(that.opts.content){
          var org_title = that.opts.content;
        } else {
          var org_title = org_elem.attr(that.opts.attribute);
        };

        that.activate_tiptip(org_elem, org_title);
      });
    };

		activate_all();

    this.tiptip_holder.data("resize-handler", activate_all);
    $(window).resize(activate_all);
  };

  TipTipX.prototype.clearNow = function() {
    var that = this;
		$(this.elements).each(function() {
			var org_elem = $(this);
      that.timeout = false;
      that.deactivate_tiptip();
    });

    $(window).unbind("resize", this.tiptip_holder.data("resize-handler"));
  };

  TipTipX.prototype.setup = function() {
    var that = this;
		$(this.elements).each(function() {
			var org_elem = $(this);
			if(that.opts.content){
				var org_title = that.opts.content;
			} else {
				var org_title = org_elem.attr(that.opts.attribute);
			};

			if(org_title != ""){
				if(!that.opts.content){
					org_elem.removeAttr(that.opts.attribute); //remove original Attribute
				}

				that.timeout = false;

				if(that.opts.activation == "hover"){
					org_elem.hover(function(){
						that.activate_tiptip(org_elem, org_title);
					}, function(){
						if(!that.opts.keepAlive){
							that.deactivate_tiptip();
						}
					});
					if(that.opts.keepAlive){
						tiptip_holder.hover(function(){}, function(){
							that.deactivate_tiptip();
						});
					}
				} else if(that.opts.activation == "focus"){
					org_elem.focus(function(){
						that.activate_tiptip(org_elem, org_title);
					}).blur(function(){
            that.deactivate_tiptip();
					});
				} else if(that.opts.activation == "click"){
					org_elem.click(function(){
						that.activate_tiptip(org_elem, org_title);
						return false;
					}).hover(function(){},function(){
						if(!that.opts.keepAlive){
							that.deactivate_tiptip();
						}
					});
					if(that.opts.keepAlive){
						tiptip_holder.hover(function(){}, function(){
							that.deactivate_tiptip();
						});
					}
				}
			}				
		});
  };

  TipTipX.prototype.remove_position_classes = function() {
    this.tiptip_holder.removeClass("tip_left tip_right tip_top tip_bottom tip_left_top tip_left_bottom tip_right_top tip_right_bottom");
  };

  TipTipX.prototype.position_at_top = function(preserve_left_right) {
    if (preserve_left_right)
      this.t_class += "_top";
    else
      this.t_class = "_top";

    this.arrow_top = this.tip_h;
    this.marg_top = Math.round(this.top - (this.tip_h + 5 + this.opts.edgeOffset));
  };

  TipTipX.prototype.position_at_bottom = function(preserve_left_right) {
    if (preserve_left_right)
      this.t_class += "_bottom";
    else
      this.t_class = "_bottom";

    this.arrow_top = -12;						
    this.marg_top = Math.round(this.top + this.height + this.opts.edgeOffset);
  };

  TipTipX.prototype.position_at_left = function() {
    this.t_class = "_left";
    this.arrow_top = Math.round(this.tip_h - 13) / 2;
    this.arrow_left =  Math.round(this.tip_w);
    this.marg_left = Math.round(this.left - (this.tip_w + this.opts.edgeOffset + 5));
    this.marg_top = Math.round(this.top + this.h_diff);
  };

  TipTipX.prototype.position_at_right = function() {
    this.t_class = "_right";
    this.arrow_top = Math.round(this.tip_h - 13) / 2;
    this.arrow_left = -12;
    this.marg_left = Math.round(this.left + this.width + this.opts.edgeOffset);
    this.marg_top = Math.round(this.top + this.h_diff);
  };

  TipTipX.prototype.position_at_left_top = function() {
    this.position_at_left();
    this.position_at_top(true);
  };

  TipTipX.prototype.position_at_right_top = function() {
    this.position_at_right();
    this.position_at_top(true);
  };

  TipTipX.prototype.position_at_left_bottom = function() {
    this.position_at_left();
    this.position_at_bottom(true);
  };

  TipTipX.prototype.position_at_right_bottom = function() {
    this.position_at_right();
    this.position_at_bottom(true);
  };

  TipTipX.prototype.activate_tiptip = function(org_elem, org_title) {
    this.tiptip_content.html(org_title);
    this.tiptip_holder.hide().css("margin","0");
    this.remove_position_classes();
    this.tiptip_arrow.removeAttr("style");

    this.opts.enter.call(this, this.tiptip_content);

    this.top = parseInt(org_elem.offset()['top']); //target element top-offset
    this.left = parseInt(org_elem.offset()['left']); //target element left-offset
    this.width = parseInt(org_elem.outerWidth()); //target element width
    this.height = parseInt(org_elem.outerHeight()); //target element height
    this.tip_w = this.tiptip_holder.outerWidth(); //tip width
    this.tip_h = this.tiptip_holder.outerHeight(); //tip height
    this.w_diff = Math.round((this.width - this.tip_w) / 2);
    this.h_diff = Math.round((this.height - this.tip_h) / 2);
    this.marg_left = Math.round(this.left + this.w_diff);
    this.marg_top = Math.round(this.top + this.height + this.opts.edgeOffset);
    this.arrow_top = "";
    this.arrow_left = Math.round(this.tip_w - 12) / 2;

    this.t_class = "";

    if (this.opts.position) {
      switch (this.opts.position) {
        case "left_top": this.position_at_left_top(); break;
        case "top": this.position_at_top(); break;
        case "right_top": this.position_at_right_top(); break;
        case "left": this.position_at_left(); break;
        case "right": this.position_at_right(); break;
        case "left_bottom": this.position_at_left_bottom(); break;
        case "bottom": this.position_at_bottom(); break;
        case "right_bottom": this.position_at_right_bottom(); break;
      };
    }

    if (! this.opts.position) {
      if (this.opts.defaultPosition.indexOf("left") == 0) {
        this.t_class = "_left";
      } 
      else if (this.opts.defaultPosition.indexOf("right") == 0) {
        this.t_class = "_right";
      }

      var exceeds_left = (this.left - this.tip_w - this.opts.edgeOffset) < parseInt($(window).scrollLeft());
      var exceeds_right = (this.left + this.width + this.tip_w + this.opts.edgeOffset) > parseInt($(window).width()) - parseInt($(window).scrollLeft());

      if (this.opts.position == "right" || ((exceeds_left && this.w_diff < 0) || (this.t_class == "_right" && !exceeds_right) || (this.t_class == "_left" && this.left < (this.tip_w + this.opts.edgeOffset + 5)))) {
        this.arrow_top = this.tip_h;
        this.marg_top = Math.round(this.top - (this.tip_h + 5 + this.opts.edgeOffset));
        this.position_at_right(); 
      } else if (this.opts.position == "left" || ((exceeds_right && this.w_diff < 0) || (this.t_class == "_left" && !exceeds_left))) {
        this.position_at_left();
      }

      if (! this.opts.position) {
        if (this.opts.defaultPosition.indexOf("top") != -1) {
          this.position_at_top(true);
        }
        else if (this.opts.defaultPosition.indexOf("bottom") != -1) {
          this.position_at_bottom(true);
        }
      }

      var exceeds_bottom = (this.top + this.height + this.opts.edgeOffset + this.tip_h + 8) > parseInt($(window).height()) + parseInt($(window).scrollTop());
      var exceeds_top = (this.top - (this.opts.edgeOffset + this.tip_h + 8)) < parseInt($(window).scrollTop());

      if (exceeds_bottom || (this.t_class == "_top" && !exceeds_top)) {
        this.position_at_top(! (this.t_class == "_top" || this.t_class == "_bottom"));
      } else if (exceeds_top || (this.t_class == "_bottom" && !exceeds_bottom)) { 
        this.position_at_bottom(! (this.t_class == "_top" || this.t_class == "_bottom"));
      }
    }

    if (this.t_class == "_right_top" || this.t_class == "_left_top") {
      this.marg_top = this.marg_top + 5;
    } else if(this.t_class == "_right_bottom" || this.t_class == "_left_bottom") {		
      this.marg_top = this.marg_top - 5;
    }

    if (this.t_class == "_left_top" || this.t_class == "_left_bottom") { 	
      this.marg_left = this.marg_left + 5;
    }

    this.tiptip_arrow.css({"margin-left": this.arrow_left+"px", "margin-top": this.arrow_top+"px"});
    this.tiptip_holder.css({"margin-left": this.marg_left+"px", "margin-top": this.marg_top+"px"}).addClass("tip"+this.t_class);

    if (this.timeout){ clearTimeout(this.timeout); }

    var that = this;
    this.timeout = setTimeout(function(){ that.tiptip_holder.stop(true,true).fadeIn(that.opts.fadeIn); }, that.opts.delay);	
  };

  TipTipX.prototype.deactivate_tiptip = function() {
    var that = this;
    setTimeout(function() {
      that.opts.exit.call(that, that.tiptip_content);
      if (that.timeout){ clearTimeout(that.timeout); }
      that.tiptip_holder.fadeOut(that.opts.fadeOut);
    }, 0);
  };

	$.fn.tipTip = function(options) {
    (new TipTipX(this, options)).setup();
	  return this; 	
	};

  $.fn.tipTipNow = function(options) {
    (new TipTipX(this, options)).showNow();
    return this;
  };

  $.fn.tipTipClear = function(options) {
    (new TipTipX(this, options)).clearNow();
    return this;
  };

  $.fn.tipTipDefaults = function(options) {
    $.extend(TipTipX.defaults, options);
  };

})(jQuery);  	
