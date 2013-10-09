$(function(){
    CardRunView = Backbone.View.extend({
        className: "run",

		events: {
            'click .review-plan': 'review_plan_'
        },
        initialize: function () {
            _.bindAll(this, 'render', 'review_plan_', 'destroy_view');
            console.log("Card view initialized.");
            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
        },

        template: Hogan.compile($("#monitor_experiment_template").html()),

        render: function () {
            console.log(this.model.changedAttributes());
            this.$('[rel="tooltip"]').tooltip('hide');
            if (this.model.reports.length > 0) {
                var king_report = this.model.reports.at(0);
                console.log(king_report);
                var met = king_report.get("analysis_metrics");
                var qc = king_report.get("quality_metrics");
            }
            var status = this.model.get('ftpStatus');
            context = {
                exp: this.model.toJSON(),
                "prettyExpName": TB.prettyPrintRunName(this.model.get('expName')),
                "king_report": king_report && king_report.toJSON(),
                "date_string": kendo.toString(this.model.get('date'), "MM/dd/yy hh:mm tt"),
                "bead_loading": met && Math.round(met.bead / (met.total_wells - met.excluded) * 1000) / 10,
                "bead_live": met && Math.round(met.live / met.bead * 1000) / 10,
                "bead_lib": met && Math.round(met.lib / met.live * 1000) / 10,
                "usable_seq": met && qc && Math.round(qc.q0_reads / met.lib * 1000) / 10,
                "progress_flows": (status == "Complete" ? this.model.get('flows') : status),
                "progress_percent": status == "Complete" ? 100 : Math.round((status / this.model.get('flows')) * 100),
                "is_proton" : this.model.get('chipInstrumentType') == "proton",
                "in_progress": status != "Complete"
            };
            var qc = this.model.get('qcThresholds'),
                key_counts = context.king_report && context.king_report.libmetrics && context.king_report.libmetrics.aveKeyCounts,
                bead_loading_threshold = qc["Bead Loading (%)"],
                key_threshold = qc["Key Signal (1-100)"],
                usable_sequence_threshold = qc["Usable Sequence (%)"];


            $(this.el).html(this.template.render(context));
            this.$('.bead-loading').strength(context.bead_loading, bead_loading_threshold, context.bead_loading, 'Loading');
            this.$('.bead-live').strength(context.bead_live, undefined, context.bead_live, 'Live ISPs');
            this.$('.bead-lib').strength(context.bead_lib, undefined, context.bead_lib, 'Library ISPs');
            this.$('.key-signal').strength(key_counts, key_threshold, key_counts, 'Key Signal', '');
            this.$('.usable-sequence').strength(context.usable_seq, usable_sequence_threshold, context.usable_seq, 'Usable Seq');
        },

        review_plan_: function(e) {
        	review_plan(e);
    	},

        destroy_view: function() {
            console.log("Destroying card run view");
            //COMPLETELY UNBIND THE VIEW
            this.undelegateEvents();
            $(this.el).removeData().unbind();
            //Remove view from DOM
            this.remove();
            Backbone.View.prototype.remove.call(this);
        }
    });

    TableRunView = Backbone.View.extend({
        tagName: 'tr',

        initialize: function () {
            _.bindAll(this, 'render');
        },

        template: Hogan.compile($("#monitor_experiment_table_template").html()),

        render: function () {
            // if (this.model.reports.length > 0) {
                // var king_report = this.model.reports.at(0);
                // console.log(king_report);
                // var met = king_report.get("analysis_metrics");
                // var qc = king_report.get("quality_metrics");
            // }
            // var status = this.model.get('ftpStatus');
            // context = {
                // exp: this.model.toJSON(),
                // "king_report": king_report && king_report.toJSON(),
                // "date_string": this.model.get('date').toString("yyyy/MM/dd hh:mm tt"),
                // "bead_loading": met && Math.round(met.bead / (met.total_wells - met.excluded) * 1000) / 10,
                // "bead_live": met && Math.round(met.live / met.bead * 1000) / 10,
                // "bead_lib": met && Math.round(met.lib / met.live * 1000) / 10,
                // "usable_seq": met && qc && Math.round(qc.q0_reads / met.lib * 1000) / 10,
                // "progress_flows": Math.round((status == "Complete" ? 1: status / 100.0) * this.model.get('flows')),
                // "progress_percent": status == "Complete" ? 100 : status,
                // "in_progress": !isNaN(parseInt(status))
            // };
            // console.log(context);
            // $(this.el).html(this.template.render(context));
        }
    });

	function review_plan(e) {
		e.preventDefault();
    	$('#error-messages').hide().empty();
		url = $(e.target).attr('href');

		$('body #modal_review_plan').remove();
		$.get(url, function(data) {
		  	$('body').append(data);
		    $( "#modal_review_plan" ).modal("show");
		    return false;
		}).done(function(data) {
	    	console.log("success:", url);
		})
	    .fail(function(data) {
	    	$('#error-messages').empty().show();
	    	$('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
	    	console.log("error:", data);

	    })
	    .always(function(data) { /*console.log("complete:", data);*/ });
	};

    RunListView = Backbone.View.extend({
        el: $("#monitor_view"),

        events: {
            'click #view_full': 'view_full',
            'click #view_table': 'view_table',
            'click #live_button': 'toggle_live_update'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'addRun', 'setup_full_view',
                'view_full', 'setup_table_view', 'view_table', 'start_update',
                'toggle_live_update', 'clear_update', 'poll_update',
                'countdown_update', 'appendRun');
            this.table_view = null;
            this.pager = new PaginatedView({collection: this.collection, el:$("#pager")});
            this.pager.render();
            this.collection.bind('add', this.addRun);
            this.collection.bind('reset', this.render);
            this.router = this.options.router;
            this.router.on("route:full_view", this.view_full);
            this.router.on("route:table_view", this.view_table);
            this.live_update = setTimeout(this.poll_update, 10000);
        },

        render: function () {
            console.log("Rendering RunListView");
            console.log(this.collection);
            $("#main_list").empty();
            this.collection.each(this.appendRun);
            return this;
        },

        addRun: function (run, collection, options) {
            console.log("Adding run");
            console.log(options);
            options = options || {index: 0};
            var tmpView = new this.RunView({model: run});
            tmpView.render();
            $(tmpView.el).hide();
            if (this.$('#main_list').children() == 0) {
                $("#main_list > div", this.el).append(tmpView.el);
            } else {
                $("#main_list > div", this.el).eq(options.index).before(tmpView.el);
            }
            $(tmpView.el).slideDown(500);
        },

         appendRun: function (run) {
            var tmpView = new this.RunView({model: run});
            tmpView.render();
            $("#main_list", this.el).append(tmpView.el);
        },

        setup_full_view: function () {
            $("#data_panel").html('<div id="main_list"></div>');
            this.RunView = CardRunView;
            $("#view_table").removeClass('active');
            $("#view_full").addClass('active');
            $('#pager').show();
        },

        view_full: function() {
            if(!(this.table_view === false)) {
                this.table_view = false;
                this.router.navigate("full");
                this.setup_full_view();
                this.render();
            }
        },

        clear_update: function () {
            if (this.live_update) {
                clearTimeout(this.live_update);
                this.live_update = null;
            }
        },

        start_update: function () {
            this.clear_update();
            this.live_update = true;
            this.poll_update();
        },

        countdown_update: function () {
            this.live_update = setTimeout(this.poll_update, 10000);
        },

        poll_update: function () {
            if (this.live_update) {
                this.collection.fetch({
                    update: true,
                    at: 0,
                    complete: this.countdown_update
                });
            }
        },

        toggle_live_update: function() {
            if (this.live_update !== null) {
                this.clear_update();
                this.$("#live_button").addClass('btn-success').text('Auto Update');
                this.$("#update_status").text('Page is static until refreshed');

            } else {
                this.start_update();
                this.$("#live_button").removeClass('btn-success').text('Stop Updates');
                this.$("#update_status").text('Page is updating automatically');
            }
        },

        setup_table_view: function () {
        	$("#data_panel").html('<div id="main_table"  class="table-dense"></div>');
            this.RunView = TableRunView;
            $("#view_full").removeClass('active');
            $("#view_table").addClass('active');
            $('#pager').hide();
			$("#main_table").kendoGrid({
				dataSource : {
					type : "json",
					transport : {
						read : {
							url : this.collection.baseUrl,
							contentType : 'application/json; charset=utf-8',
							type : 'GET',
							dataType : 'json'
						},
						parameterMap : function(options) {
							return buildParameterMap(options)
						}
					},
					schema : {
						data : "objects",
						total : "meta.total_count",
						model : {
							fields : {
								id : {
									type : "number"
								},
								pgmName : {
									type : "string"
								},
								expName : {
									type : "string"
								},
								library : {
									type : "string"
								},
								flows : {
									type : "number"
								},
								barcodeId : {
									type : "string"
								},
								chipDescription : {
									type : "string"
								},
								star : {
									type : "boolean"
								},
								date : {
									type : "string"
								},
								resultDate : {
									type : "string"
								},
								ftpStatus : {
									type : "string"
								},
								storageOptions : {
									type : "string"
								},
								notes : {
									type : "string"
								}
							}
						}
					},
					serverSorting : true,
					serverFiltering : true,
					serverPaging : true,
					pageSize : this.collection.limit,
					requestStart: function(e) {
						$('#main_table *[rel=tooltip]').tooltip('destroy');
						$('body div.tooltip').remove();
					}
				},
				height : 'auto',
				groupable : false,
				scrollable : false,
				selectable : false,
				sortable : true,
				pageable : true,
				columns : [{
					field : "star",
					title : " ",
					width : '3%',
					template : kendo.template($("#favoriteColumnTemplate").html())
				}, {
					field : "pgmName",
					width : '6%',
					title : "Instrument",
					template : '<span rel="tooltip" title="#= pgmName#">#=pgmName # </span>'
				}, {
					field : "displayName",
					width : '17%',
					title : "Run Name",
					template: kendo.template($("#expNameLinkTemplate").html())
				}, {
					field : "ftpStatus",
					title : "Status",
					template : kendo.template($("#statusColumnTemplate").html())
				}, {
					field : "date",
					title : "Started",
					template : '#= kendo.toString(new Date(Date._parse(date)),"MM/dd/yy hh:mm tt") #'
				}, {
					field : "resultDate",
					title : "Result Date",
					template : '#= kendo.toString(new Date(Date._parse(resultDate)),"MM/dd/yy hh:mm tt") #'
				}, {
					field : "chipDescription",
					width : '5%',
					title : "Chip"
				}, {
					field : "library",
					title : "Ref Genome",
					width : '6%',
					template : '#= TB.toString(library) #'
				}, {
					field : "barcodeId",
					title : "Barcode",
					width : '5%',
					template : '#= TB.toString(barcodeId) #'
				}, {
					title : "Loading",
					sortable : false,
					width : '5%',
					template : kendo.template($("#ispLoadingColumnTemplate").html())
				}, {
					title : "Live ISPs",
					sortable : false,
					width : '4%',
					template : kendo.template($("#ispLiveColumnTemplate").html())
				}, {
					title : "Library ISPs",
					sortable : false,
					width : '4%',
					template : kendo.template($("#ispLibraryColumnTemplate").html())
				}, {
					title : "Key Signal",
					sortable : false,
					width : '5%',
					template : kendo.template($("#keySignalColumnTemplate").html())
				}, {
					title : "Usable Seq",
					sortable : false,
					width : '5%',
					template : kendo.template($("#usableSequenceColumnTemplate").html())
				}],
				dataBound: function(e){
					function clickHandler(that) {
						function clickHandlerSuccess(_that, _attributes){
			            	_that.off();
			            	attributes = $.extend(_attributes, {id: _that.data('id')});
			            	var template = kendo.template($("#favoriteColumnTemplate").html());
			            	parentTD = _that.parent();
			            	parentTD.html(template({data:attributes}));
							$('.toggle-star', parentTD).click(function(e){e.preventDefault(); clickHandler($(this));});
		                }
						url = that.attr('href');
						attributes = {star: !that.data('star')};
						$.ajax({
			                url: url,
			                type: 'PATCH',
			                data: JSON.stringify(attributes),
			                contentType: 'application/json',
			                success: clickHandlerSuccess(that, attributes)
			            });
			            url = null;
			            attributes = null;
					};

					$('.toggle-star').click(function(e){e.preventDefault();clickHandler($(this));});
					$('.review-plan').click(review_plan);

					$('body div.tooltip').remove();
					initTooltip(this.content);
					// this.content.bind("DOMMouseScroll", hideTooltip(this.content)).bind("mousewheel", hideTooltip(this.content))
					// this.content.find('div.k-scrollbar').bind('scroll', hideTooltip(this.content))
				},
				requestStart: function(e) {
					$("#main_table").find('[rel="tooltip"]').tooltip('hide');
				}
			});
        },

        view_table: function () {
            if(!(this.table_view === true)) {
                this.table_view = true;
                this.router.navigate("table");
                this.setup_table_view();
                // this.render();
            }
        }
    });
});