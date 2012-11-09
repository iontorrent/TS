var mainRuns = null;
var storage_choices = {
    "KI": "Keep",
    "A": "Archive",
    "D": "Delete"
}
var storage_classes = {
    "KI": "",
    "A": "btn-warning",
    "D": "btn-danger"
}

$(function () {
    var time_start = new Date();

    ReportView = Backbone.View.extend({
        tagName: 'tr',

        events: {
            'click .export': 'export_report',
            'click .prune': 'prune',
            'click .archive': 'archive',
            'click .delete': 'delete_report',
            'click .exempt': 'exempt',
            'click .icon-thumbs-up': 'toggle_representative'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'export_report', 'prune', 'archive', 'delete_report', 'exempt',
                'post_action', 'toggle_representative', 'destroy_view');

            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
        },

        template: Hogan.compile($("#report_template").html()),

        render: function () {
            console.log("Rendering report " + this.model.id);
            $(this.el).html(this.template.render({
                "report": this.model.toJSON(),
                "is_completed": function(){
                    return this.status == "Completed";
                },
                "total_q20bp": function(){
                    return this.quality_metrics && precisionUnits(this.quality_metrics.q20_bases);
                },
                "total_q0bp": function(){
                    return this.quality_metrics && precisionUnits(this.quality_metrics.q0_bases);
                },
                "reads_q20": function(){
                    return this.quality_metrics && precisionUnits(this.quality_metrics.q0_reads);
                },
                "read_length": function(){
                    return this.quality_metrics && Math.round(this.quality_metrics.q0_mean_read_length);
                },
                "date_string": kendo.toString(this.model.get("timeStamp"),"yyyy/MM/dd hh:mm tt")
            }));
        },

        destroy_view: function() {
            console.log("Destroying report view");
            //COMPLETELY UNBIND THE VIEW
            this.undelegateEvents();
            $(this.el).removeData().unbind(); 
            //Remove view from DOM
            this.remove();  
            Backbone.View.prototype.remove.call(this);
        },

        export_report: function(e) {
        	e.preventDefault();
        	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
            return this.post_action('E', 'Export');
        },

        prune: function(e) {
        	e.preventDefault();
        	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
            return this.post_action('P', 'Prune');
        },

        archive: function(e) {
        	e.preventDefault();
        	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
            return this.post_action('A', 'Archive');
        },

        delete_report: function(e) {
        	e.preventDefault();
        	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
            return this.post_action('D', 'Delete');
        },
        
        exempt: function(e) {
        	e.preventDefault();
        	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
        	return this.post_action('Z', 'Exempt');
        },

        post_action: function (setstr, message) {
        	console.log("Creating Dialog")
            var url = '/rundb/report/' + this.model.id + '/' + setstr;
            var currentPage = window.location.href;
            var refreshPage = function() {
            	window.location.href = currentPage;
            	window.location.reload(true);
            }
            $('#modal_data_management .modal-header h3').text("Report " 
            	+ this.model.get("resultsName") + " will now " 
            	+ message + ". Proceed?");
            $("#modal_data_management .btn-primary").click(function() {
            	var data = {};
            	data.comment = $.trim($("#data_management_comments").val()) || 'noComment';
            	
                $.post(url, data, refreshPage).fail(function() {
                    	alert('Error: could not complete task. Check the report log for more details.');
						refreshPage();
                });
            });
			
			$('#modal_data_management').modal('show');
            return false;
        },

        toggle_representative: function() {
            if (this.model.get("representative")) {
                this.model.patch({representative: false});
            } else {
                this.model.patch({representative: true});
            }
        }
    });

    ReportListView = Backbone.View.extend({
        events: {
            'click .reports-show-more': 'toggleMore'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'addReport', 'toggleMore', 'showMore', 
                'hideMore', 'destroy_view');
            this.is_open = false;
            this.collection.bind('add', this.addReport);
            this.collection.bind('reset', this.render);
            this.collection.bind('change', this.render);
            this.collection.bind('remove', this.destroy_view);
            this.render();
        },

        template: Hogan.compile($("#report_list_template").html()),

        render: function () {
            $(this.el).html(this.template.render({
                'count': this.collection.length,
                'more_reports': this.collection.length > 2,
                'is_open': this.is_open
            }));
            this.elBody = this.$('.reports-top');
            this.elMore = this.$('.reports-more');

            this.elBody.empty();
            this.collection.each(function(report, index){
                this.addReport(report, index);
            }, this);
            if (this.is_open) {
                this.showMore();
            }
        },

        addReport: function (report, index) {
            if (index === undefined) {
                index = this.collection.length;
            }
            var tmpReportView = new ReportView({
                model: report
            });
            tmpReportView.render();
            if (index < 2) {
                this.elBody.append(tmpReportView.el);
            } else {
                this.elMore.append(tmpReportView.el);
            }
        },

        hideMore: function () {
            this.elMore.hide();
            this.$('.reports-show-more').html('Show all ' + this.collection.length + ' reports');
        },
        
        showMore: function () {
            this.elMore.show();
            this.$('.reports-show-more').html('Hide');
        },

        toggleMore: function () {
            if (this.is_open) {
                console.log("Hiding");
                this.is_open = false;
                this.hideMore();

            } else {
                console.log("Showing");
                this.is_open = true;
                this.showMore();
            }
            return false;
        },

        destroy_view: function() {
            console.log("Destroying report list view");
            //COMPLETELY UNBIND THE VIEW
            this.undelegateEvents();
            $(this.el).removeData().unbind(); 
            //Remove view from DOM
            this.remove();  
            Backbone.View.prototype.remove.call(this);
        }
    });

    CardRunView = Backbone.View.extend({
        className: "run",

        events: {
            'click .reanalyze-run': 'reanalyze',
            'click .edit-run': 'edit',
            'click .completedrun-star': 'toggle_star',
            'click .storage-keep': function(e){
            	e.preventDefault();
            	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
            	this.set_storage('KI');
        	},
            'click .storage-archive': function(e){
            	e.preventDefault();
            	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
            	this.set_storage('A');
        	},
            'click .storage-delete': function(e){
            	e.preventDefault();
            	$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
            	this.set_storage('D');
        	}
        },

        initialize: function () {
            _.bindAll(this, 'render', 'reanalyze', 'edit', 'toggle_star', 
                'set_storage', 'destroy_view');
            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
            this.reports = new ReportListView({
                collection: this.model.reports
            });
        },

        template: Hogan.compile($("#experiment_template").html()),

        render: function () {
            console.log("Rendering " + this.model.id);
            this.$('[rel="tooltip"]').tooltip('hide');
            var status = this.model.get("ftpStatus");
            $(this.el).html(this.template.render({
                "exp": this.model.toJSON(),
                "prettyExpName": tb.prettyPrintRunName(this.model.get('expName')),
                "date_string": kendo.toString(this.model.get("date"),"yyyy/MM/dd hh:mm tt"),
                "king_report": this.model.reports.length > 0 ? this.model.reports.at(0).toJSON() : null,
                "progress_flows": Math.round((status == "Complete" ? 1: status / 100.0) * this.model.get('flows')),
                "progress_percent": status == "Complete" ? 100 : status,
                "in_progress": !isNaN(parseInt(status)),
                "storage_choice": storage_choices[this.model.get("storage_options")],
                "storage_class": storage_classes[this.model.get("storage_options")],
                "is_proton": this.model.get("chipType") == "900"
            }));
            this.reports.render();
            this.$('.table_container').html(this.reports.el);
        },

        reanalyze: function () {
            console.log("Reanalyze run " + this.model.id);
        },

        edit: function (e) {
            console.log("Edit run " + this.model.id);
            console.log("event " , e);
            e.preventDefault();
			url = $(e.currentTarget).attr('href');
			$('body #modal_experiment_edit').remove();
			$.get(url, function(data) {
			  	$('body').append(data);
			  	$( "#modal_experiment_edit" ).data('source', e.currentTarget);
			    $( "#modal_experiment_edit" ).modal("show");
			    return false;
			}).done(function(data) { 
		    	console.log("success:",  url);
			})
		    .fail(function(data) {
		    	$('#error-messages').empty();
		    	$('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>'); 
		    	console.log("error:", data);
		    	 
		    })
		    .always(function(data) { /*console.log("complete:", data);*/ });
			// return false;		                
        },

        toggle_star: function () {
            if (this.model.get("star")) {
                this.model.patch({star: false});
            } else {
                this.model.patch({star: true});
            }
        },

        set_storage: function(storage) {
            this.model.patch({"storage_options": storage});
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
    
	function edit_run(e) {
        console.log("event " , e);
        e.preventDefault();
		url = $(e.currentTarget).attr('href');
		$('body #modal_experiment_edit').remove();
		$.get(url, function(data) {
		  	$('body').append(data);
		  	$( "#modal_experiment_edit" ).data('source', e.currentTarget);
		    $( "#modal_experiment_edit" ).modal("show");
		    return false;
		}).done(function(data) { 
	    	console.log("success:",  url);
		})
	    .fail(function(data) {
	    	$('#error-messages').empty();
	    	$('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>'); 
	    	console.log("error:", data);
	    	 
	    })
	    .always(function(data) { /*console.log("complete:", data);*/ });
	    url = null;
		// return false;		                
   };
   
	RunView = Backbone.View.extend({
        edit: function(e) {
	        console.log("Edit run " + this.model.id);
        	edit_run(e);
        }
	});

    TableRunView = RunView.extend({
        tagName: 'tr',

        events: {
        },

        initialize: function () {
            _.bindAll(this);
            this.model.bind('change', this.render);
        },

        template: Hogan.compile($("#experiment_table_template").html()),

        render: function () {
        	
        },

        toggle_star: function () {
            // this.model.patch({star: !this.model.get("star")});
        }
    });

    RunListView = RunView.extend({
        el: $("#data_view"),

        events: {
            'change .search-field': 'search',
            'click #search_text_go': 'search',
            'click #clear_filters': 'clear_filters',
            'click #view_full': 'view_full',
            'click #view_table': 'view_table',
            'click #live_button': 'toggle_live_update',
            'click #download_csv': 'csv_download'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'addRun', 'search', 'setup_full_view', 
                'view_full', 'setup_table_view', 'view_table', 'start_update',
                'toggle_live_update', 'clear_update', 'poll_update', 
                'csv_download', 'countdown_update', 'appendRun');
            $(".chzn-select").chosen({no_results_text:"No results matched", "allow_single_deselect":true});
            $('#rangeA').daterangepicker({dateFormat: 'yy-mm-dd'});
            this.table_view = null;
            this.pager = new PaginatedView({collection: this.collection, el:$("#pager")});
            this.pager.render();
            this.collection.bind('add', this.addRun);
            this.collection.bind('reset', this.render);
            this.router = this.options.router;
            this.router.on("route:full_view", this.view_full);
            this.router.on("route:table_view", this.view_table);
            this.countdown_update();
        },

        render: function () {
            console.log("Rendering RunListView");
            console.log(this.collection);
            $("#main_list").empty();
            this.collection.each(this.appendRun);
            return this;
        },

        addRun: function (run, collection, options) {
            options = options || {index: 0};
            console.log(options);
            var tmpView = new this.RunView({model: run});
            tmpView.render();
            $("#main_list > div", this.el).eq(options.index).before(tmpView.el);
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
		
        setup_table_view: function () {
            $("#data_panel").html('<div id="main_table" class="table-dense"></div>');
            // $("#data_panel").html($("#experiment_list_table_template").html());
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
								}
							}
						}
					},
					serverSorting : true,
					serverFiltering : true,
					serverPaging : true,
					pageSize : this.collection.limit,
					filter: this._table_filter(),
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
					field : "expName",
					width : '10%',
					title : "Run Name",
					template: '<span rel="tooltip" title="#= expName#">#=tb.prettyPrintRunName(expName) # </span>'
				}, {
					field : "sample",
					width : '10%',
					title : "Sample",
					template : '<span rel="tooltip" title="#= tb.toString(sample)#">#= tb.toString(sample) #<span>'
				}, {
					title : "Application",
					sortable : false, 
            		template: '# if (plan) { # <span class="#=plan.runType#" rel="tooltip" title="#=tb.runTypeDescription(plan.runType)#"></span> # } #'
				}, {
					field : "date",
					title : "Run Date",
					template : '<span rel="tooltip" title="#= kendo.toString(new Date(Date._parse(date)),"yyyy/MM/dd hh:mm tt")#">#= kendo.toString(new Date(Date._parse(date)),"yyyy/MM/dd hh:mm tt") # </span>'
				}, {
					field : "resultDate",
					title : "Analysis Date",
					template : '<span rel="tooltip" title="#= kendo.toString(new Date(Date._parse(resultDate)),"yyyy/MM/dd hh:mm tt")#">#= kendo.toString(new Date(Date._parse(resultDate)),"yyyy/MM/dd hh:mm tt") # </span>'
				}, {
					field : "ftpStatus",
					title : "Status",
					template : kendo.template($("#statusColumnTemplate").html())
				}, {
					field : "chipDescription",
					width : '5%',
					title : "Chip"
				}, {
					field : "results",
					title : "Rep Report Name",
					sortable : false,
					width : '15%',
					template : kendo.template($("#reportNameColumnTemplate").html())
				}, {
					field : "library",
					title : "Ref Genome",
					width : '5%',
					template : '<span rel="tooltip" title="#= tb.toString(library)#">#= tb.toString(library) #</span>'
				}, {
					field : "barcodeId",
					title : "Barcode",
					template : '<span rel="tooltip" title="#= tb.toString(barcodeId)#">#= tb.toString(barcodeId) #</span>'
				}, {
					field : "flows",
					width : '3%',
					title : "Flows"
				}, {
					title : "Total Reads",
					sortable : false,
					width : '4%',
					template : kendo.template($("#totalReadsColumnTemplate").html())
				}, {
					title : "Mean Read Length",
					sortable : false,
					width : '4%',
					template : kendo.template($("#meanReadLengthColumnTemplate").html())
				}, {
					title : "Q20 Bases",
					sortable : false,
					width : '4%',
					template : kendo.template($("#q20BasesColumnTemplate").html())
				}, {
					title : "Output",
					sortable : false,
					width : '4%',
					template : kendo.template($("#outputColumnTemplate").html())
				}, {
					title : " ",
					sortable : false,
					width : '4%',
					template : kendo.template($("#actionColumnTemplate").html())
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
					$('.edit-run').click(edit_run);
					
					$('body div.tooltip').remove();
					initTooltip($(this.content));
					// this.content.bind("DOMMouseScroll", hideTooltip(this.content)).bind("mousewheel", hideTooltip(this.content))
					// this.content.find('div.k-scrollbar').bind('scroll', hideTooltip(this.content))
					
				}
			}); 

        },

        view_table: function () {
            if(!(this.table_view === true)) {
                this.table_view = true;
                this.router.navigate("table");
                this.setup_table_view();
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
            clearTimeout(this.live_update);
            this.live_update = setTimeout(this.poll_update, 10000);
        },

        poll_update: function () {
            if (this.live_update) {
            	if($('#main_table').exists())
            		refreshKendoGrid("#main_table");
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
                this.$("#live_button").removeClass('active').text('Live Update Off');
                
            } else {
                this.start_update();
                this.$("#live_button").addClass('active').text('Live Update On');
            }
        },
        
        clear_filters: function() {
			window.location.reload(true);
        },
		
		_get_query: function() {
            //Date requires extra formatting
            var params = {
                'all_date': $("#rangeA").val(),
                'all_text': $("#search_text").val(),
                'result_status': $("#id_status").val(),
                'star': $("#id_star:checked").exists(),
                'results__projects__name': $("#id_project").val(),
                'sample': $("#id_sample").val(),
                'chipType': $("#id_chip").val(),
                'pgmName': $("#id_pgm").val(),
                'library': $("#id_reference").val(),
                'flows': $("#id_flows").val(),
                'order_by': $("#order_by").val()
            };
            if (params['all_date']) {
                if (!/ - /.test(params['all_date'])) {
                    params['all_date'] = params['all_date'] + ' - ' + params['all_date'];
                }
                params['all_date'] = params['all_date'].replace(/ - /," 00:00,") + " 23:59";
            }
            if (params['order_by'] == '-resultDate') {
                params['order_by'] = '';
            }
            var query = {};
            for (var key in params) {
                if (params[key]) query[key] = params[key];
            }
            console.log(query);
			return query;
		},
		
		csv_download: function() {
			var q = this._get_query();
			q = $.extend({'format':'csv'}, q);
			console.log(q);
			var params = $.param(q);
			if (params.length > 0)
				params = '&' + params
			var url = '/data/getCSV.csv'; 
			jQuery.download(url, q, 'POST');
            return false;
		},
		
		_table_filter: function() {
    		var q = this._get_query();
    		var filter = [];
    		for (var key in q) {
    			if (key in ['order_by'])
    				continue;
				if (q[key]) {
    				filter.push({
    					field: key
						, operator: ""
						, value: q[key] 
    				});
				}
    		}
    		return filter;
			
		},
		
        search: function() {
        	if ($('#main_table').exists()) {
        		$('#main_table').data("kendoGrid").dataSource.filter(this._table_filter());
        	}
            this.collection.filtrate(this._get_query());
        }

    });

});
