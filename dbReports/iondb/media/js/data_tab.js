String.prototype.endsWith = function(suffix) {
    return this.indexOf(suffix, this.length - suffix.length) !== -1;
};

var mainRuns = null;

$(function () {
    var time_start = new Date();

    ReportView = Backbone.View.extend({
        tagName: 'tr',

        events: {
            'click .icon-thumbs-up': 'toggle_representative',
			'click .dm-actions': 'dm_actions'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'post_action', 'toggle_representative', 'destroy_view', 'dm_actions');

            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
        },

        template: Hogan.compile($("#report_template").html()),

        render: function () {
            $(this.el).html(this.template.render({
                "report": this.model.toJSON(),
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
                "date_string": kendo.toString(this.model.get("timeStamp"),"MM/dd/yy hh:mm tt"),
            }));
        },

        destroy_view: function() {
            //COMPLETELY UNBIND THE VIEW
            this.undelegateEvents();
            $(this.el).removeData().unbind();
            //Remove view from DOM
            this.remove();
            Backbone.View.prototype.remove.call(this);
        },

		dm_actions: function(e){
			e.preventDefault();
			$(e.currentTarget).closest('.dropdown-menu').parent().children('.dropdown-toggle').dropdown('toggle');
			$('body #modal_dm_actions').remove();
			var url = '/configure/services/dm_actions/' + this.model.id + '/';
			$.get(url, function(data) {
				$('body').append(data);
				$( "#modal_dm_actions" ).modal("show");
			});
			return false;
		},

        post_action: function (setstr, message, showModal) {
            var url = '/report/action/' + this.model.id + '/' + setstr;
            var currentPage = window.location.href;
            var refreshPage = function() {
                window.location.href = currentPage;
                window.location.reload(true);
            }
            var post_action_helper = function(url, data, refreshPage) {
                $.post(url, data, refreshPage).error(function() {
                    $('#modal_data_management_errors').empty().append('<p>Unable to complete task. Check the report log for more details.</p>').removeClass('hide');
                    setTimeout(function() {
                        refreshPage();
                    }, 2000);
                });
            }
            if (showModal) {
                $('#modal_data_management .modal-header h3').text("Report "
                    + this.model.get("resultsName") + " will now "
                    + message + ". Proceed?");
                $('#modal_data_management_errors').addClass('hide').empty();
                $("#modal_data_management .btn-primary").click(function(e) {
                    e.preventDefault();
                    $(e.target).addClass('disabled');
                    $(e.target).unbind('click');

                    var data = {};
                    data.comment = $.trim($("#data_management_comments").val()) || 'No Comment';
                    post_action_helper(url, data, refreshPage);
                });
                $('#modal_data_management').modal('show');
            } else {
                data = {}
                post_action_helper(url, data, refreshPage);
            }

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
                this.is_open = false;
                this.hideMore();

            } else {
                this.is_open = true;
                this.showMore();
            }
            return false;
        },

        destroy_view: function() {
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
			'click .storage-exempt': function(e){
				var checked = $(e.currentTarget).is(':checked');
                // This post has no error handling
				$.post('/configure/services/preserve_data/', "exppk="+ this.model.id+ "&keep="+ checked+ "&type=sig", function(data){
				});
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
            this.$('[rel="tooltip"]').tooltip('hide');
            var status = this.model.get("ftpStatus");
            $(this.el).html(this.template.render({
                "exp": this.model.toJSON(),
                "prettyExpName": TB.prettyPrintRunName(this.model.get('expName')),
                "date_string": kendo.toString(this.model.get("date"),"MM/dd/yy hh:mm tt"),
                "king_report": this.model.reports.length > 0 ? this.model.reports.at(0).toJSON() : null,
                "progress_flows": (status == "Complete" ? this.model.get('flows') : status),
                "progress_percent": status == "Complete" ? 100 : Math.round((status / this.model.get('flows')) * 100),
                "in_progress": !isNaN(parseInt(status)),
                "is_proton" : this.model.get('chipInstrumentType') == "proton"
            }));
            this.reports.render();
            this.$('.table_container').html(this.reports.el);
        },

        reanalyze: function () {
            //Reanalyze doesn't do anything at all....
        },

        edit: function (e) {
            e.preventDefault();
	    	$('#error-messages').hide().empty();
			url = $(e.currentTarget).attr('href');
			$('body #modal_experiment_edit').remove();
			$.get(url, function(data) {
			  	$('body').append(data);
			  	$( "#modal_experiment_edit" ).data('source', e.currentTarget);
			    $( "#modal_experiment_edit" ).modal("show");
			    return false;
			}).fail(function(data) {
		    	$('#error-messages').empty().show();
		    	$('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
		    	console.log("Experiment edit error:", data);

		    });
			return false;
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
            //COMPLETELY UNBIND THE VIEW
            this.undelegateEvents();
            $(this.el).removeData().unbind();
            //Remove view from DOM
            this.remove();
            Backbone.View.prototype.remove.call(this);
        }
    });

	function edit_run(e) {
        e.preventDefault();
    	$('#error-messages').hide().empty();
		url = $(e.currentTarget).attr('href');
		$('body #modal_experiment_edit').remove();
		$.get(url, function(data) {
		  	$('body').append(data);
		  	$( "#modal_experiment_edit" ).data('source', e.currentTarget);
		    $( "#modal_experiment_edit" ).modal("show");
		    return false;
		}).fail(function(data) {
	    	$('#error-messages').empty().show();
	    	$('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
	    	console.log("Experiment edit error:", data);
	    });
	    url = null;
   };

	RunView = Backbone.View.extend({
        edit: function(e) {
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

    DevRunView = Backbone.View.extend({
        tagName: 'tr',

        events: {
            'click .completedrun-star': 'toggle_star',
            'click .edit-run': 'edit'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'destroy_view', 'toggle_star', 'edit');
            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
        },

        template: Hogan.compile($("#experiment_table_template").html()),

        render: function () {
            var king_report = this.model.reports.length > 0 ? this.model.reports.at(0).toJSON() : null;
            var status = this.model.get("ftpStatus");
            this.$el.html(this.template.render({
                "exp": this.model.toJSON(),
                "run_date_string": this.model.get('date').toString("MM/dd/yy"),
                "result_date_string": this.model.get('resultDate').toString("MM/dd/yy"),
                "king_report": king_report,
                "progress_flows": (status == "Complete" ? this.model.get('flows') : status),
                "progress_percent": status == "Complete" ? 100 : Math.round((status / this.model.get('flows')) * 100),
                "in_progress": !isNaN(parseInt(status)),
                "total_q20bp": function(){
                    return king_report && king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q20_bases);
                },
                "total_q0bp": function(){
                    return king_report && king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q0_bases);
                },
                "reads_q20": function(){
                    return king_report && king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q0_reads);
                },
                "read_length": function(){
                    return king_report && king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q0_mean_read_length);
                }
            }));
        },

        edit: function (e) {
            e.preventDefault();
            $('#error-messages').hide().empty();
            url = '/data/experiment/' + this.model.id + '/';
            $('body #modal_experiment_edit').remove();
            $.get(url, function(data) {
                $('body').append(data);
                $( "#modal_experiment_edit" ).modal("show");
            })
            .fail(function(data) {
                $('#error-messages').empty().show();
                $('#error-messages').append('<p class="error">ERROR: ' + data.responseText + '</p>');
                console.log("error:", data);
            });
        },

        toggle_star: function () {
            if (this.model.get("star")) {
                this.model.patch({star: false});
            } else {
                this.model.patch({star: true});
            }
        },

        destroy_view: function() {
            //COMPLETELY UNBIND THE VIEW
            this.undelegateEvents();
            $(this.el).removeData().unbind();
            //Remove view from DOM
            this.remove();
            Backbone.View.prototype.remove.call(this);
        }
    });

    RunListView = RunView.extend({
        el: $("#data_view"),

        events: {
            'change .search-field': 'search',
            'click #search_text_go': 'search',
            'click #clear_filters': 'clear_filters',
            'click #live_button': 'toggle_live_update',
            'click #download_csv': 'csv_download',
            'click .sort_link': 'sort'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'addRun', 'search', 'setup_full_view',
                'view_full', 'setup_table_view', 'view_table', 'start_update',
                'toggle_live_update', 'clear_update', 'poll_update',
                'csv_download', 'countdown_update', 'appendRun');
            $(".chzn-select").chosen({no_results_text:"No results matched", "allow_single_deselect":true});
            $('.chzn-drop').css('width', $(".chzn-select").outerWidth()-2);  //Patched per https://github.com/harvesthq/chosen/issues/453#issuecomment-8884310
            $('.chzn-search input').css('width', $(".chzn-select").outerWidth()*.815);  //Patched per https://github.com/harvesthq/chosen/issues/453#issuecomment-8884310
            $('#rangeA').daterangepicker({dateFormat: 'mm/dd/yy'});
            this.current_view = null;
            this.collection.bind('add', this.addRun);
            this.collection.bind('reset', this.render);
            this.pager = null;
            this.router = this.options.router;
            this.router.on("route:full_view", this.view_full);
            this.router.on("route:table_view", this.view_table);
            this.live_update = null;
            //this.countdown_update();
        },

        render: function () {
            $("#main_list").empty();
            this.collection.each(this.appendRun);
            return this;
        },

        addRun: function (run, collection, options) {
            options = options || {index: 0};
            var tmpView = new this.RunView({model: run});
            tmpView.render();
            $("#main_list > div", this.el).eq(options.index).before(tmpView.el);
        },

        appendRun: function (run, index) {
            var tmpView = new this.RunView({model: run});
            tmpView.render();
            $("#main_list", this.el).append(tmpView.el);
        },

        setup_full_view: function () {
            if(this.pager !== null) {
                this.pager.destroy_view();
            }
            $("#data_panel").html('<div id="main_list"></div><div id="pager" class="k-pager-wrap" style="text-align: left;"></div>');
            this.RunView = CardRunView;
            $("#view_table").removeClass('active');
            $("#view_full").addClass('active');
            this.pager = new PaginatedView({collection: this.collection, el:$("#pager")});
            this.pager.render();
            $('#pager').show();
        },

        view_full: function() {
            if(!(this.current_view === 'full')) {
                this.current_view = 'full';
                this.setup_full_view();
                this.render();
            }
        },

        setup_table_view: function() {
            if(this.pager !== null) {
                this.pager.destroy_view();
            }
            var template = $("#experiment_list_table_template").html();
            $("#data_panel").html(template);
            this.RunView = DevRunView;
            $("#view_table").addClass('active');
            $("#view_full").removeClass('active');
            this.pager = new PaginatedView({collection: this.collection, el:$("#pager")});
            this.pager.render();
            $('#pager').show();
        },

        view_table: function () {
            if(!(this.current_view === 'table')) {
                this.current_view = 'table';
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
            this.live_update = setTimeout(this.poll_update, 20000);
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

        clear_filters: function() {
			window.location.reload(true);
        },

        sort: function (e) {
            var name = $(e.target).data('name');
            var current_sort = $("#order_by").val();
            if (current_sort == name) {
                $("#order_by").val('-' + name);
            } else if (current_sort == '-' + name) {
                $("#order_by").val("-resultDate");
            } else {
                $("#order_by").val(name);
            }
            $("#order_by").trigger('liszt:updated');
            this.search();
        },

		_get_query: function() {
            //Date requires extra formatting
            var params = {
                'all_date': $("#rangeA").val(),
                'all_text': $("#search_text").val(),
                'result_status': $("#id_status").val(),
                'star': $("#id_star:checked").exists(),
                'results__projects__name': $("#id_project").val(),
                'samples__name': $("#id_sample").val(),
                'chipType': $("#id_chip").val(),
                'pgmName': $("#id_pgm").val(),
                'results__eas__reference': $("#id_reference").val(),
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
			return query;
		},

		csv_download: function() {
			var q = this._get_query();
			q = $.extend({'format':'csv'}, q);
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
            this.collection.filtrate(this._get_query());
        }

    });

});
