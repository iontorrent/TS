String.prototype.endsWith = function(suffix) {
    return this.indexOf(suffix, this.length - suffix.length) !== -1;
};

function jqid (id) {
  return (!id) ? null : '#' + id.replace(/(:|\.|\[|\]|,|=|@)/g, '\\$1');
}

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

        runView: null,

        render: function () {
            console.log("ReportView render");
            $(this.el).html(this.template.render({
                "is_local": this.options.is_local, //Mesh
                "url_prefix": this.options.url_prefix, //Mesh
                "exp_name_prefix": this.options.exp_name_prefix, //Mesh
                "report": this.model.toJSON(),
                "total_q20bp": function(){
                	return (this.isShowAllMetrics) ? this.quality_metrics && precisionUnits(this.quality_metrics.q20_bases) : "---";
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
			var url = '/data/datamanagement/dm_actions/' + this.model.id + '/';
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
            console.log("Toggle ReportView", this.runView);
            this.runView.change_representative(this.model);
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

        runView: null,

        render: function () {
            console.log("ReportListView render");
            $(this.el).html(this.template.render({
                'count': this.collection.length,
                'more_reports': this.collection.length > 2,
                'is_open': this.is_open,
                "is_local": this.options.is_local, //Mesh
                "url_prefix": this.options.url_prefix, //Mesh
                "exp_name_prefix": this.options.exp_name_prefix //Mesh
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
                model: report,
                "is_local": this.options.is_local, //Mesh
                "url_prefix": this.options.url_prefix, //Mesh
                "exp_name_prefix": this.options.exp_name_prefix //Mesh
            });
            tmpReportView.runView = this.runView;
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
				$.post('/data/datamanagement/preserve_data/', "exppk="+ this.model.id+ "&keep="+ checked+ "&type=reanalysis", function(data){
				});
        	}
        },

        initialize: function () {
            _.bindAll(this, 'render', 'reanalyze', 'edit', 'toggle_star', 'destroy_view',
                'change_representative');
            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
            var is_local = !this.model.get('_host');
            var url_prefix = "";
            var exp_name_prefix = "";
            if (!is_local) {
                url_prefix = "http://" + this.model.get('_host');
                exp_name_prefix = "<strong>" + this.model.get('_host') + "</strong> "
            }
            this.reports = new ReportListView({
                collection: this.model.reports,
                is_local: is_local,
                url_prefix: url_prefix
            });
            this.reports.runView = this;
            console.log("Init CardRunView", this.reports.runView);
        },

        template: Hogan.compile($("#experiment_template").html()),

        render: function () {
            console.log("CardRunView render");
            this.$('[rel="tooltip"]').tooltip('hide');
            var status = this.model.get("ftpStatus");
            var is_local = !this.model.get('_host');
            var url_prefix = "";
            var exp_name_prefix = "";
            if (!is_local) {
                url_prefix = "http://" + this.model.get('_host');
                exp_name_prefix = "<strong>" + this.model.get('_host') + "</strong> "
            }
            $(this.el).html(this.template.render({
                "is_local": is_local, //Mesh
                "url_prefix": url_prefix, //Mesh
                "exp_name_prefix": exp_name_prefix, //Mesh
                "exp": this.model.toJSON(),
                "prettyExpName": this.model.get('displayName'),
                "date_string": kendo.toString(this.model.get("date"),"MM/dd/yy hh:mm tt"),
                "king_report": this.model.reports.length > 0 ? this.model.reports.at(0).toJSON() : null,
                "progress_flows": (status == "Complete" ? this.model.get('flows') : status),
                "progress_percent": status == "Complete" ? 100 : Math.round((status / this.model.get('flows')) * 100),
                "in_progress": !isNaN(parseInt(status)),
                "is_proton" : this.model.get('platform').toLowerCase() == "proton",
                "is_s5" : this.model.get('platform').toLowerCase() == "s5",
                "other_instrument" : this.model.get('platform').toLowerCase() == "pgm" || this.model.get('platform').toLowerCase() == "",
            }));
            this.reports.setElement(this.$('.table_container'));
            this.reports.render();
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
            if(this.model.get("_host")){
                alert("Runs can only be starred on thier originating Torrent Server.");
                return;
            }
            if (this.model.get("star")) {
                this.model.patch({star: false});
            } else {
                this.model.patch({star: true});
            }
        },

        change_representative: function(report) {
            if(this.model.get("_host")){
                alert("Reports can only be marked as representative on thier originating Torrent Server.");
                return;
            }
            console.log("Toggling experiment");
            var url = report.url() + 'representative/';
            var data;
            var others = this.reports.collection.where({representative: true});
            var state = report.get("representative");
            console.log(url, state, others, report);
            _(others).each(function(other){
                other.set("representative", false, {silent: true});
            });
            if (state) {
                report.set("representative", false, {silent: true});
                data = {representative: false};
            } else {
                report.set("representative", true, {silent: true});
                data = {representative: true};
            }
            this.trigger('change');
            data = JSON.stringify(data);
            var experiment = this.model;
            $.ajax({
                type: "POST",
                contentType: "application/json; charset=utf-8",
                url: url,
                data: data,
                dataType: 'json',
                success: function () {
                    console.log("Fetching!");
                    experiment.fetch();
                }
            });
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

        initialize: function (options) {
            this.options = options;
            _.bindAll(this, 'render', 'destroy_view', 'toggle_star', 'edit');
            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
        },

        template: Hogan.compile($("#experiment_table_template").html()),
        sample_template: kendo.template($('#SampleColumnTemplate').html()),

        render: function () {
            var king_report = this.model.reports.length > 0 ? this.model.reports.at(0).toJSON() : null;
            var status = this.model.get("ftpStatus");
            var is_local = !this.model.get('_host');
            var url_prefix = "";
            var exp_name_prefix = "";
            if (!is_local) {
                url_prefix = "http://" + this.model.get('_host');
                exp_name_prefix = "<strong>" + this.model.get('_host') + "</strong> "
            }
            var other_reports = this.model.reports.map(function (report) {
                return report.toJSON()
            }).filter(function (report) {
                if (king_report && king_report.id == report.id) {
                    return false;
                }
                return true;
            });
            display_host =
            this.$el.html(this.template.render({
                "exp": this.model.toJSON(),
                "is_local": is_local, //Mesh
                "url_prefix": url_prefix, //Mesh
                "exp_name_prefix": exp_name_prefix, //Mesh
                "run_date_string": this.model.get('date').toString("MMM d yyyy"),
                "result_date_string": this.model.get('resultDate').toString("MMM d yyyy"),
                "king_report": king_report,
                "progress_flows": (status == "Complete" ? this.model.get('flows') : status),
                "progress_percent": status == "Complete" ? 100 : Math.round((status / this.model.get('flows')) * 100),
                "in_progress": !isNaN(parseInt(status)),
                "total_q20bp": function () {
                    return (king_report && king_report.isShowAllMetrics) ? king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q20_bases) : "---";
                },
                "total_q0bp": function () {
                    return king_report && king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q0_bases);
                },
                "reads_q20": function () {
                    return king_report && king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q0_reads);
                },
                "read_length": function () {
                    return king_report && king_report.quality_metrics && precisionUnits(king_report.quality_metrics.q0_mean_read_length);
                },
                "other_reports": other_reports,
                "has_other_reports": other_reports.length > 0,
                more_columns: this.options.more_columns
            }));
            var samples = this.$el.children('.samples')
            samples.html(this.sample_template(this.model.toJSON()));
            samples.find('[rel=popover]').popover({content: samples.find(jqid('sample' + this.model.id)).html()});
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
            'keypress #search_text' : 'check_search_key',
            'click #clear_filters': 'clear_filters',
            'click #live_button': 'toggle_live_update',
            'click #download_csv': 'csv_download',
            'click .sort_link': 'sort',
            'change #id_data_source': 'data_source_update',
            'click #toggle_more_filters': 'toggle_more_filters',
            'click #toggle_more_columns': 'toggle_more_columns'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'addRun', 'search', 'setup_full_view',
                'view_full', 'setup_table_view', 'view_table', 'start_update',
                'toggle_live_update', 'clear_update', 'poll_update',
                'csv_download', 'countdown_update', 'appendRun', 'fetchStart',
                'fetchAlways');
            $(".chzn-select").chosen({no_results_text:"No results matched", "allow_single_deselect":true});
            $('.chzn-drop').css('width', $(".chzn-select").outerWidth()-2);  //Patched per https://github.com/harvesthq/chosen/issues/453#issuecomment-8884310
            $('.chzn-search input').css('width', $(".chzn-select").outerWidth()*.815);  //Patched per https://github.com/harvesthq/chosen/issues/453#issuecomment-8884310
            today = Date.parse('today');
            now = new Date();
            $('#rangeA').daterangepicker({
                dateFormat: 'M d yy',
                presetRanges: [
                    {text: 'Today', dateStart: today, dateEnd: today},
                    {text: 'Last 7 Days', dateStart: 'today-7days', dateEnd: today},
                    {text: 'Last 30 Days', dateStart: 'today-30days', dateEnd: today},
                    {text: 'Last 60 Days', dateStart: 'today-60days', dateEnd: today},
                    {text: 'Last 90 Days', dateStart: 'today-90days', dateEnd: today}
                ],
            });
            this.current_view = null;
            this.collection.bind('add', this.addRun);
            this.collection.bind('reset', this.render);
            this.collection.bind('fetchStart', this.fetchStart);
            this.collection.bind('fetchAlways', this.fetchAlways);
            this.pager = null;
            this.router = this.options.router;
            this.router.on("route:full_view", this.view_full);
            this.router.on("route:table_view", this.view_table);
            this.live_update = null;

            // Load the more_* attributes in html storage
            this.more_filters = false;
            if (localStorage.getItem('data-more-filters') == "true") {
                this.toggle_more_filters();
            }
            this.more_columns = false;
            if (localStorage.getItem('data-more-columns') == "true") {
                this.toggle_more_columns();
            }

            this.isLocalDataSource = true;
            this.updateSortIndicators();
            //this.countdown_update();
            this.meshValues = null;
            this.prefetchMeshValues();
            this.meshDropdowns = [
                {element: $("#id_project"), key: "projects"},
                {element: $("#id_sample"), key: "samples"},
                {element: $("#id_pgm"), key: "rigs"},
                {element: $("#id_reference"), key: "references"},
            ];

            // If there are form fields with values loaded from the browser cache, we need to re fetch the table data
            if (!$.isEmptyObject(this._get_query())) {
                console.log("Search fields are not empty on page load. Running search.");
                this.search();
            }

        },

        prefetchMeshValues: function () {
            $.ajax({
                url: "/rundb/api/mesh/v1/meshprefetch/",
                dataType: "json"
            }).done(function (response) {
                //Save values
                this.meshValues = response.values;
                //Side effect to show error message
                if (response.warnings && response.warnings.length > 0) {
                    response.warnings.map(function (warningText) {
                        console.log(warningText)
                    });
                }
                //Enable mesh dropdown if we have any mesh nodes
                var haveCompatibleNodes = false;
                Object.keys(response.nodes).map(function (host) {
                    var optionElement = $("<option/>", {value: response.nodes[host].id})
                    if (response.nodes[host].compatible) {
                        optionElement.text(host);
                        haveCompatibleNodes = true;
                    } else {
                        var warningText = "Incompatible";
                        if(response.nodes[host].warnings.length > 0){
                            warningText = response.nodes[host].warnings[0]
                        }
                        optionElement.text(host + " (" + warningText + ")");
                        optionElement.prop('disabled', true);
                    }
                    $("#id_data_source").append(optionElement);
                })
                if (haveCompatibleNodes) {
                    $("#id_data_source").attr("disabled", false).selectpicker('refresh');
                }
            }.bind(this));
        },

        addMeshValuesToDropdowns: function () {
            this.meshDropdowns.map(function (dropdown) {
                var existingValues = $.map(dropdown.element.children(), function (option) {
                    return option.value
                });
                this.meshValues[dropdown.key].map(function (value) {
                    if ($.inArray(value, existingValues) == -1) {
                        $("<option/>", {
                            text: value,
                            value: value,
                            "data-mesh": true
                        }).appendTo(dropdown.element);
                    }
                });
                dropdown.element.selectpicker("refresh");
            }.bind(this));
        },

        removeMeshValuesFromDropdowns: function () {
            this.meshDropdowns.map(function (dropdown) {
                dropdown.element.find("[data-mesh='true']").remove();
                dropdown.element.selectpicker("refresh");
            })
        },

        render: function () {
            $("#main_list").empty();
            this.collection.each(this.appendRun);
            return this;
        },

        addRun: function (run, collection, options) {
            options = options || {index: 0};
            var tmpView = new this.RunView({model: run, more_columns: this.more_columns});
            tmpView.render();
            $("#main_list > div", this.el).eq(options.index).before(tmpView.el);
        },

        appendRun: function (run, index) {
            var tmpView = new this.RunView({model: run, more_columns: this.more_columns});
            tmpView.render();
            $("#main_list", this.el).append(tmpView.el);
        },

        fetchStart: function () {
            console.log("Fetchnig started");
            var busyDiv = '<div id="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
            $('body').prepend(busyDiv);
        },

        fetchAlways: function () {
            console.log("Fetching finished", this.collection.length);
            $('#myBusyDiv').remove();
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
            // disable more columns link
            $("#toggle_more_columns_container").addClass("disabled");
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
            var template = Hogan.compile($("#experiment_list_table_template").html());
            $("#data_panel").html(template.render({more_columns: this.more_columns}));
            this.RunView = DevRunView;
            $("#view_table").addClass('active');
            $("#view_full").removeClass('active');
            this.pager = new PaginatedView({collection: this.collection, el:$("#pager")});
            this.pager.render();
            $('#pager').show();
            // enable more columns link
            $("#toggle_more_columns_container").removeClass("disabled");
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
                this.$("#live_button").addClass('btn-success').text('Auto Refresh');
                this.$("#update_status").text('Page is static until refreshed');

            } else {
                this.start_update();
                this.$("#live_button").removeClass('btn-success').text('Stop Refresh');
                this.$("#update_status").text('Page is updating automatically');
            }
        },

        toggle_more_filters: function() {
            var toggle_button = $("#toggle_more_filters");
            var optional_filters = $(".search-field.optional");
        	if (this.more_filters) {
                this.more_filters = false;
                toggle_button.text(toggle_button.data("more-text"));
                optional_filters.addClass("hide");
            } else {
                this.more_filters = true;
                toggle_button.text(toggle_button.data("less-text"));
                optional_filters.removeClass("hide");
            }
            localStorage.setItem('data-more-filters', this.more_filters);
        },

        toggle_more_columns: function() {
            var toggle_button = $("#toggle_more_columns");
        	if (this.more_columns) {
                this.more_columns = false;
                toggle_button.text(toggle_button.data("more-text"));
            } else {
                this.more_columns = true;
                toggle_button.text(toggle_button.data("less-text"));
            }
            this.setup_table_view();
        	this.render();
        	localStorage.setItem('data-more-columns', this.more_columns);
        },

        clear_filters: function() {
			window.location.reload(true);
        },

        sort: function (e) {
            var name = $(e.target).data('name');
            var current_sort = $("#order_by").val();
            if (current_sort == name) {
                $("#order_by").val('-' + name);
            } else {
                $("#order_by").val(name);
            }
            $("#order_by").selectpicker('refresh');
            this.search();
        },

        updateSortIndicators: function(){
            var current_sort = $("#order_by").val();
            var icon_class = "k-icon k-i-arrow-a";
            if (current_sort[0] == "-"){
                current_sort = current_sort.substring(1);
                icon_class = "k-icon k-i-arrow-s";
            }
            $(".order_indicator").html("");
            $("<span/>", {
                class: icon_class
            }).appendTo($("a[data-name='" + current_sort + "'] span.order_indicator"));
        },

        _array_to_query_string_value: function(array){
            if(array){
                return array.join(",")
            } else{
                return ""
            }
        },

        _get_query: function () {
            var params = {
                //Top Row
                'star': $("#id_star:checked").exists(),
                'all_text': $("#search_text").val(),
                'all_date': $("#rangeA").data("daterange"),
                'result_status': $("#id_status").val(),
                'results__projects__name__in': this._array_to_query_string_value($("#id_project").val()),

                //Sort
                'order_by': $("#order_by").val(),

                //Bottom Row
                'samples__name__in': this._array_to_query_string_value($("#id_sample").val()),
                'results__eas__reference__in': this._array_to_query_string_value($("#id_reference").val()),
                'flows__in': this._array_to_query_string_value($("#id_flows").val()),
                'chipType__in': this._array_to_query_string_value($("#id_chip").val()),
                'pgmName__in': this._array_to_query_string_value($("#id_pgm").val()),
                'sample_prep': $("#id_sample_prep").val()
            };

            //Mesh hosts
            if(!this.isLocalDataSource) {
                params["mesh_node_ids"] = this._array_to_query_string_value($("#id_data_source").val());
            }
            if (params['all_date']) {
                var start = params['all_date'].start;
                var end = params['all_date'].end.addHours(23).addMinutes(59);
                params['all_date'] = start.toString('MM/dd/yyyy HH:mm') + "," + end.toString('MM/dd/yyyy HH:mm');
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

        data_source_update: function() {
            var dataSourceValues = $("#id_data_source").val();
            // default to local data source
            this.isLocalDataSource = true;
            if(dataSourceValues) {
                // if the only selected option is "local" we can still enable local only features
                if (dataSourceValues.length == 1 && dataSourceValues[0] == "local") {
                    this.isLocalDataSource = true;
                } else {
                    this.isLocalDataSource = false;
                }
            }
            if (!this.isLocalDataSource) {
                this.addMeshValuesToDropdowns();
            }else{
                this.removeMeshValuesFromDropdowns();
            }
            if (!this.isLocalDataSource) {
                $("#download_csv").hide();
            }else{
                $("#download_csv").show();
            }
        },

        csv_download: function () {
            var q = this._get_query();
            var url = '/data/getCSV.csv?' + $.param(q);
            jQuery.download(url, 'POST');
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

        check_search_key: function (event) {
            if (event.which == 13 || event.keyCode == 13) this.search();
        },

        search: function() {
            this.updateSortIndicators();
            this.collection.filtrate(this._get_query(), this.isLocalDataSource);
        }

    });

});
