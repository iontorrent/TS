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
            //console.log("Rendering report");
            //console.log(this.model);
            var met = this.model.get("analysismetrics");
            var qc = this.model.get("qualitymetrics");
            var exp = this.model.get('experiment');
            var status = exp.ftpStatus;
            context = {
                exp: exp,
                "prettyExpName": exp.displayName,
                "king_report": this.model && this.model.toJSON(),
                "date_string": kendo.toString(exp.date, "MM/dd/yy hh:mm tt"),
                "bead_loading": met && Math.round(met.bead / (met.total_wells - met.excluded) * 1000) / 10,
                "bead_live": met && Math.round(met.live / met.bead * 1000) / 10,
                "bead_lib": met && Math.round(met.lib / met.live * 1000) / 10,
                "usable_seq": met && qc && Math.round(qc.q0_reads / met.lib * 1000) / 10,
                "progress_flows": (status == "Complete" ? exp.flows : status),
                "progress_percent": status == "Complete" ? 100 : Math.round((status / exp.flows) * 100),
                "is_proton" : exp.platform.toLowerCase() == "proton",
                "is_s5" : exp.platform.toLowerCase() == "s5",
                "other_instrument" : exp.platform.toLowerCase() == "pgm" || exp.platform=="",
                "in_progress": status != "Complete"
            };
            console.log(context);
            var qc = exp.qcThresholds,
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
            e.preventDefault();
        	$('body').css("cursor", "wait");
            $('#error-messages').hide().empty();
            var busyDiv = '<div id="myBusyDiv"><div class="k-loading-mask" style="width:100%;height:100%"><span class="k-loading-text">Loading...</span><div class="k-loading-image"><div class="k-loading-color"></div></div></div></div>';
            $('body').prepend(busyDiv);

            var url = this.$el.find('a.review-plan').attr('href');
            console.log(url);

            $('#modal_review_plan').remove();
            $.get(url).done(function(data) {
                $('body').append(data);
                $("#modal_review_plan").modal("show");
            }).fail(function(data) {
                $('#error-messages').empty().show().append('<p class="error">ERROR: ' + data.responseText + '</p>');
                console.log("AJAX Reivew Plan Error:", data);
            }).always(function(data) {
                $('body').css("cursor", "default");
                $('body #myBusyDiv').remove();
            });
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

        events: {
            'click .completedrun-star': 'toggle_star'
        },

        initialize: function () {
            _.bindAll(this, 'render', 'destroy_view', 'toggle_star');
            this.model.bind('change', this.render);
            this.model.bind('remove', this.destroy_view);
        },

        template: Hogan.compile($("#experiment_table_template").html()),

        render: function () {
            //console.log(this.model.changedAttributes());
            this.$('[rel="tooltip"]').tooltip('hide');
            //console.log("Rendering report");
            //console.log(this.model);
            var met = this.model.get("analysismetrics");
            var qc = this.model.get("qualitymetrics");
            var exp = this.model.get('experiment');
            var status = exp.ftpStatus;
            context = {
                exp: exp,
                "prettyExpName": exp.displayName,
                "king_report": this.model && this.model.toJSON(),
                "date_string": kendo.toString(this.model.get("timeStamp"), "MM/dd/yy hh:mm tt"),
                "bead_loading": met && Math.round(met.bead / (met.total_wells - met.excluded) * 1000) / 10,
                "bead_live": met && Math.round(met.live / met.bead * 1000) / 10,
                "bead_lib": met && Math.round(met.lib / met.live * 1000) / 10,
                "usable_seq": met && qc && Math.round(qc.q0_reads / met.lib * 1000) / 10,
                "progress_flows": (status == "Complete" ? exp.flows : status),
                "progress_percent": status == "Complete" ? 100 : Math.round((status / exp.flows) * 100),
                "is_proton" : exp.chipInstrumentType == "proton",
                "in_progress": status != "Complete"
            };
            $(this.el).html(this.template.render(context));
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

    RunListView = Backbone.View.extend({
        el: $("#monitor_view"),

        events: {
            'click #view_full': 'view_full',
            'click #view_table': 'view_table',
            'click #live_button': 'toggle_live_update',
            'click .sort_link': 'sort'
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
            this.live_update = null;
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
            this.live_update = setTimeout(this.poll_update, 120000);
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

        setup_table_view: function () {
            if(this.pager !== null) {
                this.pager.destroy_view();
            }
            var template = $("#experiment_list_table_template").html();
            $("#data_panel").html(template);
            console.log('pasted html');
            this.RunView = TableRunView;
            $("#view_table").addClass('active');
            $("#view_full").removeClass('active');
            this.pager = new PaginatedView({collection: this.collection, el:$("#pager")});
            this.pager.render();
            $('#pager').show();
        },

        view_table: function () {
            if(!(this.table_view === true)) {
                this.table_view = true;
                this.router.navigate("table");
                this.setup_table_view();
                this.render();
            }
        },

        sort: function (e) {
            e.preventDefault();
            $('.order_indicator').removeClass('k-icon k-i-arrow-n k-i-arrow-s')
            var $order_span = $(e.currentTarget).children('.order_indicator');
            var name = $(e.target).data('name');
            if (this.sort_key == name) {
                this.sort_key = '-' + name;
                $order_span.addClass('k-icon k-i-arrow-s')
            } else if ((this.sort_key == '-' + name) && (name != 'timeStamp')) {
                this.sort_key = '-timeStamp';
                $('.order_indicator_default').addClass('k-icon k-i-arrow-s')
            } else {
                this.sort_key = name;
                $order_span.addClass('k-icon k-i-arrow-n')
            }
            this.collection.filtrate({'order_by': this.sort_key});
        }
    });
});
