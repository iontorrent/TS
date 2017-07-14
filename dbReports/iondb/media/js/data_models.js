var api_data_show_url = "/rundb/api/v1/compositeexperiment/show/";

function addCommas(nStr) {
    nStr += '';
    var x = nStr.split('.');
    var x1 = x[0];
    var x2 = x.length > 1 ? '.' + x[1] : '';
    var rgx = /(\d+)(\d{3})/;
    while (rgx.test(x1)) {
        x1 = x1.replace(rgx, '$1' + ',' + '$2');
    }
    return x1 + x2;
}

function precisionUnits(num, div) {
    if (typeof div === "undefined") div = 1000;
    num = parseFloat(num);
    var postfix = "";
    var exponent = Math.floor(Math.log(num) / Math.log(div));
    if (exponent >= 0) {
        num = Math.round(100 * num / Math.pow(div, exponent)) / 100;
    }
    if (Math.round(num) >= div) {
        num /= div;
        exponent += 1;
    }
    if (exponent >= 1) {
        postfix = "kMGTPEZY"[exponent - 1];
    }
    return num.toPrecision(3) + ' ' + postfix;
}

$(function(){
    // This proxies all usual API calling events to The Void.
    /*Backbone.sync = function(method, model, success, error){
        success();
    };*/

    // includes bindings for fetching/fetched

    PaginatedCollection = Backbone.Collection.extend({
        initialize: function(models, options) {
            _.bindAll(this, 'parse', 'url', 'pageInfo', 'nextPage', 'previousPage', 'filtrate', 'order_by', 
                'goToPage', 'lastPage', 'firstPage');
            typeof(options) != 'undefined' || (options = {});
            typeof(options.limit) != 'undefined' && (this.limit = options.limit);
            typeof(this.limit) != 'undefined' || (this.limit = 20);
            typeof(this.offset) != 'undefined' || (this.offset = 0);
            typeof(this.filter_options) != 'undefined' || (this.filter_options = {});
            typeof(this.sort_field) != 'undefined' || (this.sort_field = '');
        },

        fetch: function (options) {
            $(".mesh-warning").remove();
            typeof(options) != 'undefined' || (options = {});
            this.trigger("fetchStart");
            var self = this;
            var success = options.success;
            options.success = function (resp) {
                self.trigger("fetchDone fetchAlways");
                if (success) {
                    success(self, resp);
                }
            };
            var error = options.error;
            options.error = function (resp) {
                self.trigger("fetchFail fetchAlways");
                if (error) {
                    error(self, resp);
                }
            };
            return Backbone.Collection.prototype.fetch.call(this, options);
        },

        parse: function (resp) {
            this.offset = resp.meta.offset;
            this.limit = resp.meta.limit;
            this.total = resp.meta.total_count;

            //Side effect to show error message
            if (resp.warnings && resp.warnings.length > 0) {
                resp.warnings.map(function (warningText) {
                    $("<div/>", {
                        class: "alert mesh-warning",
                        html: "<strong>Warning!</strong> " + warningText
                    }).insertAfter("#search_bar");
                });
            }

            return resp.objects;
        },

        // Add a model, or list of models to the set. Pass **silent** to avoid
        // firing the `add` event for every new model.
        update: function (models, options) {
            var i, index, length, model, cid, id, cids = {}, ids = {}, dups = [];
            options || (options = {});
            models = _.isArray(models) ? models.slice() : [models];

            // Begin by turning bare objects into model references, and preventing
            // invalid models or duplicate models from being added.
            var new_length = 0;
            for (i = 0, length = models.length; i < length; i++) {
                if (!(model = models[i] = this._prepareModel(models[i], options))) {
                    throw new Error("Can't add an invalid model to a collection");
                }
                cid = model.cid;
                id = model.id;
                duplicate = cids[cid] || this._byCid[cid] || ((id != null) && (ids[id] || this._byId[id]));
                if (duplicate) {
                    dups.push(duplicate);
                } else {
                    dups.push(null);
                    new_length++;
                }
                cids[cid] = ids[id] = model;
            }

            // Listen to added models' events, and index models for lookup by
            // `id` and by `cid`.
            for (i = 0, length = models.length; i < length; i++) {
                if (!dups[i]) {
                    (model = models[i]).on('all', this._onModelEvent, this);
                    this._byCid[model.cid] = model;
                    if (model.id != null) this._byId[model.id] = model;
                }
            }

            // Removing missing models from the collection
            for (i = this.models.length - 1; i >= 0; i--) {
                model = this.models[i];
                id = model.id;
                if (!ids[id]) {
                    delete this._byId[model.id];
                    delete this._byCid[model.cid];
                    index = this.indexOf(model);
                    this.models.splice(index, 1);
                    this.length--;
                    if (!options.silent) {
                        options.index = index;
                        model.trigger('remove', model, this, options);
                    }
                    this._removeReference(model);
                }
            }

            // Insert models into the collection
            this.length += new_length;
            index = options.at != null ? options.at : this.models.length;
            for (i =0; i < length; i++) {
                model = models[i];
                if (dups[i] != null) {
                    dups[i].set(model, {silent: true});
                } else {
                    this.models.splice(index++, 0, model);
                }
            }

            //re-sorting if needed
            if (this.comparator) this.sort({silent: true});
            // Triggering `delete` and `add` events unless silenced.
            if (options.silent) return this;

            for (i = 0, length = this.models.length; i < length; i++) {
                model = this.models[i];
                options.index = i;
                if (dups[i]) {
                    model.change();
                } else {
                    model.trigger('add', model, this, options);
                }
            }
            return this;
        },

        url: function() {
            urlparams = {offset: this.offset, limit: this.limit};
            urlparams = $.extend(urlparams, this.filter_options);
            if (this.sort_field) {
                urlparams = $.extend(urlparams, {order_by: this.sort_field});
            }
            var full_url = this.baseUrl + '?' + $.param(urlparams);
            return full_url;
        },

        pageInfo: function() {
            var max = Math.min(this.total, this.offset + this.limit);
            
            var info = {
                total: this.total,
                offset: this.offset,
                limit: this.limit,
                page: Math.floor(this.offset / this.limit) + 1,
                pages: Math.ceil(this.total / this.limit),
                lower_range: this.offset + 1,
                upper_range: max,
                prev: false,
                next: false,
                is_first: this.offset == 0,
                is_last: this.offset + this.limit >= this.total
            };

            if (this.offset > 0) {
                info.prev = (this.offset - this.limit) || 1;
            }

            if (this.offset + this.limit < info.total) {
                info.next = this.offset + this.limit;
            }

            return info;
        },

        nextPage: function() {
            if (!this.pageInfo().next) {
                return false;
            }
            this.offset = this.offset + this.limit;
            return this.fetch();
        },

        previousPage: function() {
            if (!this.pageInfo().prev) {
              return false;
            }
            this.offset = (this.offset - this.limit) || 0;
            return this.fetch();
        },

        lastPage: function () {
            if (!this.pageInfo().next) {
              return false;
            }
            this.offset = this.total - this.limit;
            return this.fetch();
        },

        firstPage: function () {
            if (!this.pageInfo().prev) {
                return false;
            }
            this.offset = 0;
            return this.fetch();
        },

        goToPage: function (page) {
            var offset = (page - 1) * this.limit;
            if(offset >= this.total) {
                return false;
            }
            this.offset = offset;
            return this.fetch();
        },

        filtrate: function (options, isLocalDataSource) {
            this.filter_options = options || {};
            this.offset = 0;
            if (isLocalDataSource) {
                this.baseUrl = "/rundb/api/v1/compositeexperiment/";
            } else {
                this.baseUrl = "/rundb/api/mesh/v1/compositeexperiment/";
            }
            return this.fetch();
        },

        order_by: function (field) {
            this.sort_field = field;
            this.offset = 0;
            return this.fetch();
        }
    });
    
    PaginatedView = Backbone.View.extend({
        initialize: function() {
            _.bindAll(this, 'previous', 'next', 'render', 'first', 'last', 
                'destroy_view', 'pageRange', 'page');
            this.collection.bind('reset', this.render);
            this.collection.bind('add', this.render);
            this.collection.bind('remove', this.render);
        },

        events: {
            'click a.prev': 'previous',
            'click a.next': 'next',
            'click a.first': 'first',
            'click a.last': 'last',
            'click a.page': 'page'
        },

        template: Hogan.compile($("#pagination_template").html()),

        render: function () {
            var page_info = this.collection.pageInfo();
            page_info.page_numbers = this.pageRange(page_info);
            this.$el.html(this.template.render(page_info));
        },

        pageRange: function (info) {
            var page_numbers = {
                prev: null,
                pages: [],
                next: null
            };
            var lower_page = info.page - (info.page - 1) % 10;
            var upper_page = Math.min(info.pages, lower_page + 9);
            for(var i=lower_page; i <= upper_page; i++) {
                page_numbers.pages.push({
                    number: i,
                    is_current: i == info.page
                });
            }
            if(lower_page > 1) {
                page_numbers.prev = lower_page - 1;
            }
            if(upper_page < info.pages) {
                page_numbers.next = upper_page + 1;
            }
            return page_numbers;
        },

        previous: function () {
            this.collection.previousPage();
            return false;
        },

        next: function () {
            this.collection.nextPage();
            return false;
        },

        first: function () {
            this.collection.firstPage();
            return false;
        },

        last: function () {
            this.collection.lastPage();
            return false;
        },
        
        page: function (event) {
            var page = $(event.target).data("page");
            this.collection.goToPage(page);
            return false;
        },

        destroy_view: function() {
            //COMPLETELY UNBIND THE VIEW
            this.undelegateEvents();
            $(this.el).removeData().unbind(); 
            //Remove view from DOM
            this.remove();  
            Backbone.View.prototype.remove.call(this);
        },
    });

    TastyModel = Backbone.Model.extend({
        url: function () {
            return this.baseUrl + this.id + "/"
        },

        patch: function (attributes) {
            this.set(attributes);
            var url = this.url();
            if(this.get("_host")){
                url = "http://" + this.get("_host") + this.url()
            }
            $.ajax({
                url: url,
                type: 'PATCH',
                data: JSON.stringify(attributes),
                contentType: 'application/json'
            });
        }
    });

    Report = TastyModel.extend({
        parse: function (response) {
            var response = _.extend(response, {
                timeStamp: new Date(Date._parse(response.timeStamp))
            });
            if (response.experiment) {
                    //console.log("Experiment is not undefined " + response.experiment);
                    //console.log(this);
                response.experiment = _.extend(response.experiment, {
                    date: new Date(Date._parse(response.experiment.date)),
                    resultDate: new Date(Date._parse(response.experiment.resultDate))
                });
            }
            return response;
        },

        isCompleted: function () {
            return this.get("status") == "Complete";
        },

        baseUrl: "/rundb/api/v1/results/"
    });

    ReportList = PaginatedCollection.extend({
        model: Report,

        baseUrl: "/rundb/api/v1/compositeresult/"
    });

    Run = TastyModel.extend({
        url: function () {
            return this.baseUrl + this.get("id") + "/"
        },
        initialize: function () {
            if (this.get('results')) {
                    this.reports = new ReportList(this.get('results'), {
                    parse: true
                });
                var  run = this;
                this.bind('change', function(){
                    run.reports.reset(run.get("results"));
                });
            }
        },
        // When getting runs from the mesh, they can have the same id. We need a new id that is unique.
        idAttribute: 'uid',
        parse: function (response) {
            // For local runs the uid can just be the id.
            var uid = response.id;
            // For mesh runs the uid needs to be the combination of host and id.
            if(response._host){
                uid = response._host + ":" + response.id;
                console.log("uid", uid)
            }
            return _.extend(response, {
                uid: uid,
                date: new Date(Date._parse(response.date)),
                resultDate: new Date(Date._parse(response.resultDate))
            });
        },

        baseUrl: "/rundb/api/v1/compositeexperiment/"
    });

    RunList = PaginatedCollection.extend({
        model: Run,

        baseUrl: "/rundb/api/v1/compositeexperiment/"
    });

    RunRouter = Backbone.Router.extend({
        routes: {
            'full': 'full_view',
            'table': 'table_view'
        },

        initialize: function(options) {
            typeof(options) != 'undefined' || (options = {});
            options.server_state == "full" || (options.server_state = 'table');
            this.server_state = options.server_state;
        },

        full_view: function () {
            if (this.server_state != "full") {
                $.post(api_data_show_url, "full");
                this.server_state = "full";
            }
        },

        table_view: function () {
            if (this.server_state != "table") {
                $.post(api_data_show_url, "table");
                this.server_state = "table";
            }
        }
    });
    
});
