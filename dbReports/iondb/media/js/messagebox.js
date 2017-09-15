//Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved
// requires jquery, underscore.js, and backbone.js
$(function() {
Message = Backbone.Model.extend({
    url: function() {
        return "/rundb/api/v1/message/" + this.get("id") + "/";
    },

    close: function(){
        this.destroy({wait: true});
    }
});

MessageList = Backbone.Collection.extend({
    model: Message,

    binding: "",

    url: function() {
        var route = "&route=";
        if (this.binding) {
            route = "&route__startswith=" + this.binding;
        }
        return "/rundb/api/v1/message/?status=unread&id__gt=" + this.max_id() + route;
    },

    parse: function(response) {
        return response.objects;
    },

    max_id: function(){
        return this.length ? this.at(0).get("id") : 0;
    },

    comparator: function(msg) {
        return  -1 * msg.get('id');
    },

    add: function(models) {
        // limit to one message per tag
        var dup = [];
        for (var i = 0; i < models.length; i++) {
            var tag = models[i].tags;
            if (tag){
                dup.push.apply(dup, this.where({'tags':tag}));
            }
        }
        if (dup.length>0) this.remove(dup);

        Backbone.Collection.prototype.add.call(this, models);
    }
});
UserMessageList = MessageList.extend({
    url: function() {
        var route = "&route=";
        if (this.binding) {
            route = "&route=" + this.binding;
        }
        return "/rundb/api/v1/message/?status=unread&id__gt=" + this.max_id() + route;
    }
});

MessageView = Backbone.View.extend({
    className: "alert",

    template: function(data) {
        return '<a class="close">x</a>' + data.body;
    },

    events: {
        'click a.close': 'close'
    },

    initialize: function() {
        this.render();
    },

    close: function() {
        var self = this;
        $(this.el).slideUp(function () {
            self.remove();
            $(window).trigger("ion.messageRemoved");
        });
        this.model.close();
    },

    render: function() {
        this.$el.html(this.template(this.model.toJSON()));
        var tag = this.model.get('tags');
        if (tag) this.$el.addClass(tag);
    }
});

MessageBox = Backbone.View.extend({
    container: "#global_messages",
    
    initialize: function() {
        _.bindAll(this, 'render', 'addOne', 'addAll', 'poll', 'update');
        this.setElement($(this.container));
        this.collection = new MessageList(this.options.models);
        this.collection.bind('add', this.addOne, this);
        this.collection.bind('reset', this.addAll, this);
        this.render();
        this.poll_interval = 10000;
        this.poll_clock = setTimeout(this.update, this.poll_interval);
    },

    render: function() {
        if (this.collection.length) {
            this.addAll();
            this.$el.show();
        }
    },

    addOne: function(message) {
        var view = new MessageView({model: message});
        view.$el.hide();
        // show single message per tag
        var tag = message.get('tags');
        if (tag) this.$el.find('.'+tag).remove();
        this.$el.prepend(view.el);
        view.$el.slideDown(function () {
            $(window).trigger("ion.messageAdded");
        });
    },

    addAll: function() {
        var views = this.collection.map(function(message){
            return new MessageView({model: message}).el;
        });
        this.$el.prepend(views);
        $(window).trigger("ion.messageAdded");
    },

    poll: function() {
        if (this.poll_clock != null) {
            this.poll_clock = setTimeout(this.update, this.poll_interval);
        }
    },

    update: function() {
        this.collection.fetch({add: true, success: this.poll, error: this.poll});
    }
});

UserMessageBox = MessageBox.extend({
    container: "#user_messages",
    initialize: function() {
        _.bindAll(this, 'render', 'addOne', 'addAll', 'poll', 'update');
        this.setElement($(this.container));
        this.collection = new UserMessageList(this.options.models);
        this.collection.binding = this.options.user;
        this.collection.bind('add', this.addOne, this);
        this.collection.bind('reset', this.addAll, this);
        this.render();
        this.poll_interval = 10000;
        this.poll_clock = setTimeout(this.update, this.poll_interval);
    }
});

});