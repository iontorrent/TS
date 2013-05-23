$(function(){
    console.log("Got initial data of " + initial_runs.objects.length + " items of " + initial_runs.meta.total_count);
    exps = new RunList(initial_runs.objects, {
        parse: true,
        limit: pageSize
    });
    exps.baseUrl = "/rundb/api/v1/monitorexperiment/";
    exps.total = initial_runs.meta.total_count;
    console.log("Created RunList");
    mainRouter = new RunRouter();
    mainRuns = new RunListView({
        collection: exps,
        router: mainRouter
    });
    if (!Backbone.history.start()) {
        console.log("Not routed");
        mainRuns.view_full();
    }
});
