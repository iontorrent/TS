/**
 * JS script for Ampliseq Panel Import.
 */

function _get_query_string(val){
    val = val || "";
    if ($.isArray(val)){
        return val.join(",")
    }
    return val;
}

function _filter(panel_type) {
    var filter = [{ field: "panelType", operator: "eq", value:panel_type}];
    if ($("#id_chemistryType .active").data('value') !== undefined ) {
        filter.push({
            field: "chemistryType",
            operator: "",
            value: $("#id_chemistryType .active").data('value')
        });
    }
    if ($("#id_pipeline .active").data('value') !== undefined ) {
        console.log($("#id_pipeline .active").data('value'));
        filter.push({
            field: "pipeline",
            operator: "",
            value: $("#id_pipeline .active").data('value')
        });
    }
    var rec_app = $("#recommend_app").val();
    if (rec_app !== undefined && rec_app != "All") {
        appLists = _get_query_string($("#recommend_app").val())
        filter.push({
            field: "recommended_application",
            operator: "",
            value: appLists
        });
    }
    var design_search_text = $("#search_design").val();
    if (design_search_text !== undefined) {
        filter.push({
            field: "name",
            operator: "contains",
            value: design_search_text
        });
    }

    return filter;
}

function filter(e){
    e.preventDefault();
    e.stopPropagation();
    $(".panel-categories").each(function() {
        var panel_type = $(this).data("panel-type");
        var grid = $(this).data("kendoGrid");
        if (grid != null)
        {
          grid.dataSource.filter(_filter(panel_type));
        }
        else{
            consol.log("Grid Data is null - No Panels to be listed");
        }
      });
}

function clear_filters(){
    $(".panel-categories").each(function() {
        $("#id_pipeline a").removeClass('active');
        $("#id_chemistryType a").removeClass('active');
        $("#search_design").val("");
        $('select[id=recommend_app]').val("All");
        $('.selectpicker').selectpicker('refresh')

        var panel_type = $(this).data("panel-type");
        var grid = $(this).data("kendoGrid");
        if (grid != null) {
          grid.dataSource.filter(_filter(panel_type));
        }
    });

}
