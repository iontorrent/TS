(function ($) {
    $.QueryString = (function (a) {
        if (a == "") return {};
        var b = {};
        for (var i = 0; i < a.length; ++i) {
            var p = a[i].split('=');
            if (p.length != 2) continue;
            b[p[0]] = decodeURIComponent(p[1].replace(/\+/g, " "));
        }
        return b;
    })(window.location.search.substr(1).split('&'))
})(jQuery);

$(function () {

    var modal = '<div id="absent" class="modal hide" tabindex="-1" role="dialog" aria-labelledby="myModalLabel" aria-hidden="true">\
<div class="modal-header">\
<button type="button" class="close" data-dismiss="modal" aria-hidden="true">×</button>\
<h3 id="myModalLabel">Manually Add Variant</h3>\
</div>\
<div class="modal-body">\
\
    <form id="addform" class="form-horizontal">\
        <div class="control-group">\
        <label class="control-label" for="mChrom">Chrom</label>\
            <div class="controls">\
            <input type="text" id="mChrom" placeholder="chr">\
            </div>\
        </div>\
        <div class="control-group">\
        <label class="control-label" for="mPos">Position</label>\
            <div class="controls">\
            <input type="text" id="mPos" placeholder="">\
            </div>\
        </div>\
        <div class="control-group">\
        <label class="control-label" for="mRef">Ref</label>\
            <div class="controls">\
            <input id="mRef" type="text">\
            </div>\
        </div>\
        <div class="control-group">\
        <label class="control-label" for="mVariant">Variant</label>\
            <div class="controls">\
            <input type="text" id="mVariant">\
            </div>\
        </div>\
        <div class="control-group">\
        <label class="control-label" for="mExpect">Expected Variant</label>\
            <div class="controls">\
            <input type="text" id="mExpect">\
            </div>\
        </div>\
    </form>\
\
</div>\
<div class="modal-footer">\
<button class="btn" data-dismiss="modal" aria-hidden="true">Close</button>\
<button class="btn btn-primary" id="addAbsent">Add absent variant</button>\
</div></div>';

    $(".main").append(modal);

    //new select boxes from http://silviomoreto.github.io/bootstrap-select/
    $('.selectpicker').selectpicker();


    function exportTools() {
        // could use data[] here directly

        $("#closebutton").click(function () {
            $('#dialog').modal('hide');
        });

        var numSelected = TVC.checked.length;
        var $content = $('#dialog-content');
        $content.html('Rows selected: ' + numSelected + '<br/>');
        if (numSelected == 0) {
            $content.append('<p>You must first select rows of the table data to export.</p>');
            $('#exportOK').hide();
        } else {
            $content.append('<div id="radio"><label class="radio">\
                          <input type="radio" name="modalradio" id="table" value="table" checked>\
                          Download table file of selected rows.\
                        </label>\
                        <label class="radio">\
                          <input type="radio" name="modalradio" id="ce" value="ce">\
                          Submit variants (human only) for PCR/Sanger sequencing primer design.\
                        </label>\
                        <label class="radio">\
                          <input type="radio" name="modalradio" id="taqman" value="taqman">\
                          Submit variants for TaqMan assay design.\
                        </label>\
                        </div>');
            if (TVC.reference == 'hg19'){
                $content.find('[name=modalradio]').each(function(){
                    if (this.value == "ce" || this.value == "taqman"){
                        $(this).prop('disabled', true);
                        $(this).closest('label').css('color', 'gray').css('cursor','not-allowed');
                        $(this).closest('label').attr('title', 'Reference Genome hg19 is no longer supported for sequencing primer or TaqMan assay design');
                    }
                });
            }
            $('#exportOK').show();
        }

        $('#dialog').modal('show');

    }

    $('#exportOK').click(function (e) {
        $('#dialog').modal('hide');
    var getkey = {};
        for (var i = 0; i < TVC.checked.length; ++i) {
        getkey[TVC.checked[i]] = TVC.checked_data[i]['key'];
        }
    var checkList = TVC.checked.slice();
    var keyrows = checkList.sort(function(a, b) {
        return getkey[a] - getkey[b];
    }) + ",";
        var rows = checkList.sort(function (a, b) {
            return a - b;
        }) + ",";
        var op = $("#radio input[type='radio']:checked").val();
        if (op == "table") {
            window.open("subtable.php3?dataFile=" + dataFile + "&keyrows=" + keyrows);
        } else if (op == "taqman") {
            window.open("taqman.php3?dataFile=" + dataFile + "&rows=" + rows);
        } else if (op == "ce") {
            window.open("sanger.php3?dataFile=" + dataFile + "&rows=" + rows);
        }
    });

    function suspectClick() {
        for (var i = 0; i < TVC.checked.length; ++i) {
            if (TVC.checked[i] in TVC.inspect) {
                //do nothing if it is already there
            } else {
                TVC.inspect[TVC.checked[i]] = TVC.checked_data[i];
            }
        }
        inspectRender();
    }

    function inspectRender() {
        $("#inspectBody").html("");
        for (key in TVC.inspect) {
            $("#inspectHead").show();
            var row = "";
            row = '<tr><td> ' + TVC.inspect[key]["position"] + '</td>';
            row += '<td>' + TVC.inspect[key]["ref"] + '</td>';
            row += '<td>' + TVC.inspect[key]["variant"] + '</td>';
            if ("expected" in TVC.inspect[key]) {
                row += '<td> <input type="text" class="inspectExpected" data-id="' + key + '" value="' + TVC.inspect[key]["expected"] + ' ">';
            } else {
                row += '<td> <input type="text" class="inspectExpected" data-id="' + key + '">';
            }
            row += '</td>';
            row += '<td><span class="btn btn-danger inspectDel" data-id="' + key + '"><i class="icon-remove"> </i> Remove Variant</span></td>';
            row += '</tr>';
            $("#inspectTable").append(row);
        }
    }

    $(document).on("click", "#suspect", suspectClick);

    $(document).on("click", ".inspectDel", function () {
        var id = $(this).data("id");
        delete TVC.inspect[id];
        inspectRender();
    });

    function poll_status() {
        setTimeout(function () {

            var poll = $.ajax({
                dataType: "json",
                async: false,
                cache: false,
                url: "split_status.json"
            });

            poll.always(function (data) {
                if ('url' in data) {
                    $("#inspectOutput").html('<a class="btn btn-primary" href="' + data['url'] + '"> <i class="icon-download"></i> Download the zip</a> ');
                } else {
                    if ('split_status' in data) {
                    	if (data["split_status"]=='failed'){
                    		var error_text = '<div class="alert alert-danger" role="alert"> <i class="icon-info-sign"></i> <span class="sr-only"> Error: </span>Job Failed. Please check the ';
                    		if ('temp_path' in data){
                    			error_text += '<a href="'+ data['temp_path'] + '/TVC_drmaa_stdout.txt">log file</a> for details.</div>';
                    		}
                    		else{
                    			error_text += 'log file for details.</div>';
                    		}
                    		$("#inspectOutput").html(error_text);
                    	}else{
                            $("#inspectOutput").html("<img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'/>" + data["split_status"]);
                            poll_status();
                    	}
                    }else{
                        $("#inspectOutput").html("<img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'/>" + data["split_status"]);
                        poll_status();
                    }
                }
            });
        }, 1000);
    }


    $(document).on("click", "#exportInspect", function () {

        var sp = get_json("startplugin.json");
        var post = {"startplugin": sp, "variants": TVC.inspect, "barcode": get_barcode()};

        $("#inspectOutput").html("<img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'/>Starting");

        var slice = $.ajax({
            dataType: "json",
            url: "/rundb/api/v1/plugin/" + TVC.plugin_name + "/extend/split/?format=json",
            type: "POST",
            async: false,
            cache: false,
            data: JSON.stringify(post)
        });

        slice.always(function (data) {

            if ('failed' in data) {
                $("#inspectOutput").html('<div class="alert alert-danger" role="alert"> <i class="icon-info-sign"></i> <span class="sr-only"> Error: </span> SGE Job Failed.</div>');
            } else {
                poll_status();
            }
        });

    });

    $(document).on("click", "#manualInspectAdd", function () {
        $('#absent').modal();
    });

    $(document).on("click", "#addAbsent", function () {
        var id = TVC.manual - 1;
        var position = $("#mChrom").val() + ":" + $("#mPos").val();
        var insert = {"chrom": $("#mChrom").val(), "pos": $("#mPos").val(), "ref": $("#mRef").val(),
            "variant": $("#mVariant").val(), "expected": $("#mExpect").val(), "position": position };
        console.log(insert);
        TVC.inspect[id] = insert;
        $('#absent').modal('hide');
        $("#addform").trigger('reset');
        inspectRender();
    });

    $(document).on("change", ".inspectExpected", function () {
        $(".inspectExpected").each(function () {
            var id = $(this).data("id");
            var expected = $(this).val();
            TVC.inspect[id]["expected"] = expected;
        });
    });

    function ChromIGV(row, cell, value, columnDef, dataContext) {
        if (value == null || value === "") {
            return "N/A"
        }
        var locpath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
        var igvURL = window.location.protocol + "//" + window.location.host + "/auth" + locpath + "/igv.php3";
        // link to Broad IGV
        //var href = "http://www.broadinstitute.org/igv/projects/current/igv.php?locus="+pos+"&sessionURL="+igvURL;
        // link to internal IGV
        var launchURL = window.location.protocol + "//" + window.location.host + "/IgvServlet/igv";
        var href = launchURL + "?locus=" + value + "&sessionURL=" + igvURL;
        return "<a href='" + href + "'>" + value + "</a>";
    }

    function RightAlignFormat(row, cell, value, columnDef, dataContext) {
        return '<div style="text-align:right">' + value + "</div>";
    }

    function PercentFormat(row, cell, value, columnDef, dataContext) {
        return '<div style="text-align:right">' + value.toFixed(1) + " %</div>";
    }
    
    function PercentFormatAF(row, cell, value, columnDef, dataContext) {
        return '<div style="text-align:right">' + (value).toFixed(2) + " %</div>";
    }

    function StrandBiasFormat(row, cell, value, columnDef, dataContext) {
        if (value == 0)
            return '<div style="text-align:right">-</div>';
        else
            return '<div style="text-align:right">' + value.toFixed(3) + "</div>";
    }

    function SSEFormat(row, cell, value, columnDef, dataContext) {
        if (dataContext["type"] == "DEL")
            return '<div style="text-align:right">' + value.toFixed(3) + "</div>";
        else
            return '<div style="text-align:right">-</div>';
    }

    function Fixed3Format(row, cell, value, columnDef, dataContext) {
        return '<div style="text-align:right">' + value.toFixed(3) + "</div>";
    }

    function Fixed1Format(row, cell, value, columnDef, dataContext) {
        return '<div style="text-align:right">' + value.toFixed(1) + "</div>";
    }

    function ThousandsIntFormat(row, cell, value, columnDef, dataContext) {
        var start_str = value.toFixed(0);
        var end_str = "";
        var len = start_str.length;
        while (len > 0) {
            if (end_str.length > 0)
                end_str = ',' + end_str;
            if (len <= 3) {
                end_str = start_str + end_str;
                start_str = "";
                len = 0;
            } else {
                end_str = start_str.substring(len - 3, len);
                start_str = start_str.substring(0, len - 3);
                len -= 3;
            }
        }
        return '<div style="text-align:right">' + end_str + " </div>";
    }

    function MarkFilter(cellNode, row, dataContext, columnDef) {
        var lookupstr = (columnDef["id"] + "_filter");
        var tooltip = dataContext[lookupstr];
        if (tooltip != "-") {
            $(cellNode).addClass('highlight');
            $(cellNode).addClass('tooltip-cell');
            var tokens = $.trim(tooltip).split(',');
            var parsed_tooltip = "";
            for (var i = 0; i < tokens.length; i++)
                parsed_tooltip += "\n - " + tokens[i];
            $(cellNode).attr('title', "Parameter thresholds not met:" + parsed_tooltip);
        }
    }

    function CheckBox(row, cell, value, columnDef, dataContext) {
        if ($.inArray(value, TVC.checked) >= 0 ){
            var tmpl ='<span><input class="checkBox" value="' + value +'" data-row="' + row + '" type="checkbox" checked></span>';
        }else{
            var tmpl ='<span><input class="checkBox" value="' + value +'" data-row="' + row + '" type="checkbox"></span>';
        }
        return tmpl;
    }

    function setCheckAll() {
        var check = true;
        for(var row=0; row<TVC.data.length; row++) {
            var id = TVC.data[row]['id'];
            if($.inArray(id, TVC.checked) < 0) {
                check = false;
                break;
           }
        }
        $('#checkall').attr('checked', check);
    }

    $(document).on("click", ".checkBox", function () {
        var check = $(this).prop("checked");
        var id = parseInt($(this).attr("value"));
        var row = parseInt($(this).data("row"));

        if (check){
            if ($.inArray(id,TVC.checked) < 0) {
                TVC.checked.push(id);
                TVC.checked_data.push(TVC.data[row]);
            }
        }else{
            for (var i = 0; i < TVC.checked.length; i++){
                if (TVC.checked[i] === id) {
                    TVC.checked.splice(i, 1);
                    TVC.checked_data.splice(i, 1);
                    break;
                }
            }
        }
        setCheckAll();
        TVC.pager_update();
    });

    $(document).on("click", "#checkall", function () {
        var check = $(this).prop("checked");
        if(check) {
            for(var row=0; row<TVC.data.length; row++) {
                var id = TVC.data[row]['id'];
                if($.inArray(id, TVC.checked) < 0) {
                    TVC.checked.push(id);
                    TVC.checked_data.push(TVC.data[row]);
                    TVC.grid.updateRow(row);
               }
            }
        } else {
            for(var row=0; row<TVC.data.length; row++) {
                var id = TVC.data[row]['id'];
                var i = $.inArray(id, TVC.checked);
                if(i < 0) continue;
                TVC.checked.splice(i, 1);
                TVC.checked_data.splice(i, 1);
                TVC.grid.updateRow(row);
            }
        }
        TVC.pager_update();
    });

    var dataFile = $("#allelesTable").attr("fileurl");

    function get_barcode() {
        //TODO: this needs to get data from a json file not the DOM. This can break easily.
        //test to see if this is a barcoded page
        var barcode = false;

        $(".headvc").each(function () {
            if ($(this).html() === "Barcode") {
                barcode = $.trim($(this).parent().parent().find("div:eq(1)").text());
            }
        });

        return barcode;
    }

    function get_reference() {
        var reference = false;
        $(".headvc").each(function () {
            if ($(this).html() === "Reference Genome") {
                reference = $.trim($(this).parent().parent().find("div:eq(1)").text());
            }
        });
        return reference;
    }

    function get_json(file_name) {
        //get the startplugin json

        var startplugin = {};

        var startplugin_url = file_name;

        //check to see if it is a barcode, if it is then load the json from the parent dir
        if (file_name === "startplugin.json") {
            var barcode = get_barcode();

            if (barcode) {
                startplugin_url = "../" + file_name;
            }
        }

        var startplugin = $.ajax({
            dataType: "json",
            async: false,
            url: startplugin_url
        });

        startplugin.done(function (data) {
            startplugin = data;
        });

        return startplugin;

    }
    
    // get the column names of the db
    function get_db_columns(TVC) {
	    var plugin_extend = "/rundb/api/v1/plugin/" + TVC.plugin_name;
	    plugin_extend += "/extend/db_columns/?format=json&path=" + TVC.plugin_dir;
	    if (TVC.barcode) {
	        plugin_extend += "/" + TVC.barcode + "/";
	    }
	    //request the data from the extend.py db_columns endpoint.
	    var db_columns = $.ajax({
	        url: plugin_extend,
	        async: false,
	        dataType: "json"
	    });
	    db_columns.done(function (data) {
	    	db_columns = data;
        });
	    return db_columns;
    }
    
    function is_tagseq_db(db_columns) {
    	var has_mol_cov = (db_columns.indexOf('Mol Coverage') >= 0);
    	var has_allele_mol_cov = (db_columns.indexOf('Allele Mol Cov') >= 0);
    	var has_allele_mol_freq = (db_columns.indexOf('Allele Mol Freq') >= 0);
    	return (has_mol_cov && has_allele_mol_cov && has_allele_mol_freq);
    }

    //add this to the window object so we can grab it everywhere
    window.TVC = {};

    //map the slickgrid col names to the names in the db
    TVC.col_lookup = {
        "chrom": ["Chrom", "String"],
        "pos": ["Position", "String"],
        "ref": ["Ref", "String"],
        "variant": ["Variant", "String"],
        "allele_call": ["Allele Call", "String"],
        "allele_call_filter": ["Allele Call Filter", "String"],
        "freq": ["Frequency", "Number"],
        
        // optional
        "ppa": ["Possible Polyploidy Allele", "Number"], 
        "subset":["Subset Of", "String"],

        "quality": ["Quality", "Number"],
        "quality_filter": ["Quality Filter", "String"],

        "type": ["Type", "String"],
        "allele_source": ["Allele Source", "String"],
    	"allele_id": ["Allele Name", "String"],
        "gene_id": ["Gene ID", "String"],
        "submitted_region": ["Region Name", "String"],
        "vcf_pos": ["VCF Position", "String"],
        "vcf_ref": ["VCF Ref", "String"],
        "vcf_alt": ["VCF Variant", "String"],

        // Tagseq only
        "read_cov": ["Read Cov", "Number"],
        "allele_read_cov": ["Allele Read Cov", "Number"],
        "allele_read_freq": ["Allele Read Freq", "Number"],
        "mol_cov": ["Mol Coverage", "Number"],
        "mol_cov_filter": ["Mol Coverage Filter", "String"],
        "allele_mol_cov": ["Allele Mol Cov", "Number"],
        "allele_mol_cov_filter": ["Allele Mol Cov Filter", "String"],
        "allele_mol_freq": ["Allele Mol Freq", "Number"],
        "allele_mol_freq_filter": ["Allele Mol Freq Filter", "String"],
        "lod": ["LOD", "Number"],
        
        // Non-Tagseq only
        "total_cov": ["Original Coverage", "Number"],
        "downsampled_cov_total": ["Coverage", "Number"],
        "downsampled_cov_total_filter": ["Coverage Filter", "String"],        
        "total_cov_plus": ["Coverage+", "Number"],
        "total_cov_plus_filter": ["Coverage+ Filter", "String"],
        "total_cov_minus": ["Coverage-", "Number"],
        "total_cov_minus_filter": ["Coverage- Filter", "String"],
        "allele_cov": ["Allele Cov", "Number"],
        "allele_cov_plus": ["Allele Cov+", "Number"],
        "allele_cov_minus": ["Allele Cov-", "Number"],

        "strand_bias": ["Strand Bias", "Number"],
        "strand_bias_filter": ["Strand Bias Filter", "String"],
        "rbi": ["Common Signal Shift", "Number"],
        "rbi_filter": ["Common Signal Shift Filter", "String"],
        "refb": ["Reference Signal Shift", "Number"],
        "refb_filter": ["Reference Signal Shift Filter", "String"],
        "varb": ["Variant Signal Shift", "Number"],
        "varb_filter": ["Variant Signal Shift Filter", "String"],
        "mlld": ["Relative Read Quality", "Number"],
        "mlld_filter": ["Relative Read Quality Filter", "String"],
        "hp_length": ["HP Length", "Number"],
        "hp_length_filter": ["HP Length Filter", "String"],
        "sse_plus": ["Context Error+", "Number"],
        "sse_plus_filter": ["Context Error+ Filter", "String"],
        "sse_minus": ["Context Error-", "Number"],
        "sse_minus_filter": ["Context Error- Filter", "String"],
        "sssb": ["Context Strand Bias", "Number"],
        "sssb_filter": ["Context Strand Bias Filter", "String"],

        "sample": ["Sample Name", "String"],
        "barcode": ["Barcode", "String"], 
        "run_name": ["Run Name", "String"], 
        "allele": ["Allele", "String"],
        "position": ["Location", "String"],
    };
    
    //Store the state of the grid position
    TVC.pos = 0;
    TVC.page_size = 20;
    TVC.order_col = false;
    TVC.order_dir = false;

    //TVC inspect
    TVC.inspect = {};
    TVC.manual = 0;

    TVC.checked = [];
    TVC.checked_data = [];

    var columns = [];

    //define the grid and attach head/foot of the table
    var options = {
        editable: true,
        autoEdit: false,
        enableCellNavigation: true,
        multiColumnSort: true,
        forceFitColumns: true,
        syncColumnCellResize: true,
        enableAsyncPostRender: true,
        asyncPostRenderDelay: 0
    };

    TVC.empty_grid = function () {
        TVC.data = [];
        TVC.dataView.beginUpdate();
        TVC.dataView.setItems({});
        TVC.dataView.endUpdate();
        TVC.grid.invalidateAllRows();
        TVC.grid.updateRowCount();
        TVC.grid.render();
    };

    TVC.subload = function (offset) {
        TVC.empty_grid();
        TVC.loadtable(offset);
        setCheckAll();
        TVC.grid.render();
        TVC.pager_update();
    };

    TVC.startplugin = get_json("startplugin.json");
    TVC.variant_summary = get_json("variant_summary.json");
    TVC.plugin_dir = TVC.startplugin["runinfo"]["results_dir"];
    TVC.plugin_name = TVC.startplugin["runinfo"]["plugin_name"];
    TVC.barcode = get_barcode();
    TVC.reference = get_reference();
    TVC.total_variants = TVC.variant_summary["variants_total"]["variants"];
    TVC.db_columns = get_db_columns(TVC);
    TVC.is_tagseq = is_tagseq_db(TVC.db_columns);

    TVC.column_2_name = [];
    for (var idx = 0; idx < TVC.db_columns.length; idx++){
    	TVC.column_2_name.push("");
    }
    for (var my_key in TVC.col_lookup){
    	var column_idx = TVC.db_columns.indexOf(TVC.col_lookup[my_key][0]);
    	// Bad design of the columns named "Filter" in alleles.xls.
    	// Now I have to figure out which Filter it is.
    	if (column_idx < 0 && my_key.endsWith("_filter")){
    		var try_key = my_key.slice(0, my_key.length - "_filter".length);
            if (try_key in TVC.col_lookup){
        	    column_idx = TVC.db_columns.indexOf(TVC.col_lookup[try_key][0]);
            }else{
            	column_idx = -1;
            }
    	    if (column_idx > -1 && column_idx < TVC.db_columns.length - 1){
    	    	if (TVC.db_columns[++column_idx] != "Filter"){
    	    		column_idx = -1;
    	    	}
    	    }else{
    	    	column_idx = -1;
    	    }
    	}
    	if (column_idx > -1 && column_idx < TVC.db_columns.length){
    		TVC.column_2_name[column_idx] = my_key;
    	}
    }
    
    //the columns shown in every view
    TVC.all = [
        {
            id: "id", field: "id", width: 5, minWidth: 5, sortable: false,
            name: "<input id='checkall' type='checkbox'>", toolTip: "Select the variant for export",
            formatter: CheckBox
        },
        {
            id: "position", field: "position", width: 25, minWidth: 25, sortable: true,
            name: "Position", toolTip: "Position: Allele position in reference genome. For hotspot alleles this is the original position before left alignment.",
            formatter: ChromIGV
        },
        {
            id: "ref", field: "ref", width: 18, minWidth: 12,
            name: "Ref", toolTip: "Ref: Reference sequence."
        },
        {
            id: "variant", field: "variant", width: 18, minWidth: 12,
            name: "Variant", toolTip: "Variant: Allele sequence that replaces reference sequence in the variant."
        },
    ];
    if (TVC.is_tagseq){
    	TVC.all.push({
            id: "allele", field: "allele", width: 32, minWidth: 24, sortable: true,
            name: "Allele", toolTip: "Allele: Gene + Allele Name."
        },
    	{
            id: "freq", field: "freq", width: 15, minWidth: 15, sortable: true,
            name: "Frequency", toolTip: "Frequency: Allele frequency, the ratio between (downsampled) allele molecular coverage and total molecular coverage",
            formatter: PercentFormatAF
    	},
    	{
            id: "lod", field: "lod", width: 12, minWidth: 12, sortable: true,
            name: "LOD", toolTip: "LOD: Limit of Detection at genomic location, estimated based on the number of detected molecules",
            formatter: PercentFormatAF
    	});
    }else{
    	TVC.all.push({
            id: "allele_call", field: "allele_call", width: 22, minWidth: 20, sortable: true,
            name: "Allele Call", toolTip: "Allele Call: Decision whether the allele is detected (Het and Hom), not detected (Absent), or filtered (No Call). No Call and Absent are for hotspot calls only.",
            asyncPostRender: MarkFilter
        },
    	{
            id: "freq", field: "freq", width: 15, minWidth: 15, sortable: true,
            name: "Frequency", toolTip: "Frequency: Allele frequency, the ratio between (downsampled) allele coverage and total coverage",
            formatter: PercentFormat
        });
    }

    if (TVC.db_columns.indexOf("Possible Polyploidy Allele") >= 0){
    	TVC.all.push({
            id: "ppa", field: "ppa", width: 8, minWidth: 8, sortable: true,
            name: "PPA", toolTip: "PPA: Indicate whether the allele is a Possible Polypoidy Allele (PPA) or not",
        });
    }
    
    TVC.all.push({
            id: "quality", field: "quality", width: 10, minWidth: 10, sortable: true,
            name: "Quality", toolTip: "Quality: PHRED-scaled probability of incorrect call",
            formatter: Fixed1Format, asyncPostRender: MarkFilter
        },
        {
            id: "spacer", name: "", width: 1, minWidth: 1, maxWidth: 2, cssClass: "separator-bar"
        });

    //the columns shown in allele search view
    TVC.allele = []
    var tagseq_width_adjustment = 0;
    if (TVC.is_tagseq){
        TVC.allele.push({
            id: "allele_call", field: "allele_call", width: 18, minWidth: 18, sortable: true,
            name: "Allele Call", toolTip: "Allele Call: Decision whether the allele is detected (Het and Hom), not detected (Absent), or filtered (No Call). No Call and Absent are for hotspot calls only.",
            asyncPostRender: MarkFilter
        });
    	tagseq_width_adjustment = -3;
    }
    
    if (TVC.db_columns.indexOf("Subset Of") >= 0){
    	TVC.allele.push({
            id: "subset", field: "subset", width: 20, minWidth: 16, sortable: true,
            name: "Subset Of", toolTip: "Subset Of: The name of the called allele that is a strict superset of this allele.",
        });
    }
    
    TVC.allele.push({
            id: "type", field: "type", width: 18 + tagseq_width_adjustment, minWidth: 18 + tagseq_width_adjustment, sortable: true,
            name: "Variant Type", toolTip: "Variant Type: SNP, INS, DEL, MNP, or COMPLEX"
        },
        {
            id: "allele_source", field: "allele_source", width: 18 + tagseq_width_adjustment, minWidth: 18 + tagseq_width_adjustment, sortable: true,
            name: "Allele Source", toolTip: 'Allele Source: Hotspot for alleles in hotspots file, otherwise Novel'
        },
        {
            id: "allele_id", field: "allele_id", width: 20 + tagseq_width_adjustment, minWidth: 20 + tagseq_width_adjustment, sortable: true,
            name: "Allele Name", toolTip: "Allele Name: Read from the hotspot file"
        },
        {
            id: "gene_id", field: "gene_id", width: 20 + tagseq_width_adjustment, minWidth: 20 + tagseq_width_adjustment, sortable: true,
            name: "Gene ID", toolTip: "Gene ID: Read from the target regions file"
        },
        {
            id: "submitted_region", field: "submitted_region", width: 32 + tagseq_width_adjustment, minWidth: 32 + tagseq_width_adjustment, sortable: true,
            name: "Region Name", toolTip: "Region Name: Read from target regions file"
        });
 
    //the columns shown in coverage view
    if (TVC.is_tagseq){
        TVC.coverage = [
	        {
	            id: "read_cov", field: "read_cov", width: 20, minWidth: 20, sortable: true,
	            name: "Total Read Cov", toolTip: "Coverage: Total read coverage at this position.",
	            formatter: ThousandsIntFormat
	        },
	        {
	            id: "allele_read_cov", field: "allele_read_cov", width: 20, minWidth: 20, sortable: true,
	            name: "Allele Read Cov", toolTip: "Number of reads containing alternative allele",
	            formatter: ThousandsIntFormat
	        },
	        {
	            id: "allele_read_freq", field: "allele_read_freq", width: 20, minWidth: 20, sortable: true,
	            name: "Allele Read Freq", toolTip: "Frequency of alternative allele across all reads",
	            formatter: PercentFormatAF,
	        },
	        {
	            id: "mol_cov", field: "mol_cov", width: 20, minWidth: 20, sortable: true,
	            name: "Total Mol Cov", toolTip: "Number of Molecules covering this location",
	            formatter: ThousandsIntFormat,  asyncPostRender: MarkFilter
	        },
	        {
	            id: "allele_mol_cov", field: "allele_mol_cov", width: 20, minWidth: 20, sortable: true,
	            name: "Allele Mol Cov", toolTip: "Allele Molecular Coverage: Number of detected molecules containing alternative allele",
	            formatter: ThousandsIntFormat,  asyncPostRender: MarkFilter
	        },
	        {
	            id: "allele_mol_freq", field: "allele_mol_freq", width: 20, minWidth: 20, sortable: true,
	            name: "Allele Mol Freq", toolTip: "Allele Molecular Frequency: Frequency of molecules containing alternative allele",
	            formatter: PercentFormatAF, asyncPostRender: MarkFilter
	        }
	    ];
    }
    else{
        TVC.coverage = [
	        {
	            id: "downsampled_cov_total", field: "downsampled_cov_total", width: 20, minWidth: 20, sortable: true,
	            name: "Coverage", toolTip: "Coverage: Total coverage at this position, after downsampling",
	            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
	        },
	        {
	            id: "total_cov_plus", field: "total_cov_plus", width: 20, minWidth: 20, sortable: true,
	            name: "Coverage +", toolTip: "Coverage+: Total coverage on forward strand, after downsampling ",
	            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
	        },
	        {
	            id: "total_cov_minus", field: "total_cov_minus", width: 20, minWidth: 20, sortable: true,
	            name: "Coverage -", toolTip: "Coverage-: Total coverage on reverse strand, after downsampling",
	            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
	        },
	        {
	            id: "allele_cov", field: "allele_cov", width: 20, minWidth: 20, sortable: true,
	            name: "Allele Cov", toolTip: "Allele Cov: Reads containing this allele, after downsampling",
	            formatter: ThousandsIntFormat
	        },
	        {
	            id: "allele_cov_plus", field: "allele_cov_plus", width: 20, minWidth: 20, sortable: true,
	            name: "Allele Cov +", toolTip: "Allele Cov+: Allele coverage on forward strand, after downsampling",
	            formatter: ThousandsIntFormat
	        },
	        {
	            id: "allele_cov_minus", field: "allele_cov_minus", width: 20, minWidth: 20, sortable: true,
	            name: "Allele Cov -", toolTip: "Allele Cov-: Allele coverage on reverse strand, after downsampling",
	            formatter: ThousandsIntFormat
	        },
	        {
	            id: "strand_bias", field: "strand_bias", width: 20, minWidth: 20, sortable: true,
	            name: "Strand Bias", toolTip: "Strand Bias: Imbalance between allele frequencies on forward and reverse strands",
	            formatter: StrandBiasFormat, asyncPostRender: MarkFilter
	        }
	    ];
    }

    //the columns shown in quality view
    TVC.quality = [
        {
            id: "rbi", field: "rbi", width: 20, minWidth: 20, sortable: true,
            name: "Common Signal Shift", toolTip: "Common Signal Shift: Distance between predicted and observed signal at the allele locus. [RBI]",
            formatter: Fixed3Format, asyncPostRender: MarkFilter
        },
        {
            id: "refb", field: "refb", width: 20, minWidth: 20, sortable: true,
            name: "Reference Signal Shift", toolTip: "Reference Signal Shift: Distance between predicted and observed signal in the reference allele. [REFB]",
            formatter: Fixed3Format
        },
        {
            id: "varb", field: "varb", width: 20, minWidth: 20, sortable: true,
            name: "Variant Signal Shift", toolTip: "Variant Signal Shift: Difference between predicted and observed signal in the variant allele [VARB]",
            formatter: Fixed3Format, asyncPostRender: MarkFilter
        },
        {
            id: "mlld", field: "mlld", width: 20, minWidth: 20, sortable: true,
            name: "Relative Read Quality", toolTip: "Relative Read Quality: Phred-scaled mean log-likelihood difference between prediction under reference and variant hypothesis [MLLD]",
            formatter: Fixed1Format, asyncPostRender: MarkFilter
        },
        {
            id: "hp_length", field: "hp_length", width: 20, minWidth: 20, sortable: true,
            name: "HP Length", toolTip: "Homopolymer length",
            formatter: RightAlignFormat, asyncPostRender: MarkFilter
        },
        {
            id: "sse_plus", field: "sse_plus", width: 20, minWidth: 20, sortable: true,
            name: "Context Error +", toolTip: "Context Error+: Probability of sequence-specific-error on forward strand (DELs only)",
            formatter: SSEFormat, asyncPostRender: MarkFilter
        },
        {
            id: "sse_minus", field: "sse_minus", width: 20, minWidth: 20, sortable: true,
            name: "Context Error -", toolTip: "Context Error-: Probability of sequence-specific-error on reverse strand (DELs only)",
            formatter: SSEFormat, asyncPostRender: MarkFilter
        },
        {
            id: "sssb", field: "sssb", width: 20, minWidth: 20, sortable: true,
            name: "Context Strand Bias", toolTip: "Context Strand Bias: Basespace strand bias (DELs only)",
            formatter: SSEFormat, asyncPostRender: MarkFilter
        }
    ];

    columns = columns.concat(TVC.all, TVC.allele);    
    
    TVC.dataView = new Slick.Data.DataView({inlineFilters: true});
    TVC.grid = new Slick.Grid("#AL-grid", TVC.dataView, columns, options);
    TVC.grid.setSelectionModel(new Slick.RowSelectionModel({selectActiveRow: false}));
    
    //TODO: do this in the Django template as well
    $.each(TVC.variant_summary["variants_by_chromosome"], function (i, variant) {
        $("#AL-selectChrom").append('<option value="' + variant["chromosome"] + '">' + variant["chromosome"] + '</select>');
    });

    $('.selectpicker').selectpicker('refresh');

    $("#AL-pager").append('<div class="slick-pager"></div>')
    $(".slick-pager").append('<span class="pull-left"><button class="btn" id="export">Export Selected</button></span>');
    $(".slick-pager").append('<span class="pull-left slick-pager-status"><span>Selected </span><span id="num_checked_x">0</span>'
         + '<span> of </span>'
         + '<span id="total_variants_x">' + TVC.total_variants + '</span></span>');


    var nav_buttons = '<span class="pull-right"><button class="btn" id="back"><i class="icon-arrow-left"> </i> Back</button>';
    nav_buttons    += '<button class="btn" id="next">Next <i class="icon-arrow-right"></i></button></span>';

    var page_html = '<span class="pull-right slick-pager-status"><span>Showing </span><span id="page_count">1</span>';
    page_html += '<span> of </span>';
    page_html += '<span id="total_variants">' + TVC.total_variants + '</span></span>';
    $(".slick-pager").append(nav_buttons);
    $(".slick-pager").append(page_html);

    $('<span class="btn" id="suspect" style="margin-left:10px;">Export for Troubleshooting</span> ').appendTo(".slick-pager");
    var table = '<div style="margin-top:35px; padding:5px; width: 99.5%" align="right">';
    table += '<button type="button" class="btn clearfix" id="adjust_debug_settings">';
    table += 'Show Troubleshooting <i class="icon-chevron-down"></i></button></div>';
    table += '<div class=clearfix collapse" id="toInspect" style="data-toggle=collapse; height:0px; padding: 5px; width: 99%;"><h3 style="margin-bottom:5px"><i class="icon-zoom-in"></i> Variants to inspect (mini bam/bed/vcf files will be generated)</h3>';
    table += '<div id="manualInspectAdd" class="btn">Add Manually</div>';
    table += '<div id="exportInspect" class="btn btn-primary" style="margin-left: 10px;">Export</div>';
    table += '<div id="inspectOutput" style="padding-top: 10px;"></div>';
    table += '<table class="table" id="inspectTable">';
    table += '<thead id="inspectHead" style="display: none;">';
    table += '<tr> <th>Position</th> <th>Reference</th> <th>Variant</th> <th>Expected Variant</th> <th>Remove</th></tr>';
    table += '</thead>';
    table += '<tbody id="inspectBody"></tbody></table></div></div>';
    $("#allelesTable").parent().append(table);

    $("#adjust_debug_settings").click(function () {
        $("#toInspect").collapse('toggle');
      });
    
    $('#toInspect').on('show', function () {
        document.getElementById('suspect').style.visibility = 'visible';
        $("#adjust_debug_settings").html('Hide Troubleshooting <i class="icon-chevron-up"></i>');    
    });

    $('#toInspect').on('hide', function () {
        document.getElementById('suspect').style.visibility = 'hidden';
        $("#adjust_debug_settings").html('Show Troubleshooting <i class="icon-chevron-down"></i>');     	
    });

    if ($.QueryString["debug"]) {
    	$("#toInspect").collapse('show');
    }else{
    	$("#toInspect").collapse('hide');
    }     
    
    $("#AL-tablecontent").appendTo('#allelesTable');
    $("#AL-tablecontent").show();

    TVC.pager_update = function() {
        $("#page_count").html(TVC.pos + 1);
        $("#page_count").append(" - ");
        $("#page_count").append(TVC.data.length + TVC.pos);
        $("#total_variants").html(TVC.total_variants);
        $("#num_checked_x").html(TVC.checked.length)
        $("#total_variants_x").html(TVC.total_variants);
    };

    TVC.pager_toggle = function() {
        if (TVC.data.length + TVC.pos >= TVC.total_variants) {
            $("#next").prop("disabled", true);
        } else {
            $("#next").prop("disabled", false);
        }

        if (TVC.pos == 0) {
            $("#back").prop("disabled", true);
        } else {
            $("#back").prop("disabled", false);
        }
        TVC.pager_update();
    };

    $("#next").click(function () {
        TVC.pos = TVC.pos + TVC.page_size;
        TVC.subload(TVC.pos);
        TVC.pager_toggle();
    });

    //start back disabled
    $("#back").prop("disabled", true);


    $("#back").click(function () {
        TVC.pos = TVC.pos - TVC.page_size;
        TVC.subload(TVC.pos);
        TVC.pager_toggle();
    });

    TVC.grid.onSort.subscribe(function (e, args) {
        TVC.checked = [];
        TVC.checked_data = [];
        TVC.grid.invalidateAllRows();
        TVC.order_col = TVC.col_lookup[args["sortCols"][0]["sortCol"]["field"]][0];
        if (TVC.order_col == "position") {TVC.order_col = "Location";}
        TVC.order_dir = args["sortCols"][0]["sortAsc"];
        TVC.subload(0);
        TVC.pos = 0;
        TVC.pager_toggle();
    });

    if(TVC.is_tagseq){
    	TVC.filterSettings = {"Allele Call":["Heterozygous","Homozygous"], "Allele Source": ["Hotspot"]};
        $("#AL-selectAlleleSource").val(["Hotspot"]);
        $("#AL-selectAlleleSource").change()
    }else{
        TVC.filterSettings = {"Allele Call":["Heterozygous","Homozygous"]};
    }
    $("#AL-selectAlleleCall").val(["Heterozygous","Homozygous"]);
    $("#AL-selectAlleleCall").change()

    //for building the position string
    TVC.Position_Start = false;
    TVC.Position_Stop = false;

    //for building the var freq
    TVC.VarFreq_Start = false;
    TVC.VarFreq_Stop = false;

    function myFilter(item, args) {
        //filter nothing
        return true;
    }

    $("#AL-selectChrom").change(function (e) {
        var selected;
        if ($(this).val()) {
            selected = $(this).val();
        } else {
            selected = "";
        }

        TVC.filterSettings[TVC.col_lookup["chrom"][0]] = selected;
        updateFilter();
    });

    $("#AL-selectAlleleSource").change(function (e) {
        var selected;
        if ($(this).val()) {
            selected = $(this).val();
        } else {
            selected = "";
        }
        TVC.filterSettings[TVC.col_lookup["allele_source"][0]] = selected;
        updateFilter();
        //console.log(TVC.filterSettings);
    });

    $("#AL-selectVarType").change(function (e) {
        var selected;
        if ($(this).val()) {
            selected = $(this).val();
        } else {
            selected = "";
        }
        TVC.filterSettings[TVC.col_lookup["type"][0]] = selected;
        updateFilter();
    });

    $("#AL-selectAlleleCall").change(function (e) {
        var selected;
        if ($(this).val()) {
            selected = $(this).val();
        } else {
            selected = "";
        }
        TVC.filterSettings[TVC.col_lookup["allele_call"][0]] = selected;
        updateFilter();
    });

    $("#AL-txtSearchPosStart").keyup(function (e) {
        this.value = this.value.replace(/\D/g, "");

        if (this.value == ""){
            TVC.Position_Start = false;
        }else{
            TVC.Position_Start = this.value;
        }
        updateFilter();
    });

    $("#AL-txtSearchPosEnd").keyup(function (e) {
        this.value = this.value.replace(/\D/g, "");
        if (this.value == ""){
            TVC.Position_Stop = false;
        }else{
            TVC.Position_Stop = this.value;
        }
        updateFilter();
    });

    $("#AL-txtSearchAlleleID").keyup(function (e) {
        TVC.filterSettings['Allele Name'] = this.value;
        updateFilter();
    });

    $("#AL-txtSearchGeneID").keyup(function (e) {
        TVC.filterSettings['Gene ID'] = this.value;
        updateFilter();
    });

    $("#AL-txtSearchRegionID").keyup(function (e) {
        TVC.filterSettings['Region Name'] = this.value;
        updateFilter();
    });

    $("#AL-txtSearchFreqMin").keyup(function (e) {

        this.value = forceStringFloat(this.value);

        if (this.value == ""){
            TVC.VarFreq_Start = false;
        }else{
            TVC.VarFreq_Start = this.value;
        }
        updateFilter();
    });

    $("#AL-txtSearchFreqMax").keyup(function (e) {
        this.value = forceStringFloat(this.value);

        if (this.value == ""){
            TVC.VarFreq_Stop = false;
        }else{
            TVC.VarFreq_Stop = this.value;
        }

        updateFilter();
    });

    $("#AL-txtSearchCovMin").keyup(function (e) {
        this.value = this.value.replace(/\D/g, "");

        if (this.value == ""){
            delete TVC.filterSettings["Coverage"];
        }else{
            TVC.filterSettings["Coverage"] = ">= " + this.value;
        }
        updateFilter();
    });

    function updateFilter() {
        //this will build the JSON that is used for the filtering from the API

        //the ranges have to be custom built
        if (TVC.Position_Start && TVC.Position_Stop){
            TVC.filterSettings["Position"] = "BETWEEN " + TVC.Position_Start + " AND " + TVC.Position_Stop;
        }else{
            delete TVC.filterSettings["Position"];
        }
        if (TVC.VarFreq_Start && TVC.VarFreq_Stop){
            TVC.filterSettings["Frequency"] = "BETWEEN " + TVC.VarFreq_Start + " AND " + TVC.VarFreq_Stop;
        }else{
            delete TVC.filterSettings["Frequency"];
        }

        //The JSON gets encoded as a string, then decoded back into JSON in the Python
        //Warning: this will break if the filter gets to be larger than around 2000 characters then it should
        //be send as the body of a POST request
        TVC.filter_query_string = encodeURIComponent(JSON.stringify(TVC.filterSettings));

        //remove checked boxes
        TVC.checked = [];
        TVC.checked_data = [];
        TVC.grid.invalidateAllRows();

        //go back to the first page
        TVC.pos = 0;

        //now do the page reload
        TVC.subload(0);

        //update the pager
        TVC.pager_toggle();

    }

    $("#AL-grid").css('height', '528px');
    TVC.grid.resizeCanvas();

    //resize on browser resize
    $(window).resize(function () {
        TVC.grid.resizeCanvas();
    });

    // initialize the model after all the events have been hooked up
    TVC.data = []; // defined by file load later
    TVC.dataView.beginUpdate();
    TVC.dataView.setItems(TVC.data);
    TVC.dataView.setFilter(myFilter);
    TVC.dataView.endUpdate();
    TVC.dataView.syncGridSelection(TVC.grid, true);


    TVC.loadtable = function (offset) {
        //This gets the data one page at a time from the server

        function onLoadPartial() {
            $("#AL-grid").css('height', '528px');
            TVC.dataView.setItems(TVC.data);
            TVC.grid.resizeCanvas();
            TVC.grid.render();
        }

        //build the URL, which as the filtering and sorting as part of the querystring
        var plugin_extend = "/rundb/api/v1/plugin/" + TVC.plugin_name;
        plugin_extend += "/extend/query/?format=json&path=" + TVC.plugin_dir;
        if (TVC.barcode) {
            plugin_extend += "/" + TVC.barcode + "/";
        }
        plugin_extend += "&limit=" + TVC.page_size;
        plugin_extend += "&offset=" + offset;

        if (TVC.order_col) {
            plugin_extend += "&column=" + encodeURIComponent(TVC.order_col);
            if (TVC.order_dir) {
                plugin_extend += "&direction=ASC";
            } else {
                plugin_extend += "&direction=DESC";
            }
        }

        if (TVC.filter_query_string){
            plugin_extend += "&where=" + TVC.filter_query_string;
        }

        //request the data from the extend.py query endpoint, get one page at a time.
        var get_page = $.ajax({
            url: plugin_extend,
            async: false,
            dataType: "json"
        });
        get_page.done(function (mem) {
            TVC.total_variants = mem["total"];
            $.each(mem["items"], function (n, fields) {
                var pk = fields.shift();
                var sortValue = fields.shift();
                TVC.data[n] = {"id": Number(pk),}
                //map the API data into the SlickGrid array
                //Just keep 1 page in memory at any time
                //Now I don't hard code the index of the allele table.
                for (var col_idx = 0; col_idx < TVC.column_2_name.length; col_idx++){
                	var my_key = TVC.column_2_name[col_idx];
                	if (my_key != ""){
                		if (TVC.col_lookup[my_key][1] == 'Number'){
                			TVC.data[n][my_key] = Number(fields[col_idx]);
                		}else{
                			TVC.data[n][my_key] = fields[col_idx];
                		}
                	}
                }
                
            	// TVC.data[n]['position'] should be TVC.data[n]['location'], but there are some massage applied in 'location'. 
                TVC.data[n]['position'] = (TVC.data[n]['chrom'] + ":" + TVC.data[n]['pos']);
            });
            onLoadPartial();
        });

        get_page.fail(function () {
            alert("failed to get data from the server");
        });

    };

    //table tabs
    $("#allele_search").click(function () {
        var columns = [];
        columns = columns.concat(TVC.all, TVC.allele);
        TVC.grid.setColumns(columns);
        $("#coltabs").find("li").removeClass("active");
        $(this).parent().addClass("active");
        setCheckAll();
    });

    $("#coverage").click(function () {
        var columns = [];
        columns = columns.concat(TVC.all, TVC.coverage);
        TVC.grid.setColumns(columns);
        $("#coltabs").find("li").removeClass("active");
        $(this).parent().addClass("active");
        setCheckAll();
    });

    $("#quality").click(function () {
        var columns = [];
        columns = columns.concat(TVC.all, TVC.quality);
        TVC.grid.setColumns(columns);
        $("#coltabs").find("li").removeClass("active");
        $(this).parent().addClass("active");
        setCheckAll();
    });

    $("#export").click(function () {
        exportTools();
    });

    //load it from the top
    updateFilter();
    //TVC.subload(0);

});
