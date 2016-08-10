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
                    console.log(data);
                    if ('split_status' in data) {
                        $("#inspectOutput").html("<img style='height:30px;width:30px;' src='/site_media/resources/bootstrap/img/loading.gif'/>" + data["split_status"]);
                    }
                    poll_status();
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
                $("#inspectOutput").html('SGE Job Failed!');
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

    //add this to the window object so we can grab it everywhere
    window.TVC = {};

    //map the slickgrid col names to the names in the db
    TVC.col_lookup = {  "allele_id": "Allele Name",
        "downsampled_cov_total_filter": "Coverage Filter",
        "allele_cov_plus": "Allele Cov+",
        "sse_minus": "Context Error-",
        "pos": "Position",
        "mlld_filter": "Relative Read Quality Filter",
        "varb": "Variant Signal Shift",
        "quality": "Quality",
        "LOD": "LOD",
        "total_cov_minus": "Coverage-",
        "vcf_ref": "VCF Ref",
        "total_cov_minus_filter": "Coverage- Filter",
        "refb_filter": "Reference Signal Shift Filter",
        "type": "Type",
        "hp_length": "HP Length",
        "sssb": "Context Strand Bias",
        "strand_bias_filter": "Strand Bias Filter",
        "sssb_filter": "Context Strand Bias Filter",
        "allele_call": "Allele Call",
        "hp_length_filter": "HP Length Filter",
        "chrom": "Chrom",
        "submitted_region": "Region Name",
        "downsampled_cov_total": "Coverage",
        "strand_bias": "Strand Bias",
        "ref": "Ref",
        "quality_filter": "Quality Filter",
        "rbi": "Common Signal Shift",
        "total_cov": "Original Coverage",
        "allele_call_filter": "Allele Call Filter",
        "sse_minus_filter": "Context Error- Filter",
        "vcf_alt": "VCF Variant",
        "sse_plus_filter": "Context Error+ Filter",
        "allele_source": "Allele Source",
        "vcf_pos": "VCF Position",
        "refb": "Reference Signal Shift",
        "allele_cov": "Allele Cov",
        "total_cov_plus_filter": "Coverage+ Filter",
        "total_cov_plus": "Coverage+",
        "rbi_filter": "Common Signal Shift Filter",
        "varb_filter": "Variant Signal Shift Filter",
        "sse_plus": "Context Error+",
        "variant": "Variant",
        "gene_id": "Gene ID",
        "freq": "Frequency",
        "mlld": "Relative Read Quality",
        "allele_cov_minus": "Allele Cov-",
        "position": "position",
        "allele": "Allele"
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
            id: "ref", field: "ref", width: 10, minWidth: 10,
            name: "Ref", toolTip: "Ref: Reference sequence."
        },
        {
            id: "variant", field: "variant", width: 10, minWidth: 10,
            name: "Variant", toolTip: "Variant: Allele sequence that replaces reference sequence in the variant."
        },
        {
            id: "allele", field: "allele", width: 14, minWidth: 14, sortable: true,
            name: "Allele", toolTip: "Allele: Gene + Allele Name."
        },
        {
            id: "freq", field: "freq", width: 12, minWidth: 12, sortable: true,
            name: "Frequency", toolTip: "Frequency: Allele frequency, the ratio between (downsampled) allele coverage and total coverage",
            formatter: PercentFormatAF
        },
        {
            id: "quality", field: "quality", width: 12, minWidth: 12, sortable: true,
            name: "Quality", toolTip: "Quality: PHRED-scaled probability of incorrect call",
            formatter: Fixed1Format, asyncPostRender: MarkFilter
        },
        {
            id: "LOD", field: "LOD", width: 12, minWidth: 12, sortable: true,
            name: "LOD", toolTip: "LOD: LOD score compares the likelihood of obtaining the test data if the two loci are indeed linked, to the likelihood of observing the same data purely by chance",
            formatter: PercentFormatAF
        },
        {
            id: "spacer", name: "", width: 1, minWidth: 1, maxWidth: 10, cssClass: "separator-bar"
        }
    ];

    //the columns shown in allele search view
    TVC.allele = [
        {
            id: "allele_call", field: "allele_call", width: 14, minWidth: 14, sortable: true,
            name: "Allele Call", toolTip: "Allele Call: Decision whether the allele is detected (Het and Hom), not detected (Absent), or filtered (No Call). No Call and Absent are for hotspot calls only.",
            asyncPostRender: MarkFilter
        },
        {
            id: "type", field: "type", width: 30, minWidth: 30, sortable: true,
            name: "Variant Type", toolTip: "Variant Type: SNP, INS, DEL, MNP, or COMPLEX"
        },
        {
            id: "allele_source", field: "allele_source", width: 29, minWidth: 29, sortable: true,
            name: "Allele Source", toolTip: 'Allele Source: Hotspot for alleles in hotspots file, otherwise Novel'
        },
        {
            id: "allele_id", field: "allele_id", width: 29, minWidth: 29, sortable: true,
            name: "Allele Name", toolTip: "Allele Name: Read from the hotspot file"
        },
        {
            id: "gene_id", field: "gene_id", width: 29, minWidth: 29, sortable: true,
            name: "Gene ID", toolTip: "Gene ID: Read from the target regions file"
        },
        {
            id: "submitted_region", field: "submitted_region", width: 29, minWidth: 29, sortable: true,
            name: "Region Name", toolTip: "Region Name: Read from target regions file"
        }
    ];

    //the columns shown in coverage view
    TVC.coverage = [
        {
            id: "downsampled_cov_total", field: "downsampled_cov_total", width: 20, minWidth: 20, sortable: true,
            name: "Coverage", toolTip: "Coverage: Total coverage at this position, after downsampling",
            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
        },
        {
            id: "total_cov_plus", field: "total_cov_plus", width: 20, minWidth: 20, sortable: true,
            name: "Allele Read Coverage", toolTip: "Number of reads containing alternative allele",
            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
        },
        {
            id: "total_cov_minus", field: "total_cov_minus", width: 20, minWidth: 20, sortable: true,
            name: "Allele Read Frequency", toolTip: "Frequency of alternative allele across all reads",
            formatter: PercentFormatAF, asyncPostRender: MarkFilter
        },
        {
            id: "allele_cov", field: "allele_cov", width: 20, minWidth: 20, sortable: true,
            name: "Molecular Coverage", toolTip: "Number of Molecules covering this location",
            formatter: ThousandsIntFormat
        },
        {
            id: "allele_cov_plus", field: "allele_cov_plus", width: 20, minWidth: 20, sortable: true,
            name: "Allele Mol Cov", toolTip: "Allele Molecular Coverage: Number of detected molecules containing alternative allele",
            formatter: ThousandsIntFormat
        },
        {
            id: "allele_cov_minus", field: "allele_cov_minus", width: 20, minWidth: 20, sortable: true,
            name: "Allele Mol Freq", toolTip: "Allele Molecular Frequency: Frequency of molecules containing alternative allele",
            formatter: PercentFormatAF
        }
    ];

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

    TVC.dataView = new Slick.Data.DataView({inlineFilters: true});
    TVC.grid = new Slick.Grid("#AL-grid", TVC.dataView, columns, options);
    TVC.grid.setSelectionModel(new Slick.RowSelectionModel({selectActiveRow: false}));

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

    if ($.QueryString["debug"]) {
        $('<span class="btn" id="suspect" style="margin-left: 10px;">Export for Troubleshooting</span>').appendTo(".slick-pager");

        var table = '<div class="grid-header" id="toInspect" style="margin-top:35px; padding: 5px; width: 99%;"><h3><i class="icon-zoom-in"></i> Variants to inspect</h3>';
        table += '<table class="table" id="inspectTable">';
        table += '<thead id="inspectHead" style="display: none;">';
        table += '<tr> <th>Position</th> <th>Reference</th> <th>Variant</th> <th>Expected Variant</th> <th>Remove</th></tr>';
        table += '</thead>';
        table += '<tbody id="inspectBody"></tbody></table> <div id="manualInspectAdd" class="btn">Add Manually</div>';
        table += '<div id="exportInspect" class="btn btn-primary" style="margin-left: 10px;">Export</div>';
        table += '<div id="inspectOutput" style="padding-top: 10px;"></div> </div>';
        $("#allelesTable").parent().append(table);
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
        TVC.order_col = TVC.col_lookup[args["sortCols"][0]["sortCol"]["field"]];
        if (TVC.order_col == "position") {TVC.order_col = "Location";}
        TVC.order_dir = args["sortCols"][0]["sortAsc"];
        TVC.subload(0);
        TVC.pos = 0;
        TVC.pager_toggle();
    });

    TVC.filterSettings = {"Allele Call":["Heterozygous","Homozygous"], "Allele Source":["Hotspot"]};
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

        TVC.filterSettings[TVC.col_lookup["chrom"]] = selected;
        updateFilter();
    });

    $("#AL-selectAlleleSource").change(function (e) {
        var selected;
        if ($(this).val()) {
            selected = $(this).val();
        } else {
            selected = "";
        }
        TVC.filterSettings[TVC.col_lookup["allele_source"]] = selected;
        updateFilter();
        console.log(TVC.filterSettings);
    });

    $("#AL-selectVarType").change(function (e) {
        var selected;
        if ($(this).val()) {
            selected = $(this).val();
        } else {
            selected = "";
        }
        TVC.filterSettings[TVC.col_lookup["type"]] = selected;
        updateFilter();
    });

    $("#AL-selectAlleleCall").change(function (e) {
        var selected;
        if ($(this).val()) {
            selected = $(this).val();
        } else {
            selected = "";
        }
        TVC.filterSettings[TVC.col_lookup["allele_call"]] = selected;
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
                var chr = fields[0];
                //map the API data into the SlickGrid array
                //Just keep 1 page in memory at any time
                TVC.data[n] = {
                    id: Number(pk),
            key: TVC.pos + n,
                    chrom: chr,
                    position: (chr + ":" + fields[1]),
                    pos: fields[1],
                    ref: fields[2],
                    variant: fields[3],
                    allele: fields[49],
                    freq: Number(fields[6]),
                    quality: Number(fields[7]),
                    quality_filter: fields[8],
                    LOD: fields[9],

                    allele_call: fields[4],
                    allele_call_filter: fields[5],
                    type: fields[10],
                    allele_source: fields[11],
                    allele_id: fields[12],
                    gene_id: fields[13],
                    submitted_region: fields[14],
                    vcf_pos: fields[15],
                    vcf_ref: fields[16],
                    vcf_alt: fields[17],

                    total_cov: Number(fields[18]),
                    downsampled_cov_total: Number(fields[19]),
                    downsampled_cov_total_filter: fields[20],
                    total_cov_plus: Number(fields[21]),
                    total_cov_plus_filter: fields[22],
                    total_cov_minus: Number(fields[23]),
                    total_cov_minus_filter: fields[24],
                    allele_cov: Number(fields[25]),
                    allele_cov_plus: Number(fields[26]),
                    allele_cov_minus: Number(fields[27]),
                    //strand_bias: Number(fields[28]),
                    //strand_bias_filter: fields[29],

                    rbi: Number(fields[30]),
                    rbi_filter: fields[31],
                    refb: Number(fields[32]),
                    refb_filter: fields[33],
                    varb: Number(fields[34]),
                    varb_filter: fields[35],
                    mlld: Number(fields[36]),
                    mlld_filter: fields[37],
                    hp_length: Number(fields[38]),
                    hp_length_filter: fields[39],
                    sse_plus: Number(fields[40]),
                    sse_plus_filter: fields[41],
                    sse_minus: Number(fields[42]),
                    sse_minus_filter: fields[43],
                    sssb: Number(fields[43]),
                    sssb_filter: fields[45]
                };
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
