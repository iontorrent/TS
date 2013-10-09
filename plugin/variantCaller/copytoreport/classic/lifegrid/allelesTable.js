$(function () {

    //new select boxes from http://silviomoreto.github.io/bootstrap-select/
    $('.selectpicker').selectpicker();

    var filterSettings = {};

    function resetFilterSettings() {
        filterSettings = {
            searchStringChrom: "",
            searchStringPosStart: Number(0),
            searchStringPosEnd: Number(0),
            searchStringAlleleID: "",
            searchStringAlleleSource: "",
            searchStringVarType: "",
            searchStringAlleleCall: ["Heterozygous","Homozygous"],
            searchStringFreqMin: Number(0),
            searchStringFreqMax: Number(100),
            searchStringCovMin: Number(0)
        };
    }

    function updateFilterSettings() {
        $("#AL-selectChrom").attr('value', filterSettings['searchStringChrom']);
        $("#AL-txtSearchPosStart").attr('value', filterSettings['searchStringPosStart'] ? "" : filterSettings['searchStringPosStart']);
        $("#AL-txtSearchPosEnd").attr('value', filterSettings['searchStringPosEnd'] ? "" : filterSettings['searchStringPosEnd']);
        $("#AL-txtSearchAlleleID").attr('value', filterSettings['searchStringAlleleID']);
        $("#AL-selectAlleleSource").attr('value', filterSettings['searchStringAlleleSource']);
        $("#AL-selectVarType").attr('value', filterSettings['searchStringVarType']);
        $("#AL-selectAlleleCall").val(filterSettings['searchStringAlleleCall']);
        $("#AL-selectAlleleCall").change();
        $("#AL-txtSearchFreqMin").attr('value', filterSettings['searchStringFreqMin']);
        $("#AL-txtSearchFreqMax").attr('value', filterSettings['searchStringFreqMax']);
        $("#AL-txtSearchCovMin").attr('value', filterSettings['searchStringCovMin'] ? "" : filterSettings['searchStringCovMin']);
    }

    function myFilter(item, args) {
        // for selected only filtering ignore all other filters

        if (args.searchStringChrom != ""  && $.inArray(item["chrom"], args.searchStringChrom) < 0 ) return false;
        if (args.searchStringAlleleSource != "" && $.inArray(item["allele_source"],  args.searchStringAlleleSource) < 0 ) return false;
        if (args.searchStringVarType != "" && $.inArray(item["type"], args.searchStringVarType) <0 ) return false;
        if (args.searchStringAlleleCall != "" && $.inArray(item["allele_call"] , args.searchStringAlleleCall) < 0 ) return false;

        if (rangeNoMatch(item["pos"], args.searchStringPosStart, args.searchStringPosEnd)) return false;
        if (strNoMatch(item["allele_id"].toUpperCase(), args.searchStringAlleleID)) return false;
        if (rangeNoMatch(item["freq"], args.searchStringFreqMin, args.searchStringFreqMax)) return false;
        if (rangeLess(item["total_cov"], args.searchStringCovMin)) return false;
        return true;
    }

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
          end_str = start_str.substring(len-3,len);
          start_str = start_str.substring(0,len-3);
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
        $(cellNode).attr('title',"Parameter thresholds not met:" + parsed_tooltip);
      }
    }
    
    var dataFile = $("#allelesTable").attr("fileurl");

    //add this to the window object so we can grab it everywhere
    window.TVC = {};
    var columns = [];

    //the columns shown in every view
    TVC.all = [
        {
            id: "position", field: "position", width: 25, minWidth: 25, sortable: true,
            name: "Position", toolTip: "Allele position in reference genome. For hotspot alleles this is the original position before left alignment.",
            formatter: ChromIGV
        },
        {
            id: "ref", field: "ref", width: 10, minWidth: 10,
            name: "Ref", toolTip: "Reference sequence."
        },
        {
            id: "variant", field: "variant", width: 10, minWidth: 10,
            name: "Variant", toolTip: "Allele sequence that replaces reference sequence in the variant."
        },
        {
            id: "allele_call", field: "allele_call", width: 20, minWidth: 20, sortable: true,
            name: "Allele Call", toolTip: "Decision whether the allele is detected, not detected, or filtered",
            asyncPostRender: MarkFilter
        },
        {
            id: "freq", field: "freq", width: 15, minWidth: 15, sortable: true,
            name: "Frequency", toolTip: "Allele frequency, the ratio between (downsampled) allele coverage and total coverage",
            formatter: PercentFormat
        },
        {
          id: "quality", field: "quality", width: 15, minWidth: 15, sortable: true,
          name: "Quality", toolTip: "PHRED-scaled probability of incorrect call",
          formatter: Fixed1Format, asyncPostRender: MarkFilter
        },
        {
          id: "spacer", name: "", width: 1, minWidth: 1, maxWidth: 10, cssClass: "separator-bar"
      }
    ];

    //the columns shown in allele search view
    TVC.allele = [
        {
          id: "type", field: "type", width: 32, minWidth: 32, sortable: true,
          name: "Variant Type", toolTip: "SNP, INS, DEL, MNP, or COMPLEX"
        },
        {
          id: "allele_source", field: "allele_source", width: 32, minWidth: 32, sortable: true,
          name: "Allele Source", toolTip: '"Hotspot" for alleles in hotspots file, otherwise "Novel"'
        },
        {
          id: "allele_id", field: "allele_id", width: 32, minWidth: 32, sortable: true,
          name: "Allele Name",  toolTip: "Allele Name, read from the hotspot file"
        },
        {
            id: "gene_id", field: "gene_id", width: 32, minWidth: 32, sortable: true,
            name: "Gene ID", toolTip: "Gene Symbol, read from the target regions file"
        },
        {
            id: "submitted_region", field: "submitted_region", width: 32, minWidth: 32, sortable: true,
            name: "Region Name", toolTip: "Region name, read from target regions file"
        }
    ];

    //the columns shown in coverage view
    TVC.coverage = [
        {
            id: "downsampled_cov_total", field: "downsampled_cov_total", width: 20, minWidth: 20, sortable: true,
            name: "Coverage", toolTip: "Total coverage at this position, after downsampling",
            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
        },
        {
            id: "total_cov_plus", field: "total_cov_plus", width: 20, minWidth: 20, sortable: true,
            name: "Coverage +",toolTip: "Total coverage on forward strand, after downsampling ",
            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
        },
        {
            id: "total_cov_minus",  field: "total_cov_minus", width: 20, minWidth: 20, sortable: true,
            name: "Coverage -", toolTip: "Total coverage on reverse strand, after downsampling",
            formatter: ThousandsIntFormat, asyncPostRender: MarkFilter
        },
        {
            id: "allele_cov", field: "allele_cov", width: 20, minWidth: 20, sortable: true,
            name: "Allele Cov",toolTip: "Allele coverage (reads containing this allele), after downsampling",
            formatter: ThousandsIntFormat
        },
        {
            id: "allele_cov_plus",  field: "allele_cov_plus", width: 20, minWidth: 20, sortable: true,
            name: "Allele Cov +", toolTip: "Allele coverage on forward strand, after downsampling",
            formatter: ThousandsIntFormat
        },
        {
            id: "allele_cov_minus", field: "allele_cov_minus", width: 20, minWidth: 20, sortable: true,
            name: "Allele Cov -", toolTip: "Allele coverage on reverse strand, after downsampling",
            formatter: ThousandsIntFormat
        },
        {
            id: "strand_bias",  field: "strand_bias", width: 20, minWidth: 20, sortable: true,
            name: "Strand Bias", toolTip: "Strand bias is a measure of discrepancy between allele frequencies on forward and reverse strands",
            formatter: StrandBiasFormat, asyncPostRender: MarkFilter
        },
        {
            id: "beta_bias", field: "beta_bias", width: 20, minWidth: 20, sortable: true,
            name: "Beta Bias", toolTip: "Beta bias is an alternative measure of discrepancy between allele frequencies on forward and reverse strands",
            formatter: Fixed3Format, asyncPostRender: MarkFilter
        }
    ];

    //the columns shown in quality view
    TVC.quality = [
        {
            id: "hp_length", field: "hp_length", width: 20, minWidth: 20, sortable: true,
            name: "HP Length", toolTip: "Homopolymer length",
            formatter: RightAlignFormat, asyncPostRender: MarkFilter
        },
        {
            id: "rbi", field: "rbi", width: 20, minWidth: 20, sortable: true,
            name: "Common Signal Shift", toolTip: "rbi",
            formatter: Fixed3Format, asyncPostRender: MarkFilter
        },
        {
            id: "refb", field: "refb", width: 20, minWidth: 20, sortable: true,
            name: "Reference Signal Shift", toolTip: "REFB",
            formatter: Fixed3Format
        },
        {
            id: "varb",  field: "varb", width: 20, minWidth: 20, sortable: true,
            name: "Variant Signal Shift", toolTip: "REFB",
            formatter: Fixed3Format, asyncPostRender: MarkFilter
        },
        {
            id: "mlld", field: "mlld", width: 20, minWidth: 20, sortable: true,
            name: "Relative Read Quality", toolTip: "MLLD",
            formatter: Fixed1Format, asyncPostRender: MarkFilter
        },
        {
            id: "sse_plus", field: "sse_plus", width: 20, minWidth: 20, sortable: true,
            name: "Context Error +", toolTip: "Probability of sequence-specific-error on forward strand (DELs only)",
            formatter: SSEFormat, asyncPostRender: MarkFilter
        },
        {
            id: "sse_minus", field: "sse_minus", width: 20, minWidth: 20, sortable: true,
            name: "Context Error -", toolTip: "Probability of sequence-specific-error on reverse strand (DELs only)",
            formatter: SSEFormat, asyncPostRender: MarkFilter
        },
        {
            id: "sssb", field: "sssb", width: 20, minWidth: 20, sortable: true,
            name: "Context Strand Bias", toolTip: "Basespace strand bias (DELs only)",
            formatter: SSEFormat, asyncPostRender: MarkFilter
        }
    ];

    columns = columns.concat(TVC.all, TVC.allele);


// define the grid and attach head/foot of the table
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

    var pager = new Slick.Controls.Pager(TVC.dataView, TVC.grid, null, $("#AL-pager"));
    //var columnpicker = new Slick.Controls.ColumnPicker(columns, TVC.grid, options);

    $("#AL-tablecontent").appendTo('#allelesTable');
    $("#AL-tablecontent").show();

    var chrMap = [];

    TVC.grid.onSort.subscribe(function (e, args) {
        var cols = args.sortCols;
        TVC.dataView.sort(function (dataRow1, dataRow2) {
            for (var i = 0, l = cols.length; i < l; i++) {
                var field = cols[i].sortCol.field;

                if (field == 'position') {
                    var value1 = chrMap[dataRow1['chrom']];
                    var value2 = chrMap[dataRow2['chrom']];
                    if (value1 != value2) {
                        var sign = cols[i].sortAsc ? 1 : -1;
                        return (value1 > value2) ? sign : -sign;
                    }
                    value1 = dataRow1['pos'];
                    value2 = dataRow2['pos'];
                    if (value1 == value2) continue;
                    var sign = cols[i].sortAsc ? 1 : -1;
                    return (value1 > value2) ? sign : -sign;

                } else {
                    var value1 = dataRow1[field];
                    var value2 = dataRow2[field];
                    if (value1 == value2) continue;
                    var sign = cols[i].sortAsc ? 1 : -1;
                    return (value1 > value2) ? sign : -sign;
                }

            }
            return 0;
        });
    });


// wire up model events to drive the grid
    TVC.dataView.onRowCountChanged.subscribe(function (e, args) {
        TVC.grid.updateRowCount();
        TVC.grid.render();
    });

    TVC.dataView.onRowsChanged.subscribe(function (e, args) {
        TVC.grid.invalidateRows(args.rows);
        TVC.grid.render();
    });

    $("#AL-selectChrom").change(function (e) {

        var selected;
        if ($(this).val()){
            selected = $(this).val();
        }else{
            selected = "";
        }

        filterSettings['searchStringChrom'] = selected;
        console.log(filterSettings['searchStringChrom']);
        updateFilter();
    });

    $("#AL-selectAlleleSource").change(function (e) {
        var selected;
        if ($(this).val()){
            selected = $(this).val();
        }else{
            selected = "";
        }
        filterSettings['searchStringAlleleSource'] = selected;
        updateFilter();
    });

    $("#AL-selectVarType").change(function (e) {
        var selected;
        if ($(this).val()){
            selected = $(this).val();
        }else{
            selected = "";
        }
        filterSettings['searchStringVarType'] = selected;
        updateFilter();
    });

    $("#AL-selectAlleleCall").change(function (e) {
        var selected;
        if ($(this).val()){
            selected = $(this).val();
        }else{
            selected = "";
        }
        filterSettings['searchStringAlleleCall'] = selected;
        updateFilter();
    });

    $("#AL-txtSearchPosStart").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        this.value = this.value.replace(/\D/g, "");
        filterSettings['searchStringPosStart'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#AL-txtSearchPosEnd").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        this.value = this.value.replace(/\D/g, "");
        filterSettings['searchStringPosEnd'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#AL-txtSearchAlleleID").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        filterSettings['searchStringAlleleID'] = this.value.toUpperCase();
        updateFilter();
    });


    $("#AL-txtSearchFreqMin").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = 0;
        }
        this.value = forceStringFloat(this.value);
        filterSettings['searchStringFreqMin'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#AL-txtSearchFreqMax").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = 100;
        }
        this.value = forceStringFloat(this.value);
        filterSettings['searchStringFreqMax'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#AL-txtSearchCovMin").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        this.value = this.value.replace(/\D/g, "");
        filterSettings['searchStringCovMin'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });


    function updateFilter() {
        TVC.dataView.setFilterArgs(filterSettings);
        TVC.dataView.refresh();
    }

    resetFilterSettings();
    updateFilterSettings();

// set to default to 0 rows, including header
    $("#AL-grid").css('height', '27px');
    TVC.grid.resizeCanvas();

    //resize on browser resize
    $(window).resize(function () {
        TVC.grid.resizeCanvas();
    });

// initialize the model after all the events have been hooked up
    TVC.data = []; // defined by file load later
    TVC.dataView.beginUpdate();
    TVC.dataView.setItems(TVC.data);
    TVC.dataView.setFilterArgs(filterSettings);
    TVC.dataView.setFilter(myFilter);
    TVC.dataView.endUpdate();
    TVC.dataView.syncGridSelection(TVC.grid, true);


// define function to load the table data and add to onload call list
// - dataView, grid, columns, data and chrMap[] all defined above
    function loadtable() {
        var errorTrace = -1;
        var loadUpdate = 10000;
        var firstPartialLoad = true;
        var chrNum = 0;
        var numRecords = 0;
        var initialRowDisplay = 10;

        function onLoadPartial() {
            if (firstPartialLoad) {
                firstPartialLoad = false;
                var numDataRows = (numRecords < initialRowDisplay) ? numRecords : initialRowDisplay;
                $("#AL-grid").css('height', (numDataRows * 50 + 28) + 'px');
            }
            TVC.dataView.setItems(TVC.data);
            TVC.grid.resizeCanvas();
            TVC.grid.render();

        }

        function onLoadSuccess() {
            onLoadPartial();
            $('#AL-message').html('');
        }

        function onLoadError() {
            if (errorTrace <= 1) {
                $('#AL-pager').hide();
                $('#AL-grid').hide();
                $('#AL-titlebar').css("border", "1px solid grey");
                $('#AL-toggleFilter').attr('class', 'ui-icon ui-icon-alert');
                $('#AL-toggleFilter').attr("title", "Failed to load data.");
            }
            if (errorTrace < 0) {
                alert("Could open Variant Calls table data file\n'" + dataFile + "'.");
            } else {
                alert("An error occurred loading Variant Calls data from file\n'" + dataFile + "' at line " + errorTrace);
            }
            $('#AL-message').append('<span style="color:red;font-style:normal">ERROR</span>');
        }

        $('#AL-message').html('Loading...');
        if (dataFile == null || dataFile == undefined || dataFile == "") {
            return onLoadError();
        }

        //get the xls file and parses it, then put the data into the the 'data' array. Which is then used to fill up the grid
        $.get(dataFile,function (mem) {
            var lines = mem.split("\n");
            $.each(lines, function (n, row) {
                errorTrace = n;
                var fields = $.trim(row).split('\t');
                var chr = fields[0];
                if (n > 0 && chr != '') {
                    //this is where the map from xls to the table is made
                    TVC.data[numRecords] = {
                        id: Number(numRecords),
                        chrom: chr,
                        position: (chr + ":" + fields[1]),
                        pos:                           fields[1],
                        ref:                           fields[2],
                        variant:                       fields[3],
                        allele_id:                     fields[4],
                        allele_call:                   fields[5],
                        allele_call_filter:            fields[6],
                        freq:                   Number(fields[7]),
                        total_cov:              Number(fields[8]),

                        allele_source:                 fields[9],
                        type:                          fields[10],
                        gene_id:                       fields[11],
                        submitted_region:              fields[12],
                        vcf_pos:                       fields[13],

                        downsampled_cov_total:  Number(fields[14]),
                        downsampled_cov_total_filter:  fields[15],
                        total_cov_plus:         Number(fields[16]),
                        total_cov_plus_filter:         fields[17],
                        total_cov_minus:        Number(fields[18]),
                        total_cov_minus_filter:        fields[19],
                        allele_cov:             Number(fields[20]),
                        allele_cov_plus:        Number(fields[21]),
                        allele_cov_minus:       Number(fields[22]),
                        strand_bias:            Number(fields[23]),
                        strand_bias_filter:            fields[24],
                        beta_bias:              Number(fields[25]),
                        beta_bias_filter:              fields[26],

                        quality:                Number(fields[27]),
                        quality_filter:                fields[28],
                        hp_length:              Number(fields[29]),
                        hp_length_filter:              fields[30],
                        rbi:                    Number(fields[31]),
                        rbi_filter:                    fields[32],
                        refb:                   Number(fields[33]),
                        varb:                   Number(fields[34]),
                        varb_filter:                   fields[35],
                        mlld:                   Number(fields[36]),
                        mlld_filter:                   fields[37],
                        sse_plus:               Number(fields[38]),
                        sse_plus_filter:               fields[39],
                        sse_minus:              Number(fields[40]),
                        sse_minus_filter:              fields[41],
                        sssb:                   Number(fields[42]),
                        sssb_filter:                   fields[43]
                    };
                    if (selectAppendUnique('#AL-selectChrom', chr, chr)) {
                        chrMap[chr] = chrNum++;
                        $('.selectpicker').selectpicker('refresh');
                    }
                    ++numRecords;
                    if (loadUpdate > 0 && numRecords % loadUpdate == 0) onLoadPartial();
                }
            });
        }).success(onLoadSuccess).error(onLoadError);
    }

    loadtable();

    // I don't think this is even needed
    //TVC.grid.onMouseEnter.subscribe(handleMouseEnter);

    function handleMouseEnter(e) {
        var cell = TVC.grid.getCellFromEvent(e);
        if (cell) {
            var node = $(TVC.grid.getCellNode(cell.row, cell.cell));
            var item = TVC.dataView.getItem(cell.row);
            //title already needs to exist
            //var inner = $(node).find(".tooltip-cell");
            /*
             console.log(item);
             console.log(cell);
             console.log(TVC.grid.getColumns()[cell.cell]["id"]);
             */
        }
    }

    //table tabs
    $("#allele_search").click(function () {
        var columns = [];
        columns = columns.concat(TVC.all, TVC.allele);
        TVC.grid.setColumns(columns);
    });

    $("#coverage").click(function () {
        var columns = [];
        columns = columns.concat(TVC.all, TVC.coverage);
        TVC.grid.setColumns(columns);
    });

    $("#quality").click(function () {
        var columns = [];
        columns = columns.concat(TVC.all, TVC.quality);
        TVC.grid.setColumns(columns);
    });

});
