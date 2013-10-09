$(function () {


    $("#VC-toggleFilter").click(function (e) {
        if ($('#VC-filterpanel').is(":visible")) {
            $('#VC-filterpanel').slideUp();
        } else if ($('#VC-grid').is(":visible")) {
            $('#VC-filterpanel').slideDown();
        }
    });

    var filterSettings = {};

    function resetFilterSettings() {
        filterSettings = {
            searchSelected: false,
            searchStringChrom: "",
            searchStringPosStart: Number(0),
            searchStringPosEnd: Number(0),
            searchStringGeneSym: "",
            searchStringTargetID: "",
            searchStringHotSpotID: "",
            searchStringVarType: "",
            searchStringPloidy: "",
            searchStringFreqMin: Number(0),
            searchStringFreqMax: Number(100),
            searchStringCovMin: Number(0)
        };
    }

    function updateFilterSettings() {
        updateSelectedFilter(false);
        $("#VC-selectChrom").attr('value', filterSettings['searchStringChrom']);
        $("#VC-txtSearchPosStart").attr('value', filterSettings['searchStringPosStart'] ? "" : filterSettings['searchStringPosStart']);
        $("#VC-txtSearchPosEnd").attr('value', filterSettings['searchStringPosEnd'] ? "" : filterSettings['searchStringPosEnd']);
        $("#VC-txtSearchGeneSym").attr('value', filterSettings['searchStringGeneSym']);
        $("#VC-txtSearchTargetID").attr('value', filterSettings['searchStringTargetID']);
        $("#VC-txtSearchHotSpotID").attr('value', filterSettings['searchStringHotSpotID']);
        $("#VC-selectVarType").attr('value', filterSettings['searchStringVarType']);
        $("#VC-selectPloidy").attr('value', filterSettings['searchStringPloidy']);
        $("#VC-txtSearchFreqMin").attr('value', filterSettings['searchStringFreqMin']);
        $("#VC-txtSearchFreqMax").attr('value', filterSettings['searchStringFreqMax']);
        $("#VC-txtSearchCovMin").attr('value', filterSettings['searchStringCovMin'] ? "" : filterSettings['searchStringCovMin']);
    }

    function updateSelectedFilter(turnOn) {
        filterSettings['searchSelected'] = turnOn;
        $('#VC-checkSelected').attr('class', turnOn ? 'checkOn btn' : 'checkOff btn');
        $('.txtSearch').attr('disabled', turnOn);
        $('.numSearch').attr('disabled', turnOn);
        checkboxSelector.setFilterSelected(turnOn);
    }

    function myFilter(item, args) {
        // for selected only filtering ignore all other filters
        if (args.searchSelected) return item["check"];
        if (args.searchStringChrom != "" && item["chrom"] != args.searchStringChrom) return false;
        if (strNoMatch(item["genesym"].toUpperCase(), args.searchStringGeneSym)) return false;
        if (strNoMatch(item["targetid"].toUpperCase(), args.searchStringTargetID)) return false;
        if (rangeNoMatch(item["position"], args.searchStringPosStart, args.searchStringPosEnd)) return false;
        if (args.searchStringVarType != "" && item["vartype"] != args.searchStringVarType) return false;
        if (args.searchStringPloidy != "" && item["ploidy"] != args.searchStringPloidy) return false;
        if (rangeNoMatch(item["varfreq"], args.searchStringFreqMin, args.searchStringFreqMax)) return false;
        if (rangeLess(item["coverage"], args.searchStringCovMin)) return false;
        if (item["hotspotid"] != undefined && strNoMatch(item["hotspotid"].toUpperCase(), args.searchStringHotSpotID)) return false;
        return true;
    }

    function exportTools() {
        // could use data[] here directly

        $("#closebutton").click(function () {
            $('#VC-dialog').modal('hide');
        });

        var items = dataView.getItems();
        var numSelected = 0;
        for (var i = 0; i < items.length; ++i) {
            if (items[i]['check']) ++numSelected;
        }
        var $content = $('#VC-dialog-content');
        $content.html('Rows selected: ' + numSelected + '<br/>');
        if (numSelected == 0) {
            $content.append('<p>You must first select rows of the table data to export.</p>');
            $('#VC-exportOK').hide();
        } else {
            $content.append('<div id="VCradio"><label class="radio">\
  <input type="radio" name="vcmodalradio" id="table" value="table" checked>\
  Download table file of selected rows.\
</label>\
<label class="radio">\
  <input type="radio" name="vcmodalradio" id="taqman" value="taqman">\
  Submit variants for TaqMan assay design.\
</label>\
<label class="radio" style="color: grey">\
  <input type="radio" name="vcmodalradio" id="ce" value="ce" disabled>\
  <i>Submit variants (human only) for PCR/Sanger sequencing primer design.</i>\
</label></div>');
            $('#VC-exportOK').show();
        }

        $('#VC-dialog').modal('show');

    }

    var dataFile = $("#variantCallsTable").attr("fileurl");

    $('#VC-exportOK').click(function (e) {
        $('#VC-dialog').modal('hide');
        // use ID's and resort to original order for original input file order matching
        var items = dataView.getItems();
        var checkList = [];
        for (var i = 0; i < items.length; ++i) {
            if (items[i]['check']) {
                checkList.push(items[i]['id']);
            }
        }
        var rows = checkList.sort(function (a, b) {
            return a - b;
        }) + ",";
        var op =  $("#VCradio input[type='radio']:checked").val();
        if (op == "table") {
            window.open("subtable.php3?dataFile=" + dataFile + "&rows=" + rows);
        } else if (op == "taqman") {
            window.open("taqman.php3?dataFile=" + dataFile + "&rows=" + rows);
        }
    });

    function ChromIGV(row, cell, value, columnDef, dataContext) {
        if (value == null || value === "") {
            return "N/A"
        }
        var pos = grid.getData().getItem(row)['chrom'] + ":" + value;
        var locpath = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/'));
        var igvURL = window.location.protocol + "//" + window.location.host + "/auth" + locpath + "/igv.php3";
        // link to Broad IGV
        //var href = "http://www.broadinstitute.org/igv/projects/current/igv.php?locus="+pos+"&sessionURL="+igvURL;
        // link to internal IGV
        var launchURL = window.location.protocol + "//" + window.location.host + "/IgvServlet/igv";
        var href = launchURL + "?locus=" + pos + "&sessionURL=" + igvURL;
        return "<a href='" + href + "'>" + value + "</a>";
    }

    var columns = [];
    var checkboxSelector = new Slick.CheckboxSelectColumn();
    columns.push(checkboxSelector.getColumnDefinition());
    columns.push({
        id: "chrom", name: "Chrom", field: "chrom", width: 65, minWidth: 65, maxWidth: 65, sortable: true,
        toolTip: "The chromosome (or contig) name in the reference genome." });
    columns.push({
        id: "position", name: "Position", field: "position", width: 65, minWidth: 65, sortable: true, formatter: ChromIGV,
        toolTip: "The one-based position in the reference genome. Click the link to open the variant in IGV and view all reads covering the variant." });
    columns.push({
        id: "genesym", name: "Gene Sym", field: "genesym", width: 72, minWidth: 40, sortable: true,
        toolTip: "Gene symbol for the gene where the variant is located. This value is not available (N/A) if no target regions were defined." });
    columns.push({
        id: "targetid", name: "Target ID", field: "targetid", width: 70, minWidth: 70, sortable: true,
        toolTip: "Name of the target region where the variant is located. This value is not available (N/A) if no target regions were defined." });
    columns.push({
        id: "vartype", name: "Type", field: "vartype", width: 46, minWidth: 40, sortable: true,
        toolTip: "Type of variantion detected (SNP/INDEL)." });
    columns.push({
        id: "ploidy", name: "Zygosity", field: "ploidy", width: 54, minWidth: 40, sortable: true,
        toolTip: "Assigned zygosity of the variation: Homozygous (Hom), Heterozygous (Het) or No Call (NC)." });
    columns.push({
        id: "genotype", name: "Genotype", field: "genotype", width: 54, minWidth: 40, sortable: true,
        toolTip: "Assigned genotype." });
    columns.push({
        id: "reference", name: "Ref", field: "reference", width: 36, minWidth: 28,
        toolTip: "The reference base(s)." });
    columns.push({
        id: "variant", name: "Variant", field: "variant", width: 44, minWidth: 38,
        toolTip: "Variant allele base(s)." });
    columns.push({
        id: "varfreq", name: "Var Freq", field: "varfreq", width: 68, minWidth: 50, sortable: true, formatter: formatPercent,
        toolTip: "Frequency of the variant allele." });
    columns.push({
        id: "p_value", name: "Qual", field: "p_value", width: 60, minWidth: 40, sortable: true,
        toolTip: "Estimated probability in phred scale that the variant could be produced by chance." });
    columns.push({
        id: "coverage", name: "Cov", field: "coverage", width: 50, minWidth: 40, sortable: true,
        toolTip: "The total number of reads covering the reference base position." });
    columns.push({
        id: "refcoverage", name: "Ref Cov", field: "refcoverage", width: 60, minWidth: 50, sortable: true,
        toolTip: "The number of reads covering the reference allele." });
    columns.push({
        id: "varcoverage", name: "Var Cov", field: "varcoverage", width: 61, minWidth: 50, sortable: true,
        toolTip: "The number of reads covering the variant allele." });

// set up assumind there is no hotspot field - defined when file is loaded
    $("#VC-filterHotSpot").hide();

// define the grid and attach head/foot of the table
    var options = {
        editable: true,
        autoEdit: false,
        enableCellNavigation: true,
        multiColumnSort: true,
        forceFitColumns: true,
        syncColumnCellResize: true
    };

    var dataView = new Slick.Data.DataView({inlineFilters: true});
    var grid = new Slick.Grid("#VC-grid", dataView, columns, options);
    grid.setSelectionModel(new Slick.RowSelectionModel({selectActiveRow: false}));
    grid.registerPlugin(checkboxSelector);

    var pager = new Slick.Controls.Pager(dataView, grid, exportTools, $("#VC-pager"));
    var columnpicker = new Slick.Controls.ColumnPicker(columns, grid, options);

    $("#VC-tablecontent").appendTo('#variantCallsTable');
    $("#VC-tablecontent").show();
    $("#VC-filterpanel").appendTo('#VC-titlebar');

// multi-column sort method: uses data type but with original mapping for chromosome
    var chrMap = [];

    grid.onSort.subscribe(function (e, args) {
        var cols = args.sortCols;
        dataView.sort(function (dataRow1, dataRow2) {
            for (var i = 0, l = cols.length; i < l; i++) {
                var field = cols[i].sortCol.field;
                var value1 = dataRow1[field];
                var value2 = dataRow2[field];
                if (value1 == value2) continue;
                if (field === 'chrom') {
                    value1 = chrMap[value1];
                    value2 = chrMap[value2];
                }
                var sign = cols[i].sortAsc ? 1 : -1;
                return (value1 > value2) ? sign : -sign;
            }
            return 0;
        });
    });

// wire up model events to drive the grid
    dataView.onRowCountChanged.subscribe(function (e, args) {
        grid.updateRowCount();
        grid.render();
    });

    dataView.onRowsChanged.subscribe(function (e, args) {
        grid.invalidateRows(args.rows);
        grid.render();
        checkboxSelector.checkAllSelected();
    });

// --- filter panel methods
    $("#VC-checkSelected").click(function (e) {
        var turnOn = ($(this).attr('class') === 'checkOff btn');
        updateSelectedFilter(turnOn);
        updateFilter();
        dataView.setPagingOptions({pageNum: 0});
    });

    $("#VC-clearSelected").click(function (e) {
        resetFilterSettings();
        updateFilterSettings();
        updateFilter();
    });

    $("#VC-selectChrom").change(function (e) {
        filterSettings['searchStringChrom'] = this.value;
        updateFilter();
    });

    $("#VC-txtSearchPosStart").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        this.value = this.value.replace(/\D/g, "");
        filterSettings['searchStringPosStart'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#VC-txtSearchPosEnd").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        this.value = this.value.replace(/\D/g, "");
        filterSettings['searchStringPosEnd'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#VC-txtSearchGeneSym").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        filterSettings['searchStringGeneSym'] = this.value.toUpperCase();
        updateFilter();
    });

    $("#VC-txtSearchTargetID").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        filterSettings['searchStringTargetID'] = this.value.toUpperCase();
        updateFilter();
    });

    $("#VC-selectVarType").change(function (e) {
        filterSettings['searchStringVarType'] = this.value;
        updateFilter();
    });

    $("#VC-selectPloidy").change(function (e) {
        filterSettings['searchStringPloidy'] = this.value;
        updateFilter();
    });

    $("#VC-txtSearchFreqMin").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = 0;
        }
        this.value = forceStringFloat(this.value);
        filterSettings['searchStringFreqMin'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#VC-txtSearchFreqMax").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = 100;
        }
        this.value = forceStringFloat(this.value);
        filterSettings['searchStringFreqMax'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#VC-txtSearchCovMin").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        this.value = this.value.replace(/\D/g, "");
        filterSettings['searchStringCovMin'] = Number(this.value == "" ? 0 : this.value);
        updateFilter();
    });

    $("#VC-txtSearchHotSpotID").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        filterSettings['searchStringHotSpotID'] = this.value.toUpperCase();
        updateFilter();
    });

    function updateFilter() {
        dataView.setFilterArgs(filterSettings);
        dataView.refresh();
    }

    checkboxSelector.setUpdateFilter(updateFilter);
    resetFilterSettings();
    updateFilterSettings();

// set to default to 0 rows, including header
    $("#VC-grid").css('height', '27px');
    grid.resizeCanvas();

    //resize on browser resize
    $(window).resize(function () {
        grid.resizeCanvas();
    });

// initialize the model after all the events have been hooked up
    var data = []; // defined by file load later
    dataView.beginUpdate();
    dataView.setItems(data);
    dataView.setFilterArgs(filterSettings);
    dataView.setFilter(myFilter);
    dataView.endUpdate();
    dataView.syncGridSelection(grid, true);

// define function to load the table data and add to onload call list
// - dataView, grid, columns, data and chrMap[] all defined above
    function loadtable() {
        var errorTrace = -1;
        var loadUpdate = 10000;
        var firstPartialLoad = true;
        var haveHotSpots = false;
        var chrNum = 0;
        var numRecords = 0;
        var initialRowDisplay = 10;

        function onLoadPartial() {
            if (firstPartialLoad) {
                firstPartialLoad = false;
                var numDataRows = (numRecords < initialRowDisplay) ? numRecords : initialRowDisplay;
                $("#VC-grid").css('height', (numDataRows * 50 + 27) + 'px');
                // add HotSpot ID column if data available show filter
                if (haveHotSpots) {
                    columns.push({
                        id: "hotspotid", name: "HotSpot ID", field: "hotspotid", width: 74, minWidth: 40, maxWidth: 200, sortable: true,
                        toolTip: "HotSpot ID for one or more starting locations matching the identified variant." });
                    grid.setColumns(columns);
                    $("#VC-filterHotSpot").show();
                }
            }
            dataView.setItems(data);
            grid.resizeCanvas();
            grid.render();
        }

        function onLoadSuccess() {
            onLoadPartial();
            $('#VC-message').html('');
        }

        function onLoadError() {
            if (errorTrace <= 1) {
                $('#VC-pager').hide();
                $('#VC-grid').hide();
                $('#VC-titlebar').css("border", "1px solid grey");
                $('#VC-toggleFilter').attr('class', 'ui-icon ui-icon-alert');
                $('#VC-toggleFilter').attr("title", "Failed to load data.");
            }
            if (errorTrace < 0) {
                alert("Could open Variant Calls table data file\n'" + dataFile + "'.");
            } else {
                alert("An error occurred loading Variant Calls data from file\n'" + dataFile + "' at line " + errorTrace);
            }
            $('#VC-message').append('<span style="color:red;font-style:normal">ERROR</span>');
        }

        $('#VC-message').html('Loading...');
        if (dataFile == null || dataFile == undefined || dataFile == "") {
            return onLoadError();
        }

        $.get(dataFile,function (mem) {
            var lines = mem.split("\n");
            $.each(lines, function (n, row) {
                errorTrace = n;
                var fields = $.trim(row).split('\t');
                var chr = fields[0];
                if (n > 0 && chr != '') {
                    data[numRecords] = {
                        id: Number(numRecords),
                        check: false,
                        chrom: chr,
                        position: Number(fields[1]),
                        genesym: fields[2],
                        targetid: fields[3],
                        vartype: fields[4],
                        ploidy: fields[5],
                        genotype: fields[6],
                        reference: fields[7],
                        variant: fields[8],
                        varfreq: Number(fields[9]),
                        p_value: Number(fields[10]),
                        coverage: Number(fields[11]),
                        refcoverage: Number(fields[12]),
                        varcoverage: Number(fields[13])
                    };
                    if (fields[14] != null && fields[14] != undefined) {
                        data[numRecords]['hotspotid'] = fields[14];
                        haveHotSpots = true;
                    }
                    // record unique identifies and order of chromosomes from source
                    if (selectAppendUnique('#VC-selectChrom', chr, chr)) {
                        chrMap[chr] = chrNum++;
                    }
                    selectAppendUnique('#VC-selectVarType', fields[4], fields[4]);
                    selectAppendUnique('#VC-selectPloidy', fields[5], fields[5]);
                    ++numRecords;
                    if (loadUpdate > 0 && numRecords % loadUpdate == 0) onLoadPartial();
                }
            });
        }).success(onLoadSuccess).error(onLoadError);
    }

    postPageLoadMethods.push({callback: loadtable, priority: 40});

});
