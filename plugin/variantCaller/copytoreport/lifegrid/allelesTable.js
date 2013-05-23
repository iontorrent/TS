$(function () {

    var filterSettings = {};

    function resetFilterSettings() {
        filterSettings = {
            searchStringChrom: "",
            searchStringPosStart: Number(0),
            searchStringPosEnd: Number(0),
            searchStringHotSpotID: "",
            searchStringAlleleSource: "",
            searchStringVarType: "",
            searchStringAlleleCall: "",
            searchStringFreqMin: Number(0),
            searchStringFreqMax: Number(100),
            searchStringCovMin: Number(0)
        };
    }

    function updateFilterSettings() {
        $("#AL-selectChrom").attr('value', filterSettings['searchStringChrom']);
        $("#AL-txtSearchPosStart").attr('value', filterSettings['searchStringPosStart'] ? "" : filterSettings['searchStringPosStart']);
        $("#AL-txtSearchPosEnd").attr('value', filterSettings['searchStringPosEnd'] ? "" : filterSettings['searchStringPosEnd']);
        $("#AL-txtSearchHotSpotID").attr('value', filterSettings['searchStringHotSpotID']);
        $("#AL-selectAlleleSource").attr('value', filterSettings['searchStringAlleleSource']);
        $("#AL-selectVarType").attr('value', filterSettings['searchStringVarType']);
        $("#AL-selectAlleleCall").attr('value', filterSettings['searchStringAlleleCall']);
        $("#AL-txtSearchFreqMin").attr('value', filterSettings['searchStringFreqMin']);
        $("#AL-txtSearchFreqMax").attr('value', filterSettings['searchStringFreqMax']);
        $("#AL-txtSearchCovMin").attr('value', filterSettings['searchStringCovMin'] ? "" : filterSettings['searchStringCovMin']);
    }


    function myFilter(item, args) {
        // for selected only filtering ignore all other filters
        if (args.searchStringChrom != "" && item["chrom"] != args.searchStringChrom) return false;
        if (rangeNoMatch(item["position"], args.searchStringPosStart, args.searchStringPosEnd)) return false;
        if (strNoMatch(item["hotspotid"].toUpperCase(), args.searchStringHotSpotID)) return false;
        if (args.searchStringAlleleSource != "" && item["source"] != args.searchStringAlleleSource) return false;
        if (args.searchStringVarType != "" && item["vartype"] != args.searchStringVarType) return false;
        if (args.searchStringAlleleCall != "" && item["allelecall"] != args.searchStringAlleleCall) return false;
        if (rangeNoMatch(item["varfreqnum"], args.searchStringFreqMin, args.searchStringFreqMax)) return false;
        if (rangeLess(item["coverage"], args.searchStringCovMin)) return false;
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
	
	
    var dataFile = $("#allelesTable").attr("fileurl");

    var columns = [];
    columns.push({
        id: "posstring", name: "Allele Position", field: "posstring", width: 60, minWidth: 60, sortable: true, formatter: ChromIGV,
        toolTip: "Allele position in reference genome. For hotspot alleles this is the original position before left alignment." });
    columns.push({
        id: "hotspotid", name: "Hotspot ID", field: "hotspotid", width: 70, minWidth: 70, sortable: true,
        toolTip: "Hotspot ID provided in the hotspot file." });
    columns.push({
        id: "source", name: "Allele Source", field: "source", width: 50, minWidth: 50, sortable: true,
        toolTip: "Distinguishes hotspot alleles provided via hotspot file from novel alleles not present in hotspot file." });
    columns.push({
        id: "reference", name: "Ref", field: "reference", width: 30, minWidth: 30,
        toolTip: "Allele reference base(s)." });
    columns.push({
        id: "variant", name: "Variant", field: "variant", width: 30, minWidth: 30,
        toolTip: "Allele variant base(s)." });
    columns.push({
        id: "vartype", name: "Type", field: "vartype", width: 30, minWidth: 30, sortable: true,
        toolTip: "Type of variation detected (SNP/INS/DEL)." });
    columns.push({
        id: "allelecall", name: "Allele Call", field: "allelecall", width: 50, minWidth: 50, sortable: true,
        toolTip: "Variant calling outcome (present/absent/no call)." });
    columns.push({
        id: "calldetails", name: "Call Details", field: "calldetails", width: 90, minWidth: 90, sortable: true,
        toolTip: "Ploidy for detected alleles (hom/het). Filtering reason for no calls." });
    columns.push({
        id: "varfreq", name: "Var Freq", field: "varfreq", width: 30, minWidth: 30, sortable: true, formatter: RightAlignFormat,
        toolTip: "Frequency of the variant allele." });
    columns.push({
        id: "varcoverage", name: "Allele Cov", field: "varcoverage", width: 30, minWidth: 30, sortable: true, formatter: RightAlignFormat,
        toolTip: "The number of downsampled reads covering the variant allele." });
    columns.push({
        id: "downcoverage", name: "Downsampled Cov", field: "downcoverage", width: 30, minWidth: 30, sortable: true, formatter: RightAlignFormat,
        toolTip: "The number of downsampled reads covering all alleles (including reference allele) at this position." });
    columns.push({
        id: "coverage", name: "Total Cov", field: "coverage", width: 30, minWidth: 30, sortable: true, formatter: RightAlignFormat,
        toolTip: "The total number of reads covering all alleles (including reference allele) at this position." });
    columns.push({
        id: "vcfpos", name: "VCF Position", field: "vcfpos", width: 60, minWidth: 60,
        toolTip: "GT pos." });


// define the grid and attach head/foot of the table
    var options = {
        editable: true,
        autoEdit: false,
        enableCellNavigation: true,
        multiColumnSort: true,
        forceFitColumns: true,
        syncColumnCellResize:true
    };

    var dataView = new Slick.Data.DataView({inlineFilters: true});
    var grid = new Slick.Grid("#AL-grid", dataView, columns, options);
    grid.setSelectionModel(new Slick.RowSelectionModel({selectActiveRow: false}));

    var pager = new Slick.Controls.Pager(dataView, grid, null, $("#AL-pager"));
    var columnpicker = new Slick.Controls.ColumnPicker(columns, grid, options);

    $("#AL-tablecontent").appendTo('#allelesTable');
    $("#AL-tablecontent").show();

    var chrMap = [];

    grid.onSort.subscribe(function (e, args) {
        var cols = args.sortCols;
        dataView.sort(function (dataRow1, dataRow2) {
            for (var i = 0, l = cols.length; i < l; i++) {
                var field = cols[i].sortCol.field;
                
                if (field == 'posstring') {
                    var value1 = chrMap[dataRow1['chrom']];
                    var value2 = chrMap[dataRow2['chrom']];
                    if (value1 != value2) {
                        var sign = cols[i].sortAsc ? 1 : -1;
                        return (value1 > value2) ? sign : -sign;
                    }
                    value1 = dataRow1['position'];
                    value2 = dataRow2['position'];
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
    dataView.onRowCountChanged.subscribe(function (e, args) {
        grid.updateRowCount();
        grid.render();
    });

    dataView.onRowsChanged.subscribe(function (e, args) {
        grid.invalidateRows(args.rows);
        grid.render();
    });

    
    $("#AL-selectChrom").change(function (e) {
        filterSettings['searchStringChrom'] = this.value;
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

    $("#AL-txtSearchHotSpotID").keyup(function (e) {
        Slick.GlobalEditorLock.cancelCurrentEdit();
        if (e.which == 27) {
            this.value = "";
        }
        filterSettings['searchStringHotSpotID'] = this.value.toUpperCase();
        updateFilter();
    });
    
    $("#AL-selectAlleleSource").change(function (e) {
        filterSettings['searchStringAlleleSource'] = this.value;
        updateFilter();
    });

    $("#AL-selectVarType").change(function (e) {
        filterSettings['searchStringVarType'] = this.value;
        updateFilter();
    });

    $("#AL-selectAlleleCall").change(function (e) {
        filterSettings['searchStringAlleleCall'] = this.value;
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
        dataView.setFilterArgs(filterSettings);
        dataView.refresh();
    }

    resetFilterSettings();
    updateFilterSettings();
    
    
    
    
    
// set to default to 0 rows, including header
    $("#AL-grid").css('height', '27px');
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
        var chrNum = 0;
        var numRecords = 0;
        var initialRowDisplay = 10;

        function onLoadPartial() {
            if (firstPartialLoad) {
                firstPartialLoad = false;
                var numDataRows = (numRecords < initialRowDisplay) ? numRecords : initialRowDisplay;
                $("#AL-grid").css('height', (numDataRows *  50 + 27) + 'px');
            }
            dataView.setItems(data);
            grid.resizeCanvas();
            grid.render();
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

        $.get(dataFile,function (mem) {
            var lines = mem.split("\n");
            $.each(lines, function (n, row) {
                errorTrace = n;
                var fields = $.trim(row).split('\t');
                var chr = fields[0];
                if (n > 0 && chr != '') {
                    data[numRecords] = {
                        id: Number(numRecords),
                        chrom: chr,
                        position: Number(fields[1]),
                        posstring: (chr+":"+fields[1]),
                        hotspotid: fields[2],
                        source: fields[3],
                        reference: fields[4],
                        variant: fields[5],
                        vartype: fields[6],
                        allelecall: fields[7],
                        calldetails: fields[8],
                        varfreq: fields[9],
                        varfreqnum: parseFloat(fields[9]),
                        varcoverage: Number(fields[10]),
                        downcoverage: Number(fields[11]),
                        coverage: Number(fields[12]),
                        vcfpos: fields[13]
                    };
                    if (selectAppendUnique('#AL-selectChrom', chr, chr)) {
                        chrMap[chr] = chrNum++;
                    }
                    ++numRecords;
                    if (loadUpdate > 0 && numRecords % loadUpdate == 0) onLoadPartial();
                }
            });
        }).success(onLoadSuccess).error(onLoadError);
    }

    postPageLoadMethods.push({callback: loadtable, priority: 40});

});
