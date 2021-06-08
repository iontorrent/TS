// html for table container - note these are invisible and moved into position later
document.write('\
<div id="SIDAC-tablecontent" style="display:none">\
  <div id="SIDAC-titlebar" class="grid-header">\
    <span id="SIDAC-collapseGrid" style="float:left" class="ui-icon ui-icon ui-icon-triangle-1-n" title="Collapse view"></span>\
    <span class="table-title">Allele Coverage for Sample Identification SNPs</span>\
    <span id="SIDAC-message" class="message"></span>\
  </div>\
  <div id="SIDAC-grid" class="grid-body"></div>\
  <p class="grid-text"/>\
</div>\
');

$(function () {

var disableTitleBar = false;

$("#SIDAC-collapseGrid").click(function(e) {
  if( disableTitleBar ) return;
  if( $('#SIDAC-grid').is(":visible") ) {
    $(this).attr("class","ui-icon ui-icon-triangle-1-s");
    $(this).attr("title","Expand view");
    $('#SIDAC-grid').slideUp('slow',function(){
      $('#SIDAC-titlebar').css("border","1px solid grey");
    });
  } else {
    $(this).attr("class","ui-icon ui-icon-triangle-1-n");
    $(this).attr("title","Collapse view");
    $('#SIDAC-titlebar').css("border-bottom","0");
    $('#SIDAC-grid').slideDown();
  }
});

var dataFile = $("#sampleIDalleleCoverageTable").attr("fileurl");

function ChromIGV(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "N/A" }
  var pos = grid.getData().getItem(row)['chrom'] + ":" + value;
  var locpath = window.location.pathname.substring(0,window.location.pathname.lastIndexOf('/'));
  var igvURL = window.location.protocol + "//" + window.location.host + "/auth" + locpath + "/igv.php3";
  // link to Broad IGV
  //var href = "http://www.broadinstitute.org/igv/projects/current/igv.php?locus="+pos+"&sessionURL="+igvURL;
  // link to internal IGV
  var launchURL = window.location.protocol + "//" + window.location.host + "/IgvServlet/igv";
  var href = launchURL + "?locus="+pos+"&sessionURL="+igvURL;
  return "<a href='"+href+"'>"+value+"</a>";
}

function TaqmanAssay(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "N/A" }
  var href = "https://bioinfo.invitrogen.com/genome-database/searchResults?productTypeSelect=genotyping&targetTypeSelect=snp_all&keyword=" + value;
  return "<a href='"+href+"'>"+value+"</a>";
}

function fracToPC(row, cell, value, columnDef, dataContext) {
  if (value == null || value === "") { return "N/A" }
  return value+"%";
}

var columns = [{
  id: "chrom", name: "Chrom", field: "chrom", width: 56, minWidth: 40, maxWidth: 100, sortable: true,
  toolTip: "The chromosome (or contig) name in the reference genome."
},{
  id: "position", name: "Position", field: "position", width: 75, minWidth: 38, maxWidth: 80, sortable: true, formatter: ChromIGV,
  toolTip: "The one-based position in the reference genome. Click the link to open the position in IGV and view all reads covering the position."
},{
  id: "targetid", name: "Target ID", field: "targetid", width: 70, minWidth: 40, maxWidth: 200, sortable: true,
  toolTip: "Name of the target region containing the marker variant."
},{
  id: "hotspotid", name: "TaqMan Assay ID", field: "hotspotid", width: 110, minWidth: 40, maxWidth: 200, sortable: true, formatter: TaqmanAssay,
  toolTip: "TaqMan Assay ID string associated with the marker variant. Click the link to place an order for this assay."
},{
  id: "call", name: "Call", field: "call", width: 30, minWidth: 20, maxWidth: 40,
  toolTip: "Call based on reads at variant locus. Heterozygous calls use IUPAC SNP codes: M = A/C, R = A/G, W = A/T, S = C/G, Y = C/T, K = G/T."
},{
  id: "reference", name: "Ref", field: "reference", width: 30, minWidth: 20, maxWidth: 40,
  toolTip: "The reference base."
},{
  id: "allelefreq", name: "AF", field: "allelefreq", width: 52, minWidth: 36, maxWidth: 74, sortable: true, formatter: fracToPC,
  toolTip: "Allele frequency: Percentage of major to major plus minor allele reads. The major and minor alleles are those with the highest and second highest number of reads, after discounting any reads with adjacent insertions."
},{
  id: "coverage", name: "Cov", field: "coverage", width: 52, minWidth: 36, maxWidth: 74, sortable: true,
  toolTip: "The total reads covering the SNV locus, including reads with aligned deletions and adjacent insertions."
},{
  id: "cov_a", name: "A Reads", field: "cov_a", width: 52, minWidth: 36, maxWidth: 74,
  toolTip: "Number of reads calling A."
},{
  id: "cov_c", name: "C Reads", field: "cov_c", width: 52, minWidth: 36, maxWidth: 74,
  toolTip: "Number of reads calling C."
},{
  id: "cov_g", name: "G Reads", field: "cov_g", width: 52, minWidth: 36, maxWidth: 74,
  toolTip: "Number of reads calling G."
},{
  id: "cov_t", name: "T Reads", field: "cov_t", width: 52, minWidth: 36, maxWidth: 74,
  toolTip: "Number of reads calling T."
},{
  id: "cov_d", name: "Deletions", field: "cov_d", width: 55, minWidth: 36, maxWidth: 74,
  toolTip: "Number of reads calling deletion at the SNV locus. Deletion reads may be considered as the minor allele for the reported homozygous mahor allele frequency. But if there are sufficient deletion reads to make a heterozygous or homozygous deletion genotype call a '?' sampleID genotype is reported since this is unexpected."
},{
  id: "cov_i", name: "Insertions", field: "cov_i", width: 57, minWidth: 36, maxWidth: 74,
  toolTip: "Number of reads with an insertion aligned adjancent to the SNV locus. Insertion reads are ignored for genotyping and only contibute to the total coverage count, Cov. These have previously caused incorrect or uncalled genotypes due to ambiguous alignment at the SNV locus."
},{
  id: "cov_f", name: "Cov+", field: "cov_f", width: 52, minWidth: 36, maxWidth: 74, sortable: true,
  toolTip: "Number of forward reads aligned over the reference base excluding deletions and insertion alignments."
},{
  id: "cov_r", name: "Cov-", field: "cov_r", width: 52, minWidth: 36, maxWidth: 74, sortable: true,
  toolTip: "Number of reverse reads aligned over the reference base excluding deletions and insertion alignments."
}];

$("#sampleIDalleleCoverageTable").css('width','899');

// define the grid and attach head/foot of the table
var options = {
  editable: true,
  autoEdit: false,
  forceFitColumns: false,
  enableCellNavigation: true,
  multiColumnSort: true
};
var dataView = new Slick.Data.DataView({inlineFilters: true});
var grid = new Slick.Grid("#SIDAC-grid", dataView, columns, options);
grid.setSelectionModel( new Slick.RowSelectionModel({selectActiveRow: false}) );

var columnpicker = new Slick.Controls.ColumnPicker(columns, grid, options);

// move the hidden panels to their positioning targets and display contents
$("#SIDAC-tablecontent").appendTo('#sampleIDalleleCoverageTable');
$("#SIDAC-tablecontent").show();

// multi-column sort method: uses data type but with original mapping for chromosome
var chrMap = [];

grid.onSort.subscribe(function(e,args) {
  var cols = args.sortCols;
  dataView.sort(function (dataRow1, dataRow2) {
    for( var i = 0, l = cols.length; i < l; i++ ) {
      var field = cols[i].sortCol.field;
      var value1 = dataRow1[field];
      var value2 = dataRow2[field];
      if( value1 == value2 ) continue;
      if( field === 'chrom' ) {
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
});

// set to default to 0 rows, including header
$("#SIDAC-grid").css('height','27px');
$("#SIDAC-grid").resizable({
  alsoResize: "#sampleIDalleleCoverageTable",
  minWidth:300,
  handles:"e,s,se",
  stop:function(e,u) {
    $("#sampleIDalleleCoverageTable").css('height','auto');
  }
});
grid.resizeCanvas();

// initialize the model after all the events have been hooked up
var data = []; // defined by file load later
dataView.beginUpdate();
dataView.setItems(data);
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
    if( firstPartialLoad ) {
      firstPartialLoad = false;
      var numDataRows = (numRecords < initialRowDisplay) ? numRecords : initialRowDisplay;
      $("#SIDAC-grid").css('height',(numDataRows*25+25)+'px');
    }
    dataView.setItems(data);
    grid.resizeCanvas();
    grid.render();
  }

  function onLoadSuccess() {
    onLoadPartial();
    $('#SIDAC-message').html('');
  }

  function onLoadError() {
    if( errorTrace <= 1 ) {
      disableTitleBar = true;
      $('#SIDAC-grid').hide();
      $('#SIDAC-titlebar').css("border","1px solid grey");
      $('#SIDAC-collapseGrid').attr('class','ui-icon ui-icon-alert');
      $('#SIDAC-collapseGrid').attr("title","Failed to load data.");
      $('.grid-footnote').html('');
    }
    if( errorTrace < 0 ) {
      alert("Could open Allele Coverage table data file\n'"+dataFile+"'.");
    } else {
      alert("An error occurred loading Allele Coverage data from file\n'"+dataFile+"' at line "+errorTrace);
    }
    $('#SIDAC-message').append('<span style="color:red;font-style:normal">ERROR</span>');
  }

  $('#SIDAC-message').html('Loading...');
  if( dataFile == null || dataFile == undefined || dataFile == "" ) {
    return onLoadError();
  }

  $.get(dataFile, function(mem) {
    var lines = mem.split("\n");
    $.each(lines, function(n,row) {
      errorTrace = n;
      var fields = $.trim(row).split('\t');
      var chr = fields[0];
      if( chr == '' ) return true; // continue
      if( n > 0 ) {
        data[numRecords] = {
          id : Number(numRecords),
          chrom : chr,
          position : Number(fields[1]),
          targetid : fields[2],
          hotspotid : fields[3],
          call : fields[4],
          reference : fields[5],
          allelefreq : Number(fields[6]),
          coverage : Number(fields[7]),
          cov_a : Number(fields[8]),
          cov_c : Number(fields[9]),
          cov_g : Number(fields[10]),
          cov_t : Number(fields[11]),
          cov_d : Number(fields[12]),
          cov_i : Number(fields[13]),
          cov_f : Number(fields[14]),
          cov_r : Number(fields[15])
        };
        // record unique identifies and order of chromosomes from source
        if( selectAppendUnique('#SIDAC-selectChrom',chr,chr) ) { chrMap[chr] = chrNum++; }
        ++numRecords;
        if( loadUpdate > 0 && numRecords % loadUpdate == 0 ) onLoadPartial();
      }
    });
  }).success(onLoadSuccess).error(onLoadError);
}
postPageLoadMethods.push({callback: loadtable, priority: 30});
  
});
