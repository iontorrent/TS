<?php

if (file_exists("expMeta.dat")) {
    include("parsefiles.php");
    $meta       = parseVersion("expMeta.dat");
    $djangoPK   = parse_to_keys("primary.key");
    $istoplevel = True;
} else {
    include("../parsefiles.php");
    $meta       = parseVersion("../expMeta.dat");
    $djangoPK   = parse_to_keys("../primary.key");
    $istoplevel = False;
}

$explog_dict = array();
if (file_exists("explog.txt")) {
    $file = fopen("explog.txt", 'r') or die();
    //Output a line of the file until the end is reached
    $dataX = array();
    $dataY = array();
    while (($line = fgets($file)) !== false) {
        $explog_dict_entry = preg_split("/:/", $line, 2);
        if (count($explog_dict_entry) == 2) {
            $name               = str_replace(" version", "", $explog_dict_entry[0]);
            $explog_dict[$name] = $explog_dict_entry[1];
        }

        // old format of explog.txt cannot be stored in a dictionary
        if (preg_match('/BlockStatus:/', $line)) {
            $blockline = preg_split('/BlockStatus:/', $line);
            //print_r($blockline); print '<br><br>';
        } else {
            // new explog.txt format
            $blockline = preg_split('/block_[0-9]{3}: /', $line);
        }

        $key = $blockline[0];
        if (!$key) {
            $value = $blockline[1];
            list($X, $Y) = str_getcsv($value, ",");
            $X = trim(ltrim($X), "X");
            $Y = trim(ltrim($Y), "Y");
            //exclude thumbnail directory
            if ($X == "-1")
                continue;
            if ($X == "thumbnail")
                continue;
            $dataX[] = $X;
            $dataY[] = $Y;
        }
    }
    fclose($file);
    $dataX = array_unique($dataX);
    $dataY = array_unique($dataY);
    sort($dataX);
    sort($dataY);
}

//Render the report using the look of the Django site
$base_template = "/opt/ion/iondb/templates/rundb/reports/generated/30_php_base.html";
$blank_report = file_get_contents($base_template);

//simple template, will break the page in half where [PHP] is found
$template = explode("[PHP]", $blank_report);
$header   = $template[0];
$footer   = $template[1];

$resultsName   = $meta["Analysis Name"];
$referenceName = $meta["Reference"];
$expName       = $meta["Run Name"];
$base_name     = $expName . '_' . $resultsName;



if (file_exists("ReportConfiguration.txt")) {
    $reportconfig = parseVersion("ReportConfiguration.txt");
}

$is_proton_thumbnail = False;
$is_proton_composite = False;
$is_proton = False;

if ( preg_match('/900/', $meta["Chip Type"]) or preg_match('/^P/', $meta["Chip Type"]) ) {
    $is_proton = True;
}

$display_q17 = False;
if ($is_proton) {
    $display_q17 = True;
    if (preg_match('/Thumbnail/', $reportconfig["Type"])) {
        $is_proton_thumbnail = True;
    }
    if (preg_match('/Composite/', $reportconfig["Type"])) {
        $is_proton_composite = True;
    }
}


$sigproc_results    = "sigproc_results/";
$basecaller_results = "basecaller_results/";
$alignment_results  = "./";
$reporttitle = $resultsName;

//Hide the Library Summary when library is set to none
if ($ReferenceName == "Reference:none") {
    $tf_only = true;
} else {
    $tf_only = false;
}
//Check to see if the page should create a PDF
if (isset($_GET["do_print"])){

       $do_print = $_GET["do_print"];
       //get the url for this page without the querystring
       $page_url = "http://localhost". parse_url($_SERVER['REQUEST_URI'],PHP_URL_PATH) . "?no_header=True";

       //build the command to be ran, wkhtmltopdf has to be in the correct path
       $pdf_string = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q --javascript-delay 1000 --margin-top 15 --header-spacing 5 --header-left " '. $resultsName .' - [date]" --header-right "Page [page] of [toPage]" --header-font-size 9 --disable-internal-links --disable-external-links ' . $page_url . ' report.pdf';
       //run the command
       exec($pdf_string);

       //now deliver the report.pdf to as REPORT_NAME.pdf
       $the_path = $_SERVER["PHP_SELF"];
       $test_name = basename(dirname($the_path));
       $file_name = $test_name . ".pdf";

       header("Content-type: application/pdf");
       header("Content-Disposition: attachment; filename=". $file_name);

       readfile("report.pdf");
       exit(0);

}
//replace the title
$header = str_replace("<title>", "<title>" . $reporttitle . ", ", $header);

print $header;
?>

<div id='inner'>

<?php
    print '<h1>Report for ' . $reporttitle . '</h1>';
/*Check to see if there are any unfinished parts of the Analysis.
if the analysis is not fully complete present the user with a
verbose explination of what remains to be done */

$progress = parseProgress('progress.txt');

//TODO: Progress needs to update the status message, not just delete the parts that are done
if ($progress) {
    echo <<<START_WARNING
<script>
function fancy_load() {
    $.getJSON("parsefiles.php?progress", function (json) {
        var all_done = true;
        jQuery.each(json, function (i, val) {
            if (!val) {
                $("#" + i).fadeOut();
            };
            if (val) {
                all_done = false;
            };
        });
        if (all_done) {
            $("#warn_icon").hide();
            $("#warning_box_head").html("<br/>The report is done.  The page will automatically reload in 5 seconds. <br/><br/>");
            setTimeout("window.location.reload(true);", 5000);
        };
    });
};
</script>

<div id="progress_box" class="ui-widget">
    <div class="ui-state-highlight ui-corner-all warning_box" id="yellow_box">
        <span class="warning_icon" id="warn_icon"></span>

        <div class="warning_inner">
            <span class="warning_head" id="warning_box_head">Report Generation In
            Progress <img id="dna_spin" style=
            "float:right; right : 500px; position: relative;" src=
            "/site_media/images/dna-small-slow-yellow.gif" /></span>
        </div>
    </div>
</div>

START_WARNING;

    foreach ($progress as $task) {
        print "<ul>";
        $under_name = str_replace(' ', '_', $task[0]);
        print "<li id='$under_name'>" . $task[0] . "  : " . $task[1] . "</li>";
        print "</ul>";
    }

    echo <<<END_WARNING
                                        </div>
                                </div>
                        </div>
END_WARNING;
}

//Start Library Summary
print '<div id="LibrarySummary" class="report_block">';

print '<h2 id="libsummary">Library Summary</h2>';

print '<div><!-- start wrapper -->';

print '<div id="alignmentSummary">';

//predicted perbase quality scores  only with reported -->
print '<h3>Based on Predicted Per-Base Quality Scores - Independent of Alignment</h3>';

//DRY is violated here.
$json   = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
$layout = json_decode($json, true);

// $isb = parse_to_keys($layout["Quality Summary"]["file_path"]);

if (file_exists($basecaller_results . "datasets_basecaller.json")) {

    //$json = file_get_contents($basecaller_results . 'datasets_basecaller.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
    //$dsb = json_decode($json, true);

}

//if (file_exists("ionstats_alignment.json")) {
//
//    $json   = file_get_contents('ionstats_alignment.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
//    $ionstats = json_decode($json, true);
//
//} else
if (file_exists($basecaller_results . "ionstats_basecaller.json")) {

    $json   = file_get_contents($basecaller_results . 'ionstats_basecaller.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
    $ionstats = json_decode($json, true);

}

if (isset($ionstats)) {
    print '<table id="q_alignment" class="heading">';
    print "<col width='325px' />";
    print "<col width='520px' />";
    print "<thead>";
    print "</thead>";
    print '<tbody>';

    print "<tr>";
    print "<th>Total Number of Bases [Mbp]</th>";
    print "<td>" . try_number_format($ionstats['full']['num_bases'] / 1000000, 2) . "</td>";
    print "</tr>";
    //$total_bases = 0;
    //foreach ($dsb['read_groups'] as &$value) {
    //    $total_bases += $value['total_bases'];
    //}
    //print "<tr>";
    //print "<th>Total Number of Bases [Mbp]</th>";
    //print "<td>" . try_number_format($total_bases / 1000000, 2) . "</td>";
    //print "</tr>";



    print "<tr>";
    print "<th class='subhead'>&#8227; Number of Q20 Bases [Mbp]</th>";
    print "<td>" . try_number_format(array_sum(array_slice($ionstats['qv_histogram'],20)) / 1000000, 2) . "</td>";
    print "</tr>";
    //$sumq20 = 0;
    //foreach ($dsb['read_groups'] as &$value) {
    //    $sumq20 += $value['Q20_bases'];
    //}
    //print "<tr>";
    //print "<th class='subhead'>&#8227; sum datasets_basecaller readgroups... </th>";
    //print "<td>" . try_number_format($sumq20 / 1000000, 2) . "</td>";
    //print "</tr>";
    //print "<tr>";
    //print "<th class='subhead'>&#8227; Number of Q20 Bases from ionstats_basecaller.json mimics AQ20 - don't use[Mbp]</th>";
    //print "<td>" . try_number_format($ionstats['Q20']['num_bases'] / 1000000, 2) . "</td>";
    //print "</tr>";


    print "<tr>";
    print "<th>Total Number of Reads</th>";
    print "<td>" . try_number_format($ionstats['full']['num_reads']) . "</td>";
    print "</tr>";
    //$num_reads = 0;
    //foreach ($dsb['read_groups'] as &$value) {
    //    $num_reads += $value['read_count'];
    //}
    //print "<tr>";
    //print "<th>Total Number of Reads</th>";
    //print "<td>" . try_number_format($num_reads) . "</td>";
    //print "</tr>";

    print "<tr>";
    print "<th>Mean Length [bp]</th>";
    print "<td>" . try_number_format($ionstats['full']['mean_read_length']) . "</td>";
    print "</tr>";

    print "<tr>";
    print "<th>Longest Read [bp]</th>";
    print "<td>" . try_number_format($ionstats['full']['max_read_length']) . "</td>";
    print "</tr>";

    print "</tbody></table>";

} else {
    print '<div class="not_found">No predicted per-base quality scores found.</div>';
}

print '<div id="AlignHistograms" >';
print '<table class="image">';
print '<tbody>';
print '<tr >';

$readlenhistogram = $basecaller_results . 'readLenHisto.png';
if (file_exists($readlenhistogram)) {
    print '<td class="image"><a class="box" href="' . $readlenhistogram . '"><img src="' . $readlenhistogram . '" width="450" height="225" border="0"/></a></td>';
}

if (file_exists('iontrace_Library.png')) {
    print "<td class='image'><a class='box' href='iontrace_Library.png'><img src='iontrace_Library.png' width='450' height='225' border='0'/></a></td>";
}
print "</tr>";
print '</tbody>';
print '</table>';
print '</div>';

print '<div style="clear: both;"></div>';

print '<h3>Reference Genome Information</h3>';

if ($meta['Reference']) {

    $reference_info = parseReferenceInfo("/results/referenceLibrary/tmap-f3/" . $meta['Reference'] . "/" . $meta['Reference'] . ".info.txt");

    print "<table class='noheading'>";
    print "<col width='325px' />";
    print "<col width='520px' />";

    print "<tr>";
    print "<th>Genome Name</th>";
    print "<td class='italic'>" . $reference_info['genome_name'] . "</td>";
    print "</tr>";

    print "<tr>";
    print "<th>Genome Size</th>";
    print "<td>" . $reference_info['genome_length'] . "</td>";
    print "</tr>";

    print "<tr>";
    print "<th>Genome Version</th>";
    print "<td>" . $reference_info['genome_version'] . "</td>";
    print "</tr>";

    print "<tr>";
    print "<th>Index Version</th>";
    print "<td>" . $reference_info['index_version'] . "</td>";
    print "</tr>";

    print "</table>";
    print "<br/>";
}

//If there is alignment info, print it
$json   = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
$layout = json_decode($json, true);

print '<h3>Based on Full Library Alignment to Provided Reference</h3>';

    if (file_exists('ionstats_alignment.json')) {
        //$json   = file_get_contents($layout["Alignment Summary"]["file_path"], FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
        $json   = file_get_contents('ionstats_alignment.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
        $isa = json_decode($json, true);

        if ($isa) {
            print '<table id="alignment" class="heading">';
            if ($display_q17) {
                print "<col width='325px' />";
                print "<col width='173px' />";
                print "<col width='173px' />";
                print "<col width='173px' />";
            } else {
                print "<col width='325px' />";
                print "<col width='260px' />";
                print "<col width='260px' />";
            }

            print "<thead><tr><th> </th>";

            //header info here
            if ($display_q17) {
               print "<th>AQ17</th>";
            }
            print "<th>AQ20</th>";
            print "<th>Perfect</th>";

            print "</tr></thead><tbody>";

            print "<tr>";
            print "<th>Total Number of Bases [Mbp]</th>";
            //Convert bases to megabases
            if ($display_q17) {
               print "<td>" . try_number_format($isa['AQ17']['num_bases'] / 1000000, 2) . "</td>";
            }
            print "<td>" . try_number_format($isa['AQ20']['num_bases'] / 1000000, 2) . "</td>";
            print "<td>" . try_number_format($isa['AQ47']['num_bases'] / 1000000, 2) . "</td>";
            print "</tr>";

            print "<tr>";
            print "<th>Mean Length [bp]</th>";
            if ($display_q17) {
               print "<td>" . try_number_format($isa['AQ17']['mean_read_length']) . "</td>";
            }
            print "<td>" . try_number_format($isa['AQ20']['mean_read_length']) . "</td>";
            print "<td>" . try_number_format($isa['AQ47']['mean_read_length']) . "</td>";
            print "</tr>";

            print "<tr>";
            print "<th>Longest Alignment [bp]</th>";
            if ($display_q17) {
               print "<td>" . try_number_format($isa['AQ17']['max_read_length']) . "</td>";
            }
            print "<td>" . try_number_format($isa['AQ20']['max_read_length']) . "</td>";
            print "<td>" . try_number_format($isa['AQ47']['max_read_length']) . "</td>";
            print "</tr>";
            if ($reference_info) {

                print "<tr>";
                print "<th>Mean Coverage Depth</th>";
                if ($display_q17) {
                    print "<td>" . try_number_format($isa['AQ17']['num_bases'] / $reference_info['genome_length'], 2) . "&times;</td>";
                }
                print "<td>" . try_number_format($isa['AQ20']['num_bases'] / $reference_info['genome_length'], 2) . "&times;</td>";
                print "<td>" . try_number_format($isa['AQ47']['num_bases'] / $reference_info['genome_length'], 2) . "&times;</td>";
                print "</tr>";

                print "<tr>";
                print "<th>Percentage of Library Covered</th>";
                if ($display_q17) {
                   print "<td> N/A" . "%</td>";
                }
                print "<td> N/A" . "%</td>";
                print "<td> N/A" . "%</td>";
                print "</tr>";
            }
            print "</tbody>";
            print "</table>";
        }
    }



print '</div>';
print '<div style="clear: both;"></div>';

print '</div><!-- end wrapper -->';
print '</div>';
//End Library Summary





//barcode block
function graphCreate($title, $pk, $lookupString)
{
    print '<h3>';
    print $title;
    print '</h3>';

    print '<div style="padding-top: 10px; padding-bottom: 10px; margin-left: -5px margin-top: 0px">';
    print '<iframe src="/rundb/graphiframe/' . $pk . '/?metric=' . $lookupString . '"';
    print 'width="101%" height="300" marginwidth="0" marginheight="0" align="top" scrolling="No" frameBorder="0" hspace="0" vspace="0"></iframe>';
    print '</div>';
}

if (file_exists("alignment_barcode_summary.csv") and $istoplevel) {
    $barCodeSet = $meta["Barcode Set"];

    print '<div id="barCodes" class="report_block">';
    print '<h2>Barcode Reports</h2><div>';
    graphCreate("Total number of Reads", $djangoPK["ResultsPK"], "Total%20number%20of%20Reads");
    graphCreate("AQ 20 Bases", $djangoPK["ResultsPK"], "Filtered%20Mapped%20Bases%20in%20Q20%20Alignments");
    graphCreate("Mean AQ20 read length", $djangoPK["ResultsPK"], "Filtered%20Q20%20Mean%20Alignment%20Length");
    graphCreate("AQ20 Reads", $djangoPK["ResultsPK"], "Filtered%20Q20%20Alignments");
    print '</div></div>';

}

if ($istoplevel) {
    //Start TF Summary
    print "<div id='TFSummary' class='report_block'>";

    print '<h2 id="tf" >Test Fragment Report</h2>';
    print '<div>'; //start wrapper

    $tfJson = file_get_contents($basecaller_results . 'TFStats.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
    $tf     = json_decode($tfJson, true);

    $json   = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
    $layout = json_decode($json, true);

    $tfKeys = array();
    if ($tf) {
        $tfKeys = array_keys($tf);
        if ($tfKeys)
            sort($tfKeys);
    }

    //print the tf summary table
    if (count($tfKeys) > 0) {
        print "<h3>Test Fragment Summary</h3>";

        print "<div class='half_width'>";
        print "<table class='heading half_width'>";
        print "<col width='175px' />";
        print "<col width='250px' />";

        print "<tr><th><strong>Test Fragment</strong></th><th><strong>Percent (50AQ17 / Num) </stong> </th></tr>";


        foreach ($tfKeys as $TF) {
            print "<tr>";
            print "<th>";
            print $TF;
            print "</th>";
            print "<td>";
            $num     = $tf[$TF]['Num'];
            $i50aq17 = $tf[$TF]['50Q17'];

            print try_number_format(($i50aq17 / $num), 2) * 100;
            print "%";

            print "</td>";
            print "</tr>";
        }
        print "</table>";
        print "</div>";

        print "<a class='box' style='padding-left: 80px;' href='iontrace_Test_Fragment.png'><img src='iontrace_Test_Fragment.png' width='450' height='225' border='0'/></a>";
        print "<div style='clear:both;'></div>";


        foreach ($tfKeys as $TF) {
            print "<h3>Test Fragment - $TF</h3>";
            print "<h4>Quality Metrics</h4>";
            print "<div id=$TF>";

            print "<table class='noheading'>";

            print "<col width='325px' />";
            print "<col width='520px' />";

            foreach ($layout['Quality Metrics']['metrics'] as $metrics) {
                print "<tr>";
                print "<th style='white-space: nowrap;'>$metrics[0]</th>";
                print "<td>";
                print wordwrap(try_number_format($tf[$TF][$metrics[1]]), 50, "\n", true);
                print "</td>";
                print "</tr>";
            }
            print "</table>";
            print "</div>";

            print "<div style='clear:both;'></div>";

            print "<br\>";

            print "<h4>Graphs</h4>";
            print "<div id='alignment' style='float:right'>";
            print "<a class='box' href='new_Q17_$TF.png'><img src='new_Q17_$TF.png' width='900' height='150' border='0'/></a>";
            print "</div>";
            print "<div style='clear:both;'></div>";
        }
    } else {
        print '<div class="not_found">No Test Fragments found.</div>';
    }

    print '</div>'; //end wrapper
    print '</div>';
    //End TF Summary Div
}

//start ion sphere

print '<div id="IonSphere" class="report_block">';

print '<h2>Ion Sphere&trade; Particle (ISP) Identification Summary</h2>';
print '<div>';

if ($is_proton_composite) {
    print '<table id="ispMap" border="0" cellpadding="0" cellspacing="0">';

    foreach (array_reverse($dataY) as $dy) {
        print "<tr class='slimRow'>";
        foreach ($dataX as $dx) {
            $needle      = 'block_X' . $dx . '_Y' . $dy;
            $needleImage = $needle . "/Bead_density_raw.png";
            print '<td class="slim">';
            print '<a href="' . $needle . '/Default_Report.php">';
            if (file_exists($needle . '/badblock.txt')) {
                if (file_exists($needle . '/sigproc_results/analysis_return_code.txt')) {
                    $anastatusfile = file_get_contents($needle . "/sigproc_results/analysis_return_code.txt");
                    if (!preg_match('/0/', $anastatusfile) and !preg_match('/3/', $anastatusfile)) {
                        print 'SigProc('.$anastatusfile.')<br>';
                        continue;
                    } else {
                        print 'error<br>';
                        continue;
                    }
                } else {
                    print 'error<br>';
                }
            } else if (file_exists($needle)) {
                if (file_exists($needle . '/blockstatus.txt')) {
                    $blockstatusfile = file_get_contents($needle . "/blockstatus.txt");
                    if (preg_match('/1/', $blockstatusfile)) {
                        print 'error<br>';
                        continue;
                    }
                }
                if (file_exists($needle . '/progress.txt')) {
                    $progress = parseProgress($needle . '/progress.txt');
                    if (count($progress) > 0) {
                        switch ($progress[0][0]) {
                            case "Well Characterization":
                                echo "Beadfind";
                                break;
                            case "Signal Processing":
                                echo "SigProc";
                                break;
                            case "Basecalling":
                                echo "BaseCall";
                                break;
                            case "Aligning Reads":
                                echo "Align";
                                break;
                            default:
                                print '<img src="/site_media/images/dna-small-slow-yellow.gif" class="tileSize" alt="' . $needle . '" border="0"/>';
                        }
                    } else {
                        if (file_exists($needleImage) ) {
                            print '<img src="' . $needleImage . '" class="tileSize" alt="' . $needle . '" border="0"/>';
                        } else {
                            print 'error';
                        }
                    }
                } else {
                    print '<img src="/site_media/images/dna-small-slow-yellow.gif" class="tileSize" alt="' . $needle . '" border="0"/>';
                }
            } else {
                print 'X' . $dx;
                print 'Y' . $dy;
            }
            print '</a></td>';
        }
        print "</tr>";
    }
    print "</table>";
}



function printISPTableEntry($category, $tip1, $tip2, $count, $total, $indent, $strong)
{
    print "<tr>";
    print "<th class='subhead'>";
    print "<span class='tip' title='" . $tip1 . "'>";
    print $indent . "<span class='tippy'>";
    if ($strong)
        print "<strong>";
    print $category;
    if ($strong)
        print "</strong>";
    print "</span>";
    print "</span>";
    print "</th>";

    print "<td class='right_td'>";
    if ($strong)
        print "<strong>";
    print try_number_format($count);
    print "</td>";

    print "<td class='right_td'>";
    print "<span class='tip_r' title='" . $tip2 . "'><span class='tippy'> ";
    if ($strong)
        print "<strong>";
    print format_percent($count / $total, $count);
    if ($strong)
        print "</strong>";
    print "</span></span></td>";

    print "</tr>";
}

    $bflist  = array(
        "default" => $sigproc_results . "analysis.bfmask.stats"
    );
    if (!file_exists($bflist['default'])) {
        $bflist['default'] = $sigproc_results . "bfmask.stats";
    }
    $bftlist = array(
        "default" => $basecaller_results . "BaseCaller.json"
    );
    $pnglist = array(
        "default" => "Bead_density_contour.png"
    );

foreach ($bflist as $bfName => $bfFile) {
    print '<div style="overflow: hidden;">'; //start wrapper
    print '<div class="half_width">';
    print '<div class="beadTable">';

    if (file_exists($bfFile)) {
        if (count($bflist) > 1) {
            print '<br>' . $bfName . ' Run';
        }

        $bf = parse_to_keys($bfFile);

        print '<table class="noheading half_width">';
        print '<col width="265px">';
        print '<col width="105px">';
        print '<col width="105px">';

        //$json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
        //$layout = json_decode($json,true);
        //peak signal
        // $bf = parse_to_keys($layout['Beadfind']['file_path']);

        //unrolling the loop so that I have greater control over what is displayed
        if ($bf) {
            print "<tr> <th> </th>    <th class='right_td'>Count</th>";
            print "<th class='right_td'>Percentage</th> </tr> ";

            //Total Addressable Wells
            print "<tr>";
            print "<th>";
            print "Total Addressable Wells";
            print "</th>";

            print "<td class='right_td'>";
            //addressable wells = total wells - excluded wells
            $total_addressable_wells = $bf["Total Wells"] - $bf["Excluded Wells"];
            print try_number_format($total_addressable_wells);
            print "</td>";

            print "<td class='right_td'>";
            print "";
            print "</td>";

            print "</tr>";

            //Wells with Ion Sphere Particles
            print "<tr>";
            print "<th class='subhead'>";
            print "<span class='tip' title='Percent of addressable wells that are loaded'>";
            print "&#8227; <span class='tippy'> Wells with ISPs </span>";
            print "</span>";
            print "</th>";

            print "<td class='right_td'>";
            $wells_with_isp = $bf["Bead Wells"];
            print try_number_format($wells_with_isp);
            print "</td>";

            print "<td class='right_td'>";
            print "<span class='tip_r' title='Wells with ISPs / Total Addressable Wells'><span class='tippy'> ";
            print format_percent($wells_with_isp / $total_addressable_wells, $wells_with_isp);
            print "</span></span></td>";

            print "</tr>";

            //live
            print "<tr>";
            print "<th class='subhead2'>";
            print "<span class='tip' title='Percent of loaded wells that are live'>";
            print "&#8227; <span class='tippy'>Live ISPs</span>";
            print "</span>";
            print "</th>";

            print "<td class='right_td'>";
            $live_wells = $bf["Live Beads"];
            print try_number_format($live_wells);
            print "</td>";

            print "<td class='right_td'>";
            print "<span class='tip_r' title='Live ISPs / Wells with ISPs'><span class='tippy'> ";
            print format_percent($live_wells / $wells_with_isp, $live_wells);
            print "</span></span></td>";

            print "</tr>";

            //tf
            print "<tr>";
            print "<th class='subhead3'>";
            print "<span class='tip' title='Percent of live wells that are TF'>";
            print "&#8227; <span class='tippy'>Test Fragment ISPs</span>";
            print "</span>";
            print "</th>";

            print "<td class='right_td'>";
            $tf_isp = $bf["Test Fragment Beads"];
            print try_number_format($tf_isp);
            print "</td>";

            print "<td class='right_td'>";
            print "<span class='tip_r' title='Test Fragment ISPs / Live ISPs'><span class='tippy'> ";
            print format_percent($tf_isp / $live_wells, $tf_isp);
            print "</span></span></td>";

            print "</tr>";

            //library
            print "<tr>";
            print "<th class='subhead3'>";
            print "<span class='tip' title='Percent of live wells that are library'>";
            print "&#8227; <span class='tippy'>Library ISPs</span>";
            print "</span>";
            print "</th>";

            print "<td class='right_td'>";
            $library_isp = $bf["Library Beads"];
            print try_number_format($library_isp);
            print "</td>";

            print "<td class='right_td'>";
            print "<span class='tip_r' title='Library ISPs/Live ISPs'><span class='tippy'> ";
            print format_percent($library_isp / ($live_wells), $library_isp);
            print "</span></span></td>";

            print "</tr>";

        }

        print "</table>";

        $bft_file = $bftlist["$bfName"];

        //now the second table for library ISP info
        $json = file_get_contents($bft_file, FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
        $bc   = json_decode($json, true);

        $ISP_polyclonal  = $bc["Filtering"]["LibraryReport"]["filtered_polyclonal"];
        $ISP_clipAdapter = $bc["Filtering"]["LibraryReport"]["filtered_primer_dimer"];
        $ISP_valid       = $bc["Filtering"]["LibraryReport"]["final_library_reads"];
        $ISP_lowQual     = $bc["Filtering"]["LibraryReport"]["filtered_low_quality"];

        print "<table class='noheading half_width'>";
        print '<col width="265px">';
        print '<col width="105px">';
        print '<col width="105px">';
        print "<tr> <th> </th>    <th class='right_td'>Count</th>";
        print "<th class='right_td'>Percentage</th> </tr> ";

        printISPTableEntry("Library ISPs / Percent Enrichment", "Percent of non-TF loaded wells that are library", "Library Enrichment = Library ISPs / (Wells with ISPs - Test Fragment ISPs)", $library_isp, $wells_with_isp - $tf_isp, "", false);
        printISPTableEntry("Filtered: Polyclonal", "More than one template per ISP", "Polyclonal / Library ISPs", $ISP_polyclonal, $library_isp, "&#8227; ", false);
        printISPTableEntry("Filtered: Primer dimer", "Insert length less than 8 bases", "Primer dimer / Library ISPs", $ISP_clipAdapter, $library_isp, "&#8227; ", false);
        printISPTableEntry("Filtered: Low quality", "Low quality", "Low quality / Library ISPs", $ISP_lowQual, $library_isp, "&#8227; ", false);
        printISPTableEntry("Final Library Reads", "Reads passing all filters", "Final Library Reads / Library ISPs", $ISP_valid, $library_isp, "&#8227; ", true);

        print "</table>";

        print "</table>";

    } //if

    print '</div>'; //beadtable
    print '</div>'; //<!-- end half width -->

    $Bead_density_contour_file = $pnglist["$bfName"];
    //                                        print $Bead_density_contour_file;

    if (file_exists($Bead_density_contour_file)) {
        print '<div class="on_right">';
        print '<a class="box" href="' . $Bead_density_contour_file . '">';
        print '<img src="' . $Bead_density_contour_file . '" width="400" height="300" border="0"/>';
        print '</a>';
        print '</div>';
    }
    print '</div>'; //end wrapper
} //end foreach

print '</div>';
print '</div>';

//<!-- stop ion sphere -->


//<!--  Start Report Information -->
print '<div id="ReportInfo" class="report_block">';


print '<h2>Report Information</h2>';

print '<div>'; // report information

print '<div>'; //<!-- start wrapper -->
print '<h3>Analysis Info</h3>';
print '<div id="AnalysisInfo">';

    print '<table class="noheading">';
    print '<col width="325px" />';
    print '<col width="520px" />';
    $analysis_info_blacklist = array(
        "Run Cycles" => "",
        "Analysis Flows" => "",
    );
    if ($meta) {
        $meta = array_diff_key($meta, $analysis_info_blacklist);
        foreach ($meta AS $key => $value) {
            print "<tr><th>$key</th> <td>$value</td></tr>";
        }
    }

    print '</table>';

print '</div>';
print '</div>'; //<!--  end wrapper -->

print '<h3>Software Version</h3>';
print '<div id="SoftwareVersion">';
$versiontxt           = parseVersion("version.txt");
$versiontxt_blacklist = array(
    "ion-onetouchupdater" => "",
    "ion-pgmupdates" => "",
    "ion-publishers" => "",
    "ion-referencelibrary" => "",
    "ion-sampledata" => "",
    "ion-docs" => "",
    "ion-tsconfig" => "",
    "ion-rsmts" => "",
    "ion-usbmount" => ""
);
$explogtxt_whitelist  = array(
    "Script" => "",
    "LiveView" => "",
    "Datacollect" => "",
    "OIA" => "",
    "OS" => "",
    "Graphics" => ""
);
if ($versiontxt):
// This will get the fist line read from versions.txt
// which we will bold and use as the header row below.
    list($name, $version) = each($versiontxt);
    unset($versiontxt[$name]);
    $versionstxt = array_diff_key($versiontxt, $versiontxt_blacklist);
    $explog_dict = array_intersect_key($explog_dict, $explogtxt_whitelist);
    $versions    = array_merge($versionstxt, $explog_dict);
    // This sorts the software => version list alphabetically.
    ksort($versions);
    print '<table class="noheading">';
    print '<col width="325px" />';
    print '<col width="520px" />';

    print '<tr>';
    print '<th>' . $name . '</th>';
    print '<td><strong>' . $version . '</strong></td>';
    print '</tr>';
    foreach ($versions as $name => $version):
        print '<tr>';
        print '<th>' . $name . '</th>';
        print '<td>' . $version;
        print '</td>';
        print '</tr>';
    endforeach;
    print '</table>';
endif;

print '</div>';
print '</div>'; // report information

print '</div>';
//<!-- stop report info -->


//<!-- start of file -->

print '<div id="FileLinks" class="report_block"><h2>File Links</h2><div><table class="noheading">';

if ($istoplevel) {
        $bamfile              = $alignment_results . $base_name . '.bam';
        $baifile              = $alignment_results . $base_name . '.bam.bai';
} else {
    //block case
    $bamfile              = 'rawlib.bam';
    $baifile              = 'rawlib.bam.bai';
}

if (file_exists("drmaa_stdout_block.txt")) {
    print "<tr><td><a href='drmaa_stdout_block.txt'>Developer link: Block stdout</a> </td> </tr>";
}
if (file_exists("drmaa_stderr_block.txt")) {
    print "<tr><td><a href='drmaa_stderr_block.txt'>Developer link: Block stderr</a> </td> </tr>";
}
if (file_exists("drmaa_stdout.txt")) {
    print "<tr><td><a href='drmaa_stdout.txt'>Developer link: std_out_err</a> </td> </tr>";
}
if (file_exists("sigproc_results/sigproc.log")) {
    print "<tr><td><a href='sigproc_results/sigproc.log'>Developer link: sigproc.log</a> </td> </tr>";
}

if ($istoplevel) {
    $url_1 = "/". parse_url($_SERVER['REQUEST_URI'],PHP_URL_PATH);
    $url_split = explode('_', $url_1);
    $url_split = array_reverse($url_split);
    $pk = substr($url_split[0], 0, -1);
    if(substr($pk, 0, 1) == '0') {
        $pk = substr($pk, 1);
    }
    print "<tr><td><a href='../../../report/" . $pk . "'>Plugins</a> </td> </tr>";
}

if (file_exists("/opt/ion/.developerversion")) {
    if ($is_proton_composite) {
        if(file_exists("./sigproc_results/timing.txt")) {
            print "<tr><td><a href='sigproc_results/timing.txt'>Developer link: timing.txt</a> </td> </tr>";
        } else {
            print "<tr><td><a href='sigproc_results/timing.txt'>Developer link: timing.txt missing, no OIA?</a> </td> </tr>";
        }
    }
}
print "</table></div></div>";

?>
<script type="text/javascript">

function htmlEscape(str) {
    return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br\>');
}

//string endswith
String.prototype.endsWith = function (str) {
    return (this.match(str + "$") == str)
}

$(document).ready(function(){

//remove the old header
$(".topbar").remove()


<?php
//write the javascript var with php
print 'djangoPK = ' . $djangoPK["ResultsPK"] . ';';
print 'djangoURL = "/rundb/api/v1/results/' . $djangoPK["ResultsPK"] . '/pluginresults/";';


//Check to see if this is a mobile device
print 'djangoMobileURL = "/rundb/mobile/report/' . $djangoPK["ResultsPK"] . '";';

if (!isset($_GET["force_desktop"]) || $_GET["force_desktop"] != True) {
    print 'if (jQuery.browser.mobile) { window.location = djangoMobileURL; }';
}

?>
$(".heading tr:odd").addClass("zebra");
$(".noheading tr:odd").addClass("zebra");
//add tooltips
$().tipTipDefaults({
    delay: 0
});
$('.tip').tipTip({
    position: 'bottom'
});
$('.tip_r').tipTip({
    position: 'bottom'
});
$('h2').append('<a href="javascript:;" class="expandCollapseButton" title="Collapse Section"></a>');
$('.expandCollapseButton').toggle(function () {
    if ($(this).attr('title') == 'Collapse Section') {
        $(this).css('background-position', 'right top');
        $(this).attr('title', 'Expand Section');
    }
    else {
        $(this).css('background-position', 'left top');
        $(this).attr('title', 'Collapse Section');
    }
}, function () {
    if ($(this).attr('title') == 'Expand Section') {
        $(this).css('background-position', 'left top');
        $(this).attr('title', 'Collapse Section');
    }
    else {
        $(this).css('background-position', 'right top');
        $(this).attr('title', 'Expand Section');
    }
});
$('.expandCollapseButton').click(function (event) {
    $(event.target).parent().parent().toggleClass('small');
    $(event.target).parent().next().slideToggle();
});

<?php
if ($progress) {
    echo 'setInterval("fancy_load()", 5000 );';
    echo '$(".topbar, .darkline, #progress_box, h1").expose();';
}

//when to show and hide the tf
$hide_tf = true;

if (isset($_GET['no_header']) == True) {
    if ($_GET['no_header'] == True) {
        $hide_tf  = false;
        $is_print = true;
    } else {
        //most of the time this should be hidden -- set to true
        $hide_tf = true;
    }
}

$hide_lib = false;

if ($tf_only == true) {
    $hide_tf  = false;
    $hide_lib = true;
}

//keep the TF collapsed unless this is a print or a tf only run
if ($hide_tf) {
    print "$('#tf .expandCollapseButton').css('background-position','right top');";
    print "$('#tf .expandCollapseButton').attr('title','Expand Section');";
    print "$('#tf').parent().toggleClass('small');";
    print "$('#tf').next().toggle();";
}
//hide this, unless it is a print
if ($hide_lib && !$is_print) {
    print "$('#libsummary .expandCollapseButton').css('background-position','right top');";
    print "$('#libsummary .expandCollapseButton').attr('title','Expand Section');";
    print "$('#libsummary').parent().toggleClass('small');";
    print "$('#libsummary').next().toggle();";
    print "$('#LibrarySummary').css('page-break-inside', 'auto')";
}

?>

//start overlay
$(".box").colorbox({transition:"none", speed:0});
//do the row mouse hover
$(".noheading tbody tr, .heading tbody tr").live("mouseover", function(){
    $(this).addClass("table_hover");
});
$(".noheading tbody tr, .heading tbody tr").live("mouseout",function(){
    $(this).removeClass("table_hover");
});

<?php
//hide the header if no_header = True in the querystring
if (isset($_GET['no_header'])) {
    if ($_GET['no_header'] == True) {
        echo '$(".topbar").toggle();';
        echo '$(".darkline").toggle();';
        print "$('.expandCollapseButton').hide();";
    }
}
?>

var keys = [];
var konami = '38,38,40,40,37,39,37,39,66,65';

$(document).keydown(function (e) {
    keys.push(e.keyCode);
    if (keys.toString().indexOf(konami) >= 0) {
        $(".zebra").css("background-color", "#FF0080");
        keys = [];
    }
});
});
</script>

<?php
//now if there was a footer found, write it.
print $footer;
?>
