<?php
        //Include PHP parser
        include ("parsefiles.php");

        $djangoPK  =  parse_to_keys("primary.key");

        //Render the report using the look of the Django site
        $blank_report= file_get_contents('/opt/ion/iondb/templates/rundb/php_base.html');

        //simple template, will break the page in half where [PHP] is found
        $template = explode("[PHP]",$blank_report);
        $header = $template[0];
        $footer = $template[1];

    if(file_exists("expMeta.dat")){
        $meta = parseVersion("expMeta.dat");
        }
    else{
        $meta = parseVersion("../expMeta.dat");
    }
    $resultsName = $meta["Analysis Name"];
    $libName = $meta["Library"];
    $expName = $meta["Run Name"];
    $base_name = $expName . '_' . $resultsName;

    $sigproc_results = "./";
    $basecaller_results = "./";
    $alignment_results = "./";
    if (file_exists($alignment_results . "/fwd_alignment.summary")){
        $PairedEnd=True;
        $corrected_path = "./corrected/";
        $fwdmeta = parseVersion("fwd_expMeta.dat");
        $revmeta = parseVersion("rev_expMeta.dat");
        $reporttitle = $resultsName;
    }else{
        $PairedEnd=False;
        $reporttitle = $resultsName;
    }

        //Hide the Library Summary when library is set to none
        if ($libName == "Library:none"){
                $tf_only = true;
        }else{
                $tf_only = false;
        }

        //Check to see if the page should create a PDF
        if (isset($_GET["do_print"])){

                $do_print = $_GET["do_print"];
                //get the url for this page without the querystring
                $page_url = "http://localhost". parse_url($_SERVER['REQUEST_URI'],PHP_URL_PATH) . "?no_header=True";

                //build the command to be ran, wkhtmltopdf has to be in the correct path
                $pdf_string = '/opt/ion/iondb/bin/wkhtmltopdf-amd64 -q --margin-top 15 --header-spacing 5 --header-left " '. $resultsName .' - [date]" --header-right "Page [page] of [toPage]" --header-font-size 9 --disable-internal-links --disable-external-links ' . $page_url . ' report.pdf';
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
        $header = str_replace("<title>","<title>". $reporttitle . ", ", $header );

        print $header;

    $json = file_get_contents('ion_params_00.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
    $params = json_decode($json,true);
    $explog_dict = array();
    if(file_exists("explog.txt"))
    {
      $file = fopen("explog.txt", 'r') or die();
      //Output a line of the file until the end is reached
      $dataX = array();
      $dataY = array();
      while(($line = fgets($file)) !== false)
      {
        // explog_dict added by Brian Kennedy because this seemed like the
        // logical place to scoop stuff out of explog.txt; however, I am
        // unsure of the funtion of the code following this short block:
        $explog_dict_entry = preg_split("/: /", $line);
        if (count($explog_dict_entry) == 2) {
          $name = str_replace(" version", "", $explog_dict_entry[0]);
          $explog_dict[$name] = $explog_dict_entry[1];
        }
        // unsure of the funtion of the following lines of code:
        $blockline = preg_split('/BlockStatus:/', $line);
        $key=$blockline[0];
        if (!$key)
        {
          $value=$blockline[1];
          list($X, $Y) = str_getcsv($value, ",");
          $X = trim($X, "X");
          $Y = trim($Y, "Y");
          //exclude thumbnail directory
          if ($X=="-1") continue;
          if ($X=="thumbnail") continue;
          $dataX[] = $X;
          $dataY[] = $Y;
        }
      }
      fclose($file);
      $dataX=array_unique($dataX);
      $dataY=array_unique($dataY);
      sort($dataX);
      sort($dataY);
    }

?>

        <link rel="stylesheet" type="text/css" href="/site_media/jquery/colorbox/colorbox.css" media="screen" />
        <script type="text/javascript" src="/site_media/jquery/colorbox/jquery.colorbox-min.js"></script>
        <script type="text/javascript" src="/site_media/jquery/js/jquery.tools.min.js"></script>
        <script type="text/javascript" src="/site_media/jquery/js/jquery.activity-indicator-1.0.0.min.js"></script>

        <!-- style for the reports -->
        <link rel="stylesheet" type="text/css" href="/site_media/report.css" media="screen" />


<div id='inner'>


<?php
if ($PairedEnd){
   print '<h1>Paired End Report for ' . $reporttitle . '</h1>';
   print '<center>based on<br>';
   print '<h3><a href="fwd_folder/Default_Report.php">' . $fwdmeta["Analysis Name"] . '</a><br>';
   print '<a href="rev_folder/Default_Report.php">' . $revmeta["Analysis Name"] . '</a></h3>';
   print '</center><br>';
}
else {
   print '<h1>Report for ' . $reporttitle . '</h1>';
}
/*Check to see if there are any unfinished parts of the Analysis.
  if the analysis is not fully complete present the user with a
  verbose explination of what remains to be done */

$progress = parseProgress('progress.txt');

//TODO: Progress needs to update the status message, not just delete the parts that are done
if ($progress){
echo <<<START_WARNING
<script>
        function fancy_load(){
                $.getJSON("parsefiles.php?progress", function(json) {
                        var all_done = true;
                    jQuery.each(json, function(i, val) {
                        if (!val){
                          $("#" + i).fadeOut();
                    };
                        if (val){
                          all_done = false;
                    };
                    });
                    if (all_done){
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
                                         <span class="warning_head" id="warning_box_head"> Report Generation In Progress
                                         <img id="dna_spin" style="float:right; right : 500px; position: relative;" src="/site_media/images/dna-small-slow-yellow.gif"></img>
                                        </span>
START_WARNING;

foreach ($progress as $task){
        print "<ul>";
        $under_name = str_replace(' ','_',$task[0]);
        print "<li id='$under_name'>" . $task[0] . "  : " . $task[1] . "</li>" ;
        print "</ul>";
}

echo <<<END_WARNING
                                        </div>
                                </div>
                        </div>
END_WARNING;
}

//PE Tooltips
$toolTipArray = array("Union" => " The combined output from forward and <br/> reverse runs and the joined, corrected <br/> reads from the composite report",
                      "Corrected" => "The single reads that have been created from <br/> overlapping forward and reverse pairs",	
                      "Unpaired Forward" => "Forward reads that have no reverse partner",	
                      "Unpaired Reverse" => "Reverse reads that have no forward partner",	
                );

//Start Library Summary
print '<div id="LibrarySummary" class="report_block">';

        print '<h2 id="libsummary">Library Summary</h2>';

        print '<div><!-- start wrapper -->';

                print '<div id="alignmentSummary">';

                //predicted perbase quality scores  only with reported -->
                print '<h3>Based on Predicted Per-Base Quality Scores - Independent of Alignment</h3>';

                //DRY is violated here.
                $json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
                $layout = json_decode($json,true);

                if ($PairedEnd){
                    $union_as = parse_to_keys("quality.summary");
                    $fwd_as = parse_to_keys("fwd_quality.summary");                     $display_fwd_as = False;
                    $rev_as = parse_to_keys("rev_quality.summary");                     $rev_as = False;
                    $corr_as = parse_to_keys("corrected.quality.summary");
                    $paired_fwd_as = parse_to_keys("Paired_Fwd.quality.summary");       $display_paired_fwd_as = False;
                    $paired_rev_as = parse_to_keys("Paired_Rev.quality.summary");       $paired_rev_as = False;
                    $singleton_fwd_as = parse_to_keys("Singleton_Fwd.quality.summary");
                    $singleton_rev_as = parse_to_keys("Singleton_Rev.quality.summary");
                } else {
                    // $as = parse_to_keys($layout["Quality Summary"]["file_path"]);
                    $as = parse_to_keys($basecaller_results . "quality.summary");
                }

    if ($PairedEnd) {

        print '<table id="pairing_statistics" class="heading">';
        print "<col width='445px' />";
        print "<col width='400px' />";
        print "<thead><tr></tr></thead>";
        print "<tbody>";

        print "<tr>";
        print "<th><span class='tip' title='Pairing rate is defined as the percentage<br/> of forward reads that have a reverse read pair'>";
        	print "<span class='tippy'>Pairing Rate</span>";
        print "</span></th>";
        print "<td>" .  try_number_format( ($paired_fwd_as['Number of Reads at Q0']+$corr_as['Number of Reads at Q0'])/$fwd_as['Number of Reads at Q0'] * 100 ,2 )."  %</td>";
        print "</tr>";

        print "<tr>";
        print "<th>Number of reads present in FWD and REV (corrected)</th>";
        print "<td>" .  try_number_format( ($corr_as['Number of Reads at Q0'])/$fwd_as['Number of Reads at Q0'] * 100 ,2 )."  %</td>";
        print "</tr>";

        print "<tr>";
        print "<th>Number of reads present in FWD and REV (non corrected)</th>";
        print "<td>" .  try_number_format( $paired_fwd_as['Number of Reads at Q0']/$fwd_as['Number of Reads at Q0'] * 100 ,2 )."  %</td>";
        print "</tr>";

        print "<tr>";
        print "<th>Number of reads present in FWD but not in REV</th>";
        print "<td>" .  try_number_format( $singleton_fwd_as['Number of Reads at Q0']/$fwd_as['Number of Reads at Q0'] * 100 ,2 )."  %</td>";
        print "</tr>";

        print "</tbody></table>";

    }


                if ( file_exists($basecaller_results . "quality.summary") or file_exists("fwd_quality.summary") ) {

                    print '<table id="q_alignment" class="heading">';
                    if ($PairedEnd){
                        print "<col width='100px' />";
                        if ($union_as)         print "<col width='70px' />";
                        if ($display_fwd_as)   print "<col width='70px' />";
                        if ($rev_as)           print "<col width='70px' />";
                        if ($corr_as)          print "<col width='80px' />";
                        if ($display_paired_fwd_as) print "<col width='70px' />";
                        if ($paired_rev_as)    print "<col width='70px' />";
                        if ($singleton_fwd_as) print "<col width='70px' />";
                        if ($singleton_rev_as) print "<col width='70px' />";
                    } else {
                        print "<col width='325px' />";
                        print "<col width='520px' />";
                    }

                    if ($PairedEnd) {
                        print "<thead><tr>";

                        //header info here
                        print "<th></th>";
                        
                        
                        function tableHead($label){
                            global $toolTipArray;
                            print "<th><span class='tip' title='". $toolTipArray[$label] . "'>";
                                    print "<span class='tippy'>" . $label . "</span>";
                            print "</span></th>";
                        }
                        
                        if ($union_as)                 tableHead("Union");
                        if ($display_fwd_as)           tableHead("Forward");
                        if ($rev_as)                   tableHead("Reverse");
                        if ($corr_as)                  tableHead("Corrected");
                        if ($display_paired_fwd_as)    tableHead("Paired Forward");
                        if ($paired_rev_as)            tableHead("Paired Reverse");
                        if ($singleton_fwd_as)         tableHead("Unpaired Forward");
                        if ($singleton_rev_as)         tableHead("Unpaired Reverse");

                        print "</tr></thead>";
                    } else {
                        print "<thead>";
                        print "</thead>";
                    }


                    print '<tbody>';
                    print "<tr>";
                    print "<th>Total Number of Bases [Mbp]</th>";
                    if ($PairedEnd){
                                //print "<td>" . try_number_format( ($fwd_as['Number of Bases at Q0']+$rev_as['Number of Bases at Q0'])/1000000 , 2 ) ."</td>";
                                if ($union_as)         print "<td>" . try_number_format( $union_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                                if ($display_fwd_as)   print "<td>" . try_number_format( $fwd_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                                if ($rev_as)           print "<td>" . try_number_format( $rev_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                                if ($corr_as)          print "<td>" . try_number_format( $corr_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                                if ($display_paired_fwd_as)    print "<td>" . try_number_format( $paired_fwd_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                                if ($paired_rev_as)    print "<td>" . try_number_format( $paired_rev_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                                if ($singleton_fwd_as) print "<td>" . try_number_format( $singleton_fwd_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                                if ($singleton_rev_as) print "<td>" . try_number_format( $singleton_rev_as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                    } else {
                                print "<td>" . try_number_format( $as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
                    }
                    print "</tr>";


                                print "<tr>";
                                print "<th class='subhead'>&#8227; Number of Q20 Bases [Mbp]</th>";
                                if ($PairedEnd){
                                        //print "<td>" . try_number_format( ($fwd_as['Number of Bases at Q20']+$rev_as['Number of Bases at Q20'])/1000000 , 2 ) ."</td>";
                                        if ($union_as)         print "<td>" . try_number_format( $union_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                        if ($display_fwd_as)   print "<td>" . try_number_format( $fwd_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                        if ($rev_as)           print "<td>" . try_number_format( $rev_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                        if ($corr_as)          print "<td>" . try_number_format( $corr_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                        if ($display_paired_fwd_as)    print "<td>" . try_number_format( $paired_fwd_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                        if ($paired_rev_as)    print "<td>" . try_number_format( $paired_rev_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                        if ($singleton_fwd_as) print "<td>" . try_number_format( $singleton_fwd_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                        if ($singleton_rev_as) print "<td>" . try_number_format( $singleton_rev_as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                } else {
                                        print "<td>" . try_number_format( $as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
                                }
                                print "</tr>";

                                print "<tr>";
                                print "<th>Total Number of Reads</th>";
                                if ($PairedEnd){
                                        //print "<td>" . try_number_format( $fwd_as['Number of Reads at Q0']+$rev_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($union_as)         print "<td>" . try_number_format( $union_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($display_fwd_as)   print "<td>" . try_number_format( $fwd_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($rev_as)           print "<td>" . try_number_format( $rev_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($corr_as)          print "<td>" . try_number_format( $corr_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($display_paired_fwd_as)    print "<td>" . try_number_format( $paired_fwd_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($paired_rev_as)    print "<td>" . try_number_format( $paired_rev_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($singleton_fwd_as) print "<td>" . try_number_format( $singleton_fwd_as['Number of Reads at Q0'] ) ."</td>";
                                        if ($singleton_rev_as) print "<td>" . try_number_format( $singleton_rev_as['Number of Reads at Q0'] ) ."</td>";
                                } else {
                                        print "<td>" . try_number_format( $as['Number of Reads at Q0'] ) ."</td>";
                                }
                                print "</tr>";

                                print "<tr>";
                                print "<th>Mean Length [bp]</th>";
                                if ($PairedEnd){
                                        //print "<td>" . try_number_format(($fwd_as['Mean Read Length at Q0']*$fwd_as['Number of Reads at Q0']+$rev_as['Mean Read Length at Q0']*$rev_as['Number of Reads at Q0'])/($fwd_as['Number of Reads at Q0']+$rev_as['Number of Reads at Q0']) ) ."</td>";
                                        if ($union_as)     print "<td>" . try_number_format( $union_as['Mean Read Length at Q0']) ."</td>";
                                        if ($display_fwd_as)  print "<td>" . try_number_format( $fwd_as['Mean Read Length at Q0']) ."</td>";
                                        if ($rev_as)     print "<td>" . try_number_format( $rev_as['Mean Read Length at Q0']) ."</td>";
                                        if ($corr_as)     print "<td>" . try_number_format( $corr_as['Mean Read Length at Q0']) ."</td>";
                                        if ($display_paired_fwd_as)     print "<td>" . try_number_format( $paired_fwd_as['Mean Read Length at Q0'] ) ."</td>";
                                        if ($paired_rev_as)     print "<td>" . try_number_format( $paired_rev_as['Mean Read Length at Q0'] ) ."</td>";
                                        if ($singleton_fwd_as)     print "<td>" . try_number_format( $singleton_fwd_as['Mean Read Length at Q0'] ) ."</td>";
                                        if ($singleton_rev_as)     print "<td>" . try_number_format( $singleton_rev_as['Mean Read Length at Q0'] ) ."</td>";
                                } else {
                                        print "<td>" . try_number_format( $as['Mean Read Length at Q0'] ) ."</td>";
                                }
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Longest Read [bp]</th>";
                                if ($PairedEnd){
                                        //print "<td>" . try_number_format( max($fwd_as['Max Read Length at Q0'],$rev_as['Max Read Length at Q0']) ) ."</td>";
                                        if ($union_as)   print "<td>" . try_number_format( $union_as['Max Read Length at Q0'] ) ."</td>";
                                        if ($display_fwd_as)   print "<td>" . try_number_format( $fwd_as['Max Read Length at Q0'] ) ."</td>";
                                        if ($rev_as)   print "<td>" . try_number_format( $rev_as['Max Read Length at Q0'] ) ."</td>";
                                        if ($corr_as)   print "<td>" . try_number_format( $corr_as['Max Read Length at Q0'] ) ."</td>";
                                        if ($display_paired_fwd_as)   print "<td>" . try_number_format( $paired_fwd_as['Max Read Length at Q0'] ) ."</td>";
                                        if ($paired_rev_as)   print "<td>" . try_number_format( $paired_rev_as['Max Read Length at Q0'] ) ."</td>";
                                        if ($singleton_fwd_as)   print "<td>" . try_number_format( $singleton_fwd_as['Max Read Length at Q0'] ) ."</td>";
                                        if ($singleton_rev_as)   print "<td>" . try_number_format( $singleton_rev_as['Max Read Length at Q0'] ) ."</td>";
                                } else {
                                        print "<td>" . try_number_format( $as['Max Read Length at Q0'] ) ."</td>";
                                }
                                print "</tr></tbody></table>";

                        }
                        else{
                                print '<div class="not_found">No predicted per-base quality scores found.</div>';
                        }

                print '<div id="AlignHistograms" >';
                print '<table class="image">';
                        print '<tbody>';
                        print '<tr >';

                                if(file_exists('readLenHisto.png') )
                                {
                                        print '<td class="image"><a class="box" href="readLenHisto.png"><img src="readLenHisto.png" width="450" height="225" border="0"/></a></td>';
                                }

                                if(file_exists('iontrace_Library.png'))
                                {
                                        print "<td class='image'><a class='box' href='iontrace_Library.png'><img src='iontrace_Library.png' width='450' height='225' border='0'/></a></td>";
                                }
                        print "</tr>";
                        print '</tbody>';
                print '</table>';
                print '</div>';

                print '<div style="clear: both;"></div>';

                print '<h3>Reference Genome Information</h3>';

                        //If there is a Reference Genome print the info
                        $json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
                        $layout = json_decode($json,true);
                        $as = parse_to_keys($layout["Alignment Summary"]["file_path"]);
                        if($as){
                                print "<table class='noheading'>";
                                print "<col width='325px' />";
                                print "<col width='520px' />";
                                foreach(array_keys($layout['Alignment Summary']['pre_metrics']) as $label){
                                        print "<tr>";
                                        print "<th>$label</th>";
                                        if(count($layout['Alignment Summary']['pre_metrics'][$label]) > 1){
                                                $l = $layout['Alignment Summary']['pre_metrics'][$label][0];
                                                $units = $layout['Alignment Summary']['pre_metrics'][$label][1];
                                                print "<td>";
                                                print try_number_format($as[$l]);
                                                print " $units</td>";
                                        }
                                        else{
                                                $l = $layout['Alignment Summary']['pre_metrics'][$label];
                                                if ($label == "Genome Name"){
                                                        print "<td class='italic'>";
                                                }else{
                                                        print "<td>";
                                                }
                                                print "" . $as[$l] . "</td>";
                                        }
                                        print "</tr>";
                                }
                                print "</table>";
                                print "<br/>";
                        }elseif( file_exists("alignment.error")){
                                print '<div class="not_found">';
                                print '<p>There was an alignment error. For details see the <a href="log.html">Report Log</a></p>';
                                //print the contents of the alignment.error file
                                $alignerror = "alignment.error";
                                $fh = fopen($alignerror, 'r');
                                $alignErrorData= fread($fh, filesize($alignerror));
                                fclose($fh);
                                //a missing library will generage an error.  Print that here.
                                $no_ref = alignQC_noref();
                                if ($no_ref){
                                        print '<p>  Unable to process alignment for genome, because the <strong>'. $no_ref .'</strong> reference library was not found.</p>';
                                }
                                print '</div>';
                        }

        // paired end report
        if ($PairedEnd){
            $alignmentSummaryFilelist = array(
                  "Union"  => $alignment_results . "/alignment.summary",
                  //"Forward" => $alignment_results . "/fwd_alignment.summary",
                  //"Reverse" => $alignment_results . "/rev_alignment.summary",
                  "Corrected" => $alignment_results . "/corrected.alignment.summary",
                  //"Paired Forward"  => $alignment_results . "/Paired_Fwd.alignment.summary",
                  //"Paired Reverse"  => $alignment_results . "/Paired_Rev.alignment.summary",
                  "Unpaired Forward"  => $alignment_results . "/Singleton_Fwd.alignment.summary",
                  "Unpaired Reverse"  => $alignment_results . "/Singleton_Rev.alignment.summary",
            );
        }
        else{
            $alignmentSummaryFilelist = array("default" => $alignment_results . "/alignment.summary");
        }


                        //If there is alignment info, print it
                        $json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
                        $layout = json_decode($json,true);
                        $as = parse_to_keys($layout["Alignment Summary"]["file_path"]);
                        //once we have as we can check to see if it contains the full data, or the sampled data
                        if (isset($as['Total number of Sampled Reads'])) {
                                $align_full = false;
                                print '<h3>Based on Sampled Library Alignment to Provided Reference</h3>';
                        }else{
                                $align_full = true;
                                print '<h3>Based on Full Library Alignment to Provided Reference</h3>';
                        }


        //start alignment.summary
        foreach($alignmentSummaryFilelist as $alignTableName => $alignFile) {
            if (file_exists($alignFile)) {
                 $as = parse_to_keys($alignFile);
                        if (count($alignmentSummaryFilelist) > 1){
                            print "<span class='tip_r' title='". $toolTipArray[$alignTableName] ."'>";
                            	print "<span class='headerTip'>" . $alignTableName . "</span>";
                            print "</span>";
                        }


                        if($align_full == true && $as){
                                print '<table id="alignment" class="heading">';
                                if (! $PairedEnd){
                                    print "<col width='325px' />";
                                }
                                print "<col width='260px' />";
                                print "<col width='260px' />";

                                print "<thead><tr><th> </th>";

                                //header info here
                                print "<th>AQ20</th>";
                                print "<th>Perfect</th>";

                                print "</tr></thead><tbody>";

                                print "<tr>";
                                        print "<th>Total Number of Bases [Mbp]</th>";
                                        //Convert bases to megabases
                                        print "<td>" . try_number_format( $as['Filtered Mapped Bases in Q20 Alignments']/1000000 , 2 ) ."</td>";
                                        print "<td>" . try_number_format( $as['Filtered Mapped Bases in Q47 Alignments']/1000000 , 2 ) ."</td>";
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Mean Length [bp]</th>";
                                        print "<td>" . try_number_format( $as['Filtered Q20 Mean Alignment Length']) ."</td>";
                                        print "<td>" . try_number_format( $as['Filtered Q47 Mean Alignment Length']) ."</td>";
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Longest Alignment [bp]</th>";
                                        print "<td>" . try_number_format( $as['Filtered Q20 Longest Alignment']) ."</td>";
                                        print "<td>" . try_number_format( $as['Filtered Q47 Longest Alignment']) ."</td>";
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Mean Coverage Depth</th>";
                                        print "<td>" . try_number_format( $as['Filtered Q20 Mean Coverage Depth'], 2) ."&times;</td>";
                                        print "<td>" . try_number_format( $as['Filtered Q47 Mean Coverage Depth'], 2) ."&times;</td>";
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Percentage of Library Covered</th>";
                                        print "<td>" . try_number_format( $as['Filtered Q20 Coverage Percentage']) ."%</td>";
                                        print "<td>" . try_number_format( $as['Filtered Q47 Coverage Percentage']) ."%</td>";
                                print "</tr>";

                                print "</tbody>";
                                print "</table>";

                        }elseif ($align_full == false && $as){
                                print '<table id="alignment" class="heading">';
                                print "<col width='325px' />";

                                print "<col width='86px' />";
                                print "<col width='86px' />";
                                print "<col width='86px' />";
                                print "<col width='86px' />";
                                print "<col width='86px' />";
                                print "<col width='86px' />";


                                print "<thead><tr> <th class='empty'> </th>";
                                print "<th colspan=3 class='tiptop'>Random sample of " . try_number_format($as['Total number of Sampled Reads']) . " reads</th>";
                                print "<th colspan=3 class='tiptop'>Extrapolation to all " . try_number_format($as['Total number of Reads']) . " reads</th></tr>";

                                //header info here
                                print "<th> </th>";
                                print "<th>AQ20</th>";
                                print "<th>Perfect</th>";
                                print "<th>AQ20</th>";
                                print "<th>Perfect</th>";

                                print "</tr></thead><tbody>";

                                print "<tr>";
                                        print "<th>Total Number of Bases [Mbp]</th>";
                                        //Convert bases to megabases
                                        print "<td>" . try_number_format( $as['Sampled Filtered Mapped Bases in Q20 Alignments']/1000000 , 2 ) ."</td>";
                                        print "<td>" . try_number_format( $as['Sampled Filtered Mapped Bases in Q47 Alignments']/1000000 , 2 ) ."</td>";
                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Mapped Bases in Q20 Alignments']/1000000 , 2 ) ."</td>";
                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Mapped Bases in Q47 Alignments']/1000000 , 2 ) ."</td>";
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Mean Length [bp]</th>";
                                        print "<td>" . try_number_format( $as['Sampled Filtered Q20 Mean Alignment Length']) ."</td>";
                                        print "<td>" . try_number_format( $as['Sampled Filtered Q47 Mean Alignment Length']) ."</td>";
                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Mean Alignment Length']) ."</td>";
                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Mean Alignment Length']) ."</td>";
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Longest Alignment [bp]</th>";
                                        print "<td>" . try_number_format( $as['Sampled Filtered Q20 Longest Alignment']) ."</td>";
                                        print "<td>" . try_number_format( $as['Sampled Filtered Q47 Longest Alignment']) ."</td>";
                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Longest Alignment']) ."</td>";
                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Longest Alignment']) ."</td>";
                                print "</tr>";

                                print "<tr>";
                                        print "<th>Mean Coverage Depth</th>";


                                        print "<td>" . try_number_format( $as['Sampled Filtered Q20 Mean Coverage Depth'], 2);
                                        print (is_numeric($as['Sampled Filtered Q20 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;

                                        print "<td>" . try_number_format( $as['Sampled Filtered Q47 Mean Coverage Depth'], 2);
                                        print (is_numeric($as['Sampled Filtered Q47 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;


                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Mean Coverage Depth'],2 );
                                        print (is_numeric($as['Extrapolated Filtered Q20 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;

                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Mean Coverage Depth'],2 );
                                        print (is_numeric($as['Extrapolated Filtered Q47 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;
                                print "</tr>";

                                print "<tr>";
                                        //slightly special case because we only print % in the cell if the call has a number
                                        print "<th>Percentage of Library Covered</th>";


                                        print "<td>" . try_number_format( $as['Sampled Filtered Q20 Coverage Percentage']);
                                        print (is_numeric($as['Sampled Filtered Q20 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;

                                        print "<td>" . try_number_format( $as['Sampled Filtered Q47 Coverage Percentage']);
                                        print (is_numeric($as['Sampled Filtered Q47 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;

                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Coverage Percentage']);
                                        print (is_numeric($as['Extrapolated Filtered Q20 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;

                                        print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Coverage Percentage']);
                                        print (is_numeric($as['Extrapolated Filtered Q47 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;

                                print "</tr>";

                                print "</tbody>";
                                print "</table>";
            }
        }
    }
                print "<h3>Read Alignment Distribution</h3>";

        // paired end report
        if ($PairedEnd){
            $alignFilelist = array(
                  "Union" => $alignment_results . "/alignTable.txt",
                  //"Forward" => $alignment_results . "/fwd_alignTable.txt",
                  //"Reverse" => $alignment_results . "/rev_alignTable.txt",
                  "Corrected" => $alignment_results . "/corrected.alignTable.txt",
                  //"Paired Forward"  => $alignment_results . "/Paired_Fwd.alignTable.txt",
                  //"Paired Reverse"  => $alignment_results . "/Paired_Rev.alignTable.txt",
                  "Unpaired Forward"  => $alignment_results . "/Singleton_Fwd.alignTable.txt",
                  "Unpaired Reverse"  => $alignment_results . "/Singleton_Rev.alignTable.txt",
            );
        }
        else{
            $alignFilelist = array("default" => $alignment_results . "/alignTable.txt");
        }

        //start alignTable.txt
        foreach($alignFilelist as $alignTableName => $alignFile) {
            if (file_exists($alignFile)) {
                $alignTable = tabs_parse_to_keys($alignFile);
                if ($alignTable){
                        if (count($alignFilelist) > 1){
                            print "<span class='tip_r' title='". $toolTipArray[$alignTableName] ."'>";
                            	print "<span class='headerTip'>" . $alignTableName . "</span>";
                            print "</span>";
                        }

                        print '<table id="alignTable" class="heading">';
                        print "<col width='150px' />";
                        print "<col width='90px' />";
                        print "<col width='90px' />";
                        print "<col width='90px' />";
                        print "<col width='90px' />";
                        print "<col width='90px' />";
                        print "<col width='110px' />";
                        print "<col width='135px' />";

                        print "<tr>";

                        print "<th><span class='tip float_right' title='The number of bases in each read </br> considered for the row in the table'>";
                                print "<span class='tippy'>Read Length [bp]</span>";
                        print "</span></th>";

                        print "<th><span class='tip float_right' title='Number of reads with at least Read Length bases'>";
                                print "<span class='tippy'>Reads</span>";
                        print "</span></th>";

                        print "<th><span class='tip float_right' title='Number of reads that tmap could not map'>";
                                print "<span class='tippy'>Unmapped</span>";
                        print "</span></th>";

                        print "<th><span class='tip float_right' title='Number of reads mapped but not having 90% accuracy in first 50 bases'>";
                                print "<span class='tippy'>Excluded</span>";
                        print "</span></th>";

                        print "<th><span class='tip float_right' title='Number of reads mapped and with accuracy > 90% in first 50 bases, </br> but with aligned length less than the Read Length threshold'>";
                                print "<span class='tippy'>Clipped </span>";
                        print "</span></th>";

                        print "<th><span class='tip float_right' title='Number of aligned reads with 0 mismatches in the first Read Length bases'>";
                                print "<span class='tippy'>Perfect</span>";
                        print "</span></th>";

                        print "<th><span class='tip float_right' title='Number of aligned reads with one mismatch in the first Read Length bases'>";
                                print "<span class='tippy'>1 mismatch</span>";
                        print "</span></th>";

                        print "<th><span class='tip float_right' title='Number of aligned reads with two or more </br>  mismatches in the first Read Length bases'>";
                                print "<span class='tippy'>&ge;2 mismatches</span>";
                        print "</span></th>";

                        print "</tr>";

                        foreach ($alignTable as $key => $inarray) {
                            // if it is 0 it is a header
                                $alignTableCol = 0;
				if ($inarray[1] > 0 && $inarray[6] > 0 ){
				    print "<tr>";
				    foreach($inarray as $val) {
					    if ($alignTableCol == 0){
						    print "<th class='right_td'>";
						print try_number_format($val);
						    print "</th>";
					    }elseif ($alignTableCol != 5){
						    print "<td class='right_td'>";
						print try_number_format($val);
						    print "</td>";
					    }
					    $alignTableCol = $alignTableCol + 1;
				    }
				    print "</tr>";
				}
                        }
                        print "</table>";
                }
                else{
                        print 'no alignments found<br>';
                }
        }
        else{
            print $alignFile.' not found<br>';
        }
}
        //end alignTable.txt


        print '</div>';
        print '<div style="clear: both;"></div>';

        print '</div><!-- end wrapper -->';
print '</div>';
//End Library Summary





//barcode block
function graphCreate($title, $pk , $lookupString){
                        print '<h3>';
                        print $title;
                        print '</h3>';

                        print '<div style="padding-top: 10px; padding-bottom: 10px; margin-left: -5px margin-top: 0px">';
                        print '<iframe src="/rundb/graphiframe/' . $pk . '/?metric=' . $lookupString . '"' ;
                        print 'width="101%" height="300" marginwidth="0" marginheight="0" align="top" scrolling="No" frameBorder="0" hspace="0" vspace="0"></iframe>';
                        print '</div>';
}

if (file_exists("alignment_barcode_summary.csv")) {

                $barCodeSet = $meta["Barcode Set"];

                print '<div id="barCodes" class="report_block">';
                print   '<h2>Barcode Reports</h2><div>';
                        graphCreate("Total number of Reads",$djangoPK["ResultsPK"],"Total%20number%20of%20Reads");
                        graphCreate("AQ 20 Bases",$djangoPK["ResultsPK"],"Filtered%20Mapped%20Bases%20in%20Q20%20Alignments");
                        graphCreate("Mean AQ20 read length",$djangoPK["ResultsPK"],"Filtered%20Q20%20Mean%20Alignment%20Length");
                        graphCreate("AQ20 Reads",$djangoPK["ResultsPK"],"Filtered%20Q20%20Alignments");
                print '</div></div>';

        }


if (! $PairedEnd){
    //Start TF Summary
    print "<div id='TFSummary' class='report_block'>";

        print '<h2 id="tf" >Test Fragment Report</h2>';
        print '<div>'; //start wrapper

                        $tfJson = file_get_contents('TFStats.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
                        $tf = json_decode($tfJson,true);

                        $json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
                        $layout = json_decode($json,true);

                        $tfKeys = array();
                        if($tf)
                        {
                                $tfKeys = array_keys($tf);
                                if ($tfKeys) sort($tfKeys);
                        }

                        //print the tf summary table
                        if(count($tfKeys) > 0) {

                                print"<h3>Test Fragment Summary</h3>";

                                print "<div class='half_width'>";
                                print "<table class='heading half_width'>";
                                        print "<col width='175px' />";
                                        print "<col width='250px' />";

                                        print "<tr><th><strong>Test Fragment</strong></th><th><strong>Percent (50AQ17 / Num) </stong> </th></tr>";


                                foreach($tfKeys as $TF)
                                {
                                        print "<tr>";
                                        print "<th>";
                                        print $TF;
                                        print "</th>";
                                        print "<td>";
                                        $num = $tf[$TF]['Num'];
                                        $i50aq17 = $tf[$TF]['50Q17'];

                                        print try_number_format( ($i50aq17 / $num ) , 2) * 100 ;
                                        print "%";

                                        print "</td>";
                                        print "</tr>";
                                }
                                print "</table>";
                                print "</div>";

                                print"<a class='box' style='padding-left: 80px;' href='iontrace_Test_Fragment.png'><img src='iontrace_Test_Fragment.png' width='450' height='225' border='0'/></a>";
                                print"<div style='clear:both;'></div>";


                                foreach($tfKeys as $TF)
                                {
                                        print"<h3>Test Fragment - $TF</h3>";
                                        print"<h4>Quality Metrics</h4>";
                                        print"<div id=$TF>";

                                        print "<table class='noheading'>";

                                        print "<col width='325px' />";
                                        print "<col width='520px' />";

                                        foreach($layout['Quality Metrics']['metrics'] as $metrics)
                                        {
                                                print "<tr>";
                                                print "<th style='white-space: nowrap;'>$metrics[0]</th>";
                                                print "<td>";
                                                print wordwrap( try_number_format($tf[$TF][$metrics[1]]),50,"\n",true);
                                                print "</td>";
                                                print "</tr>";
                                        }
                                        print "</table>";
                                        print"</div>";

                                        print"<div style='clear:both;'></div>";

                                        print"<br\>";

                                        print"<h4>Graphs</h4>";
                                        print"<div id='alignment' style='float:right'>";
                                        print"<a class='box' href='Q17_$TF.png'><img src='Q17_$TF.png' width='450' height='225' border='0'/></a>";
                                        print"<a class='box' href='Average Corrected Ionogram_$TF.png'><img src='Average Corrected Ionogram_$TF.png' width='450' height='225' border='0'/></a>";
                                        print"</div>";
                                        print"<div style='clear:both;'></div>";
                                }
                        }else{
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

        if (file_exists("explog.txt") and count($dataX) > 0) {
        print '<table id="ispMap" border="0" cellpadding="0" cellspacing="0">';

        foreach (array_reverse($dataY) as $dy) {
        print "<tr class='slimRow'>";
                foreach($dataX as $dx) {
                        $needle = 'block_X'.$dx.'_Y'.$dy;
                        $needleImage = $needle . "/Bead_density_raw.png";
                        print '<td class="slim">';
                        print '<a href="'.$needle.'/Default_Report.php">';
                        if (file_exists($needle . '/badblock.txt')) {
                            print 'error<br>';
                        }else if (file_exists($needleImage)){
                            print '<img src="'.$needleImage.'" class="tileSize" alt="'.$needle.'" border="0"/>';
                        }else if (file_exists($needle)) {
                            if (file_exists($needle . '/progress.txt')) {
                                $progress = parseProgress($needle . '/progress.txt');
                                switch ($progress[0][0]) {
                                case "Well Characterization":
                                    echo "Well Ch";
                                    break;
                                case "Signal Processing":
                                    echo "Signal Pr";
                                    break;
                                case "Basecalling":
                                    echo "Baseca";
                                    break;
                                case "Creating Fastq":
                                    echo "Crea Fastq";
                                    break;
                                case "Aligning Reads":
                                    echo "Alig Reads";
                                    break;
                                default:
                                    print '<img src="/site_media/images/dna-small-slow-yellow.gif" class="tileSize" alt="'.$needle.'" border="0"/>';
                                }
                            } else {
                                print '<img src="/site_media/images/dna-small-slow-yellow.gif" class="tileSize" alt="'.$needle.'" border="0"/>';
                            }
                        }else{
                                print 'X'.$dx.'<br>';
                                print 'Y'.$dy;
                        }
                        print '</a></td>';
                }
                print "</tr>";
        }
        print "</table>";
        }



                                        function printISPTableEntry($category, $tip1, $tip2, $count, $total, $indent, $strong){
                                                print "<tr>";
                                                                print "<th class='subhead'>";
                                                                        print "<span class='tip' title='" . $tip1 . "'>";
                                                                                print $indent . "<span class='tippy'>";
                                                                                if($strong)
                                                                                        print "<strong>";
                                                                                print $category;
                                                                                if($strong)
                                                                                        print "</strong>";
                                                                                print "</span>";
                                                                        print "</span>";
                                                                print "</th>";

                                                                print "<td class='right_td'>";
                                                                        if($strong)
                                                                                print "<strong>";
                                                                        print try_number_format($count  );
                                                                print "</td>";

                                                        print "<td class='right_td'>";
                                                                        print "<span class='tip_r' title='" . $tip2 . "'><span class='tippy'> ";
                                                                        if($strong)
                                                                                print "<strong>";
                                                                                print format_percent($count / $total , $count) ;
                                                                        if($strong)
                                                                                print "</strong>";
                                                                        print "</span></span></td>";

                                                print "</tr>";
                                        }

              // paired end report
              if ($PairedEnd){
                 $bflist = array(
                       "Forward" => $sigproc_results . "/fwd_bfmask.stats",
                       "Reverse" => $sigproc_results . "/rev_bfmask.stats",
                 );
                 $bftlist = array(
                       "Forward" => $sigproc_results . "/fwd_beadSummary.filtered.txt",
                       "Reverse" => $sigproc_results . "/rev_beadSummary.filtered.txt",
                 );
                 $pnglist = array(
                       "Forward" => $sigproc_results . "/fwd_Bead_density_contour.png",
                       "Reverse" => $sigproc_results . "/rev_Bead_density_contour.png",
                 );

              } else {
                 $bflist = array("default" => $sigproc_results . "/bfmask.stats");
                 $bftlist = array("default" => $sigproc_results . "/beadSummary.filtered.txt");
                 $pnglist = array("default" => $sigproc_results . "/Bead_density_contour.png");
              }

        foreach($bflist as $bfName => $bfFile) {

            print '<div style="overflow: hidden;">';//start wrapper
            print '<div class="half_width">';
            print '<div class="beadTable">';

            if (file_exists($bfFile)) {

                    if (count($bflist) > 1) {
                        print  '<br>' . $bfName . ' Run';
                    }

                    $bf = parse_to_keys($bfFile);

                    print '<table class="noheading half_width">';
                                        print '<col width="265px">';
                                        print '<col width="105px">';
                                        print '<col width="105px">';

//                                      $json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
//                                      $layout = json_decode($json,true);
//                    //peak signal
//                                      $bf = parse_to_keys($layout['Beadfind']['file_path']);

                                        //unrolling the loop so that I have greater control over what is displayed
                                        if($bf){
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
                                                                                print format_percent($wells_with_isp / $total_addressable_wells, $wells_with_isp) ;
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
                                                                        print "<span class='tip_r' title='Live ISPs / Total Addressable Wells'><span class='tippy'> ";
                                                                                print format_percent($live_wells / $wells_with_isp,$live_wells) ;
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
                                                                                print format_percent($tf_isp / $live_wells,$tf_isp)  ;
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
                                                                        print "<span class='tip_r' title='Library ISPs/Wells with ISPs'><span class='tippy'> ";
                                                                                print format_percent($library_isp / ($live_wells) ,$library_isp)  ;
                                                                print "</span></span></td>";

                                                        print "</tr>";

                                        }

                                        print "</table>";


                                        $bft_file=$bftlist["$bfName"];
//                                        print $bft_file;

                                        //now the second table for library ISP info
                                        $ISP_beadSummary = parse_beadSummary($bft_file);

                                        $ISP_polyclonal  = $ISP_beadSummary[1][2];
                                        $ISP_highPPF     = $ISP_beadSummary[1][3];
                                        $ISP_zero        = $ISP_beadSummary[1][4];
                                        $ISP_short       = $ISP_beadSummary[1][5];
                                        $ISP_badKey      = $ISP_beadSummary[1][6];
                                        $ISP_highRes     = $ISP_beadSummary[1][7];
                                        $ISP_clipAdapter = $ISP_beadSummary[1][8];
                                        $ISP_clipQual    = $ISP_beadSummary[1][9];
                                        $ISP_valid       = $ISP_beadSummary[1][10];

                                        $ISP_lowQual = $ISP_highPPF + $ISP_zero + $ISP_short + $ISP_badKey + $ISP_highRes + $ISP_clipQual;


                                        print "<table class='noheading half_width'>";
                                        print '<col width="265px">';
                                        print '<col width="105px">';
                                        print '<col width="105px">';
                                        print "<tr> <th> </th>    <th class='right_td'>Count</th>";
                                        print "<th class='right_td'>Percentage</th> </tr> ";

                                        printISPTableEntry("Library ISPs / Percent Enrichment", "Percent of non-TF loaded wells that are library",  "Library Enrichment = Library ISPs / (Wells with ISPs - Test Fragment ISPs)", $library_isp,     $wells_with_isp-$tf_isp, "",         false);
                                        printISPTableEntry("Filtered: Polyclonal",              "More than one template per ISP",                   "Polyclonal / Library ISPs",                                                  $ISP_polyclonal,  $library_isp,            "&#8227; ", false);
                                        printISPTableEntry("Filtered: Primer dimer",            "Insert length less than 8 bases",                  "Primer dimer / Library ISPs",                                                $ISP_clipAdapter, $library_isp,            "&#8227; ", false);
                                        printISPTableEntry("Filtered: Low quality",             "Low quality",                                      "Low quality / Library ISPs",                                                 $ISP_lowQual,     $library_isp,            "&#8227; ", false);
                                        printISPTableEntry("Final Library Reads",               "Reads passing all filters; in SFF/FASTQ",          "Final Library Reads / Library ISPs",                                         $ISP_valid,       $library_isp,            "&#8227; ", true);

                                        print "</table>";

                                print "</table>";

        }//if

                    print '</div>';//beadtable
                print '</div>'; //<!-- end half width -->

                                        $Bead_density_contour_file=$pnglist["$bfName"];
//                                        print $Bead_density_contour_file;

                                                if(file_exists($Bead_density_contour_file))
                                                {
                                                        print '<div class="on_right">';
                                                        print '<a class="box" href="' . $Bead_density_contour_file . '">';
                                                        print '<img src="' . $Bead_density_contour_file . '" width="400" height="300" border="0"/>';
                                                        print '</a>';
                                                        print '</div>';
                                                }
        print '</div>';//end wrapper
    }//end foreach

print '</div>';
print '</div>';

//<!-- stop ion sphere -->


//<!--  Start Report Information -->
print '<div id="ReportInfo" class="report_block">';


if ($PairedEnd){
        print '<h2>Paired End Report Information</h2>';
} else {
        print '<h2>Report Information</h2>';
}        

print '<div>'; // report information

print '<div>';//<!-- start wrapper -->
print '<h3>Analysis Info</h3>';
print '<div>';

if ($PairedEnd){
                        print '<table class="noheading">';
                                print '<col width="180px" />';
                                print '<col width="360px" />';
                                print '<col width="360px" />';
                                print "<th></th>";
                                print "<th>Forward</th>";
                                print "<th>Reverse</th>";

                                        if($fwdmeta and $revmeta){
                                                foreach($fwdmeta as $key => $value){
                                                        if ($fwdmeta["$key"] == $revmeta["$key"]) {
                                                            print "<tr><th>$key</th><td>" . wordwrap($value,30,"</br>",true) . "</td><td>" . $revmeta["$key"] . "</td></tr>";
                                                        } else {
                                                            print "<tr><th>$key</th><td style='color:red'>" . wordwrap($value,30,"</br>",true) . "</td><td style='color:red'>" . $revmeta["$key"] . "</td></tr>";
                                                        }
                                                }
                                        }
                                        $fwdprocess = parse_to_keys("fwd_processParameters.txt");
                                        $revprocess = parse_to_keys("rev_processParameters.txt");
                                        if ($fwdprocess){
                                                print "<tr><th>Flow Order</th>";
                                                if ($fwdprocess["flowOrder"] == $revprocess["flowOrder"]) { $tdcolor = "<td>"; } else { $tdcolor = "<td style='color:red'>"; }
                                                print $tdcolor . wordwrap($fwdprocess["flowOrder"],39,"</br>",true) . "</td>";
                                                print $tdcolor . wordwrap($revprocess["flowOrder"],39,"</br>",true) . "</td>";
                                                print "</tr>";
                                                print "<tr><th>Library Key</th>";
                                                if ($fwdprocess["libraryKey"] == $revprocess["libraryKey"]) { $tdcolor = "<td>"; } else { $tdcolor = "<td style='color:red'>"; }
                                                print $tdcolor . $fwdprocess["libraryKey"] . "</td>";
                                                print $tdcolor . $revprocess["libraryKey"] . "</td>";
                                                print "</tr>";
                                        }                                        

                        print '</table>';
       
}
else {
                        print '<table class="noheading">';
                                print '<col width="325px" />';
                                print '<col width="520px" />';

                                        $process = parse_to_keys("processParameters.txt");
                                        if($meta){
                                                foreach ($meta AS $key => $value) {
                                                        print "<tr><th>$key</th> <td>$value</td></tr>";
                                                }
                                        }
                                        if ($process){
                                                print "<tr>";
                                                print "<th>Flow Order</th>";
                                                print "<td>";
                                                print wordwrap($process["flowOrder"],51,"</br>",true);
                                                print "</td>";
                                                print "</tr>";

                                                print "<tr>";
                                                print "<th>Library Key</th>";
                                                print "<td>";
                                                print $process["libraryKey"];
                                                print "</td>";
                                                print "</tr>";
                                        }

                        print '</table>';
}
      print '</div>';
      print '</div>'; //<!--  end wrapper -->

            print '<h3>Software Version</h3>';
            print '<div>';
                if ($PairedEnd) {
                    $versiontxt_fwd = parseVersion("fwd_version.txt");
                    $versiontxt_rev = parseVersion("rev_version.txt");
                }
                $versiontxt = parseVersion("version.txt");
                $versiontxt_blacklist = array(
                    "ion-onetouchupdater" => "",
                    "ion-pgmupdates" => "",
                    "ion-publishers" => "",
                    "ion-referencelibrary" => "",
                    "ion-sampledata" => "",
                    "ion-docs" => "",
                    "ion-tsconfig" => "",
                    "ion-rsmts" => "",
                    "ion-usbmount" => "",
                );
                $explogtxt_whitelist = array(
                    "Script" => "",
                    "LiveView" => "",
                    "Datacollect" => "",
                    "OS" => "",
                    "Graphics" => "",
                );
                if($versiontxt):
                    // This will get the fist line read from versions.txt
                    // which we will bold and use as the header row below.
                    list($name, $version) = each($versiontxt);
                    unset($versiontxt[$name]);
                    $versionstxt = array_diff_key($versiontxt, $versiontxt_blacklist);
                    $explog_dict = array_intersect_key($explog_dict, $explogtxt_whitelist);
                    $versions = array_merge($versionstxt, $explog_dict);
                    // This sorts the software => version list alphabetically.
                    ksort($versions);
                    print '<table class="noheading">';
                    print '<col width="325px" />';
                    print '<col width="520px" />';

                    print '<tr>';
                    print '<th>' . $name . '</th>';
                    print '<td><strong>' . $version . '</strong></td>';
                    print '</tr>';
                    foreach($versions as $name => $version):
                        print '<tr>';
                        print '<th>' . $name . '</th>';
                        print '<td>' . $version;
                        //append version information if different
                        if ($PairedEnd and array_key_exists($name,$versiontxt_fwd)) {
                          if( $versionstxt["$name"] != $versiontxt_fwd["$name"] or
                              $versionstxt["$name"] != $versiontxt_rev["$name"] ) {
                            print ' (fwd: ' . $versiontxt_fwd["$name"] . ') (rev: ' . $versiontxt_rev["$name"] . ')';
                          } 
                        }
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

            if(file_exists("expMeta.dat")){
              if ($PairedEnd){
                $postfixsff_corrected = $basecaller_results . '/' . $base_name . '_corrected.sff' . '.zip';
                $bamfile_corrected = $corrected_path . '/' . $base_name . '_corrected.bam';
                $baifile_corrected = $corrected_path . '/' . $base_name . '_corrected.bam.bai';
                $postfixsff_sampled_corrected = $corrected_path . '/' . $base_name . '_corrected.sampled.sff' . '.zip';
                $postfixfastq_corrected = $corrected_path . '/' . $base_name . '_corrected.fastq' . '.zip';
                $postfixfastq_sampled_corrected = $corrected_path . '/' . $base_name . '_corrected.sampled.fastq' . '.zip';

                $postfixsff = $basecaller_results . '/' . $base_name . '.sff' . '.zip';
                $bamfile = $alignment_results . '/' . $base_name . '.bam';
                $baifile = $alignment_results . '/' . $base_name . '.bam.bai';
                $postfixsff_sampled = $base_name . '.sampled.sff' . '.zip';
                $postfixfastq = $base_name . '.fastq' . '.zip';
                $postfixfastq_sampled = $base_name . '.sampled.fastq' . '.zip';

                $postfixsff_fwd = $basecaller_results . '/' . $base_name . '_forward.sff' . '.zip';
                $tf_postfixsff_fwd = $basecaller_results . '/' . $base_name . '_forward.tf.sff' . '.zip';
                $bamfile_fwd = $alignment_results . '/' . $base_name . '_forward.bam';
                $baifile_fwd = $alignment_results . '/' . $base_name . '_forward.bam.bai';
                $postfixfastq_fwd = $base_name . '_forward.fastq' . '.zip';

                $postfixsff_rev = $basecaller_results . '/' . $base_name . '_reverse.sff' . '.zip';
                $tf_postfixsff_rev = $basecaller_results . '/' . $base_name . '_reverse.tf.sff' . '.zip';
                $bamfile_rev = $alignment_results . '/' . $base_name . '_reverse.bam';
                $baifile_rev = $alignment_results . '/' . $base_name . '_reverse.bam.bai';
                $postfixfastq_rev = $base_name . '_reverse.fastq' . '.zip';

                $postfixsff_prd_fwd = $basecaller_results . '/' . $base_name . '_Paired_Fwd.sff' . '.zip';
                $postfixsff_prd_rev = $basecaller_results . '/' . $base_name . '_Paired_Rev.sff' . '.zip';

                $postfixsff_singleton_fwd = $basecaller_results . '/' . $base_name . '_Singleton_Fwd.sff' . '.zip';
                $postfixfastq_singleton_fwd = $basecaller_results . '/Singleton_Fwd/' . $base_name . '_Singleton_Fwd.fastq' . '.zip';
                $bamfile_singleton_fwd = $basecaller_results . '/Singleton_Fwd/' . $base_name . '_Singleton_Fwd.bam';
                $baifile_singleton_fwd = $basecaller_results . '/Singleton_Fwd/' . $base_name . '_Singleton_Fwd.bam.bai';

                $postfixsff_singleton_rev = $basecaller_results . '/' . $base_name . '_Singleton_Rev.sff' . '.zip';
                $postfixfastq_singleton_rev = $basecaller_results . '/Singleton_Rev/' . $base_name . '_Singleton_Rev.fastq' . '.zip';
                $bamfile_singleton_rev = $basecaller_results . '/Singleton_Rev/' . $base_name . '_Singleton_Rev.bam';
                $baifile_singleton_rev = $basecaller_results . '/Singleton_Rev/' . $base_name . '_Singleton_Rev.bam.bai';

              }
              else{
                $postfixsff = $basecaller_results . '/' . $base_name . '.sff' . '.zip';
                $tf_postfixsff = $basecaller_results . '/' . $base_name . '.tf.sff' . '.zip';
                $bamfile = $alignment_results . '/' . $base_name . '.bam';
                $baifile = $alignment_results . '/' . $base_name . '.bam.bai';
                $postfixsff_sampled = $base_name . '.sampled.sff' . '.zip';
                $postfixfastq = $base_name . '.fastq' . '.zip';
                $postfixfastq_sampled = $base_name . '.sampled.fastq' . '.zip';
                }
            }
            else{
                //block case
                $postfixsff = 'rawlib.sff';
                $tf_postfixsff = 'rawtf.sff';
                $bamfile = 'rawlib.bam';
                $baifile = 'rawlib.bam.bai';
                $postfixsff_sampled = $base_name . '.sampled.sff';
                $postfixfastq = $base_name . '.fastq';
                $postfixfastq_sampled = $base_name . '.sampled.fastq';
            }

                        //If the Analysis is done, present the files
                        if (!$progress){
                if ($PairedEnd){
                  print "<thead><tr>";

                  //header info here
//	                  print "<th></th>";
                  print "<th>Union</th>";
                  print "<th>Corrected</th>";
//                  print "<th>Paired Forward</th>";
//                  print "<th>Paired Reverse</th>";
                  print "<th>Unpaired Forward</th>";
                  print "<th>Unpaired Reverse</th>";

                  print "</tr></thead>";
                }else{
                           print "<thead>";
                           print "</thead>";
                }

                                if (!$align_full){
                                        print "<tr>";
                                        print "<td><a href='$postfixsff'>Library Sequence (SFF)</a></td>";
                                        print "</tr>";
                                        print "<tr><td><a href='$postfixsff_sampled'>Sampled Library Sequence (SFF) </a> </td> </tr>";
                                }else{
                                        print "<tr>";
                                        print "<td><a href='$postfixsff'>Library Sequence (SFF)</a></td> ";
                                        if ($PairedEnd){
                                            print "<td><a href='$postfixsff_corrected'>Library Sequence (SFF)</a></td> ";
//                                            print "<td><a href='$postfixsff_fwd'>Library Sequence (SFF)</a></td> ";
//                                            print "<td><a href='$postfixsff_rev'>Library Sequence (SFF)</a></td> ";
                                            print "<td><a href='$postfixsff_singleton_fwd'>Library Sequence (SFF)</a></td> ";
                                            print "<td><a href='$postfixsff_singleton_rev'>Library Sequence (SFF)</a></td> ";
                                        }
                                        print "</tr>";
                                }

                                if (!$align_full){
                                        print "<tr><td><a href='$postfixfastq'>Library Sequence (FASTQ) </a> </td></tr>";
                                        print "<tr><td><a href='$postfixfastq_sampled'>Sampled Library Sequence (FASTQ) </a> </td> </tr>";
                                }else{
                                        print "<tr>";
                                        print "<td><a href='$postfixfastq'>Library Sequence (FASTQ) </a> </td>";
                                        if ($PairedEnd){
                                            print "<td><a href='$postfixfastq_corrected'>Library Sequence (FASTQ)</a></td>";
//                                            print "<td><a href='$postfixfastq_fwd'>Library Sequence (FASTQ)</a></td> ";
//                                            print "<td><a href='$postfixfastq_rev'>Library Sequence (FASTQ)</a></td> ";
                                            print "<td><a href='$postfixfastq_singleton_fwd'>Library Sequence (FASTQ)</a></td>";
                                            print "<td><a href='$postfixfastq_singleton_rev'>Library Sequence (FASTQ)</a></td>";
                                        }
                                        print "</tr>";
                                }

                                if (!$align_full){
                                        print "<tr><td><a href='$bamfile'>Sampled Library Alignments (BAM)</a ></td> </tr>";
                                        print "<tr><td><a href='$baifile'>Sampled Library Alignments (BAI)</a ></td> </tr>";
                                }else{
                                        print "<tr>";
                                        print "<td><a href='$bamfile'>Full Library Alignments (BAM)</a></td>";
                                        if ($PairedEnd){
                                            print "<td><a href='$bamfile_corrected'>Full Library Alignments (BAM)</a></td> ";
//                                            print "<td><a href='$bamfile_fwd'>Full Library Alignments (BAM)</a></td> ";
//                                            print "<td><a href='$bamfile_rev'>Full Library Alignments (BAM)</a></td> ";
                                            print "<td><a href='$bamfile_singleton_fwd'>Full Library Alignments (BAM)</a></td> ";
                                            print "<td><a href='$bamfile_singleton_rev'>Full Library Alignments (BAM)</a></td> ";
                                        }
                                        print "</tr>";

                                        print "<tr>";
                                        print "<td><a href='$baifile'>Full Library Alignments (BAI)</a></td>";
                                        if ($PairedEnd){
                                            print "<td><a href='$baifile_corrected'>Full Library Alignments (BAI)</a></td> ";
//                                            print "<td><a href='$baifile_fwd'>Full Library Alignments (BAI)</a></td> ";
//                                            print "<td><a href='$baifile_rev'>Full Library Alignments (BAI)</a></td> ";
                                            print "<td><a href='$baifile_singleton_fwd'>Full Library Alignments (BAI)</a></td> ";
                                            print "<td><a href='$baifile_singleton_rev'>Full Library Alignments (BAI)</a></td> ";
                                        }
                                        print "</tr>";
                                }

                        if (file_exists("barcodeList.txt")) {
                                //barcode sff and fastq files
                                print "<tr><td><a href='$base_name.barcode.sff.zip'>Barcode-specific Library Sequence (SFF)</a></td> </tr>";
                                print "<tr><td><a href='$base_name.barcode.fastq.zip'>Barcode-specific Library Sequence (FASTQ)</a></td> </tr>";
                            if (file_exists("barcodeFilter.txt")) {
                                print "<tr><td><a href='barcodeFilter.txt'>Barcode read frequency filter</a></td> </tr>";
                            }
                        }

                        if (file_exists("alignment_barcode_summary.csv")) {
                                //if there are barcodes zip the bam and bai files

                                print "<tr><td><a href='alignment_barcode_summary.csv'>Barcode Alignment Summary</a ></td> </tr>";
                                print "<tr><td><a href='$base_name.barcode.bam.zip'>Barcode-specific Library Alignments (BAM)</a></td> </tr>";
                                print "<tr><td><a href='$base_name.barcode.bai.zip'>Barcode-specific Library Alignments Index Files (BAI)</a></td> </tr>";
                        }
                                if ($PairedEnd) {
//                                    print "<td></td>";
//                                    print "<td></td>";
//                                    print "<td><a href='$tf_postfixsff_fwd'>Test Fragments (SFF)</a></td> ";
//                                    print "<td><a href='$tf_postfixsff_rev'>Test Fragments (SFF)</a></td> ";
                                } else {
                                    print "<tr><td><a href='$tf_postfixsff'>Test Fragments (SFF)</a> </td> </tr> ";
                                }

                                if ($PairedEnd){
                                    print '</table><table class="noheading">';
                                }
                                print "<tr><td><a href='Default_Report.php?do_print=True'>PDF of this Report </a> </td> </tr>";
                        }

            if (file_exists("csa.php")) {
                print "<tr><td><a href='csa.php'>Customer Support Archive</a> </td> </tr>";
            }
            if (!file_exists("status.txt")) {
                print "<tr><td><a href='log.html'>Report Log</a> </td> </tr>";
            }
            if (file_exists("/opt/ion/.developerversion")) {
                if (file_exists("drmaa_stdout_block.html")) {
                    print "<tr><td><a href='drmaa_stdout_block.html'>Developer link: Block stdout</a> </td> </tr>";
                }
                if (file_exists("drmaa_stderr_block.txt")) {
                    print "<tr><td><a href='drmaa_stderr_block.txt'>Developer link: Block stderr</a> </td> </tr>";
                }
                if (file_exists("drmaa_stdout.txt")) {
                    print "<tr><td><a href='drmaa_stdout.txt'>Developer link: std_out_err</a> </td> </tr>";
                }
            }
                        print "</table></div></div>";

                        ?>
<!--  end of files -->

        <div id="pluginStatus" class='report_block'>
                <h2>Plugin Summary</h2>

                <div>
                        <div id="toolbar" class="ui-widget-header ui-corner-all">
                                <a id="pluginDialogButton">Select Plugins To Run</a>
                                <span style="left: 200px; position: relative;" id="pluginLoadMsg"></span>
                                <a id="pluginRefresh">Refresh Plugin Status</a>
                        </div>
                        <div id="pluginStatusLoad"></div>
                                <table id="pluginStatusTable" >
                                        <col width='70%' />
                                        <col width='30%' />
                                </table>

                </div> <!--plugin status wrapper -->
        </div>


        <div id="pluginDialog">
                <div id="pluginLoad"></div>
                <div id="pluginList"></div>
        </div>

        <pre id="logBox" style="display:hidden"></pre>
<!--  end plugin dialog -->

</div>

 <script type="text/javascript">


        function htmlEscape(str) {
            return String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#39;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/\n/g, '<br\>');
        }

        //string endswith
        String.prototype.endsWith = function(str)
       {return (this.match(str+"$")==str)}

        //get the status of the plugins from the API
        function pluginStatusLoad(){

                //init the spinner -- this is inside of the refresh button
                $('#pluginRefresh').activity({segments: 10, width:3, space:2, length: 3, color: '#252525', speed: 1.5, padding: '3', align: 'left'});

                $("#pluginStatusTable").fadeOut();
                $("#pluginStatusTable").html("");

                $.ajax({
                    type: 'GET',
                    url: djangoURL,
                    contentType: "application/json; charset=utf-8",
                    dataType: "json",
                    async: false,
                    success: function(data) {
                        for (var i=0; i < data.length; i++ ) {
                            var row = "";
                           //Write plugin name
                           row += "<tr><th>";
                           row += data[i].Name;
                           row += " &#8212; v";
                           row += data[i].Version;
                           row +="</th><th>";
                          //Write the job status
                           row += data[i].State;
                           row += '<span style="float: right;"><a class="pluginLog ui-icon ui-icon-script" data-title="Log for ' + data[i].Name +'" title="Log File for ' + data[i].Name + '" href="plugin_out/' + data[i].Name + '_out/drmaa_stdout.txt">log</a></span>' ;
                           row +="</th></tr>";
                          //for each file make a new row under the plugin name
                          //if it ends with _block.html or _block.php then render that to a div inline
                          row += '<tr colspan=2 id="' + data[i].Name + '">';
                          if (data[i].Files){
                             for (var j=0; j < data[i].Files.length; j++) {
                                  if (data[i].Files[j].endsWith("_block.html") || data[i].Files[j].endsWith("_block.php")) {
                                       row += '<tr ><td  colspan=2 style="padding: 0px;"><div class="pluginLevel1">';
                                       row += '<iframe id="'+data[i].Name+'" class="pluginBlock" src="plugin_out/' + data[i].Name + '_out/' + data[i].Files[j]+'" width="95%" frameborder="0" height="0px" ></iframe>';
                                       row += "</div></td></td>";
                                       }
                                                                                                                                                                                                    else {
                                        row += '<tr><td colspan=2 style="padding: 0px;">';
                                        row += '<div class="pluginLevel1">&#8227; <a href=plugin_out/' + data[i].Name + '_out/' + data[i].Files[j] + '>';
                                        row += data[i].Files[j];
                                        row += "</a></div></td></td>";
                                                                                                                                                                                                    }

                             }
                          }
                          row += '</tr>';
                          $("#pluginStatusTable").append(row);
                          $('.pluginLog').tipTip({position: 'left' });
                        }
                    },
                    error: function(msg) {
                        $("#pluginStatusTable").text("Failed to get Plugin Status: " + msg);
                     }
              });//for ajax call


                $("#pluginStatusTable").fadeIn();
                $('#pluginRefresh').activity(false);

                //not pretty
                //try to resize the iframes for a while
                for(i = 100; i < 10000; i = i+200){
                        setTimeout("resizeiFrames()", 200);
                }

        }

        function resizeiFrames(){
                //Resize the iframe blocks
                $(".pluginBlock").each(function(){
                        $(this).height($(this).contents().height());
                });
        }


 $(document).ready(function(){

        $(window).scroll(function () {
                resizeiFrames();
        });

        <?php
                //write the javascript var with php
                print 'djangoPK = ' . $djangoPK["ResultsPK"] .';';
        print 'djangoURL = "/rundb/api/v1/results/' . $djangoPK["ResultsPK"] .'/pluginresults/";';


                //Check to see if this is a mobile device
                print 'djangoMobileURL = "/rundb/mobile/report/' . $djangoPK["ResultsPK"] .'";';

                if (!isset($_GET["force_desktop"]) || $_GET["force_desktop"] != True){
                        print 'if (jQuery.browser.mobile) { window.location = djangoMobileURL; }';
                }

        ?>

        $(".heading tr:odd").addClass("zebra");
        $(".noheading tr:odd").addClass("zebra");

        //add tooltips
        $().tipTipDefaults({ delay : 0 });
        $('.tip').tipTip({ position : 'bottom' });
        $('.tip_r').tipTip({ position : 'bottom' });

        //do the initial plugin status load
        pluginStatusLoad();

        //provide the SGE log for plugins
        $('.pluginLog').live("click", function() {
                var url = $(this).attr("href");
                dialog = $("#logBox");
                dialog.html("");
                var title = $($(this)).data("title");

                // load remote content
                var logParent = $($(this).parent());
                logParent.activity({segments: 10, width:3, space:2, length: 2.5, color: '#252525', speed: 1.5, padding: '3' });
                $.get(
                    url,
                    function (responseText) {
                                dialog.html(htmlEscape(responseText));
                                logParent.activity(false);
                                dialog.dialog({
                                    width : 900,
                                    height : 600,
                                    title : title,
                                    buttons :   [
                                                        {
                                                                text: "Close",
                                                                click: function() {
                                                                        $(this).dialog("close");
                                                                }
                                                        }
                                                ]
                                });
                        }
                );
                //prevent the browser to follow the link
                return false;
        });

         $("#pluginRefresh").button({
             icons: {
                 primary: 'ui-icon-refresh'
             }
         }).click(function() {
                        pluginStatusLoad()
                }
         );

         //the plugin launcher
        $("#pluginDialogButton").button({
            icons: {
                primary: 'ui-icon-wrench'
            }
        }).click(function() {

                //open the dialog
                $("#pluginDialog").dialog({
                    width : 500,
                    height : 600,
                    title : "Plugin List",
                    buttons : [
                        {
                            text: "Close",
                            click: function() {
                                $(this).dialog("close");
                            }
                        }
                    ]
                });

                //init the loader
                $("#pluginList").html("");
                $("#pluginLoad").html("<span>Loading Plugin List <img src='/site_media/jquery/colorbox/images/loading.gif'></img></span>");

                $("#pluginList").html("");
                $("#pluginList").hide();

                //get the list of plugins from the API
                $.ajax({
                        url : '/rundb/api/v1/plugin/?selected=true&limit=0&format=json&order_by=name',
                        dataType : 'json',
                        type : 'GET',
                        async : false,
                        success : function(data) {
                                var items = [];

                                if (data.objects.length == 0){
          $("#pluginLoad").html("");
          $("#pluginList").html("<p> There are no plugins what are enabled </p>");
          return false;
                                }
        $("#pluginList").html('<ul id="pluginUL" class="expandable"></ul>');

        plugins = data.objects.sort(function(a, b) {
            return a.name.toLowerCase() > b.name.toLowerCase() ? 1 : -1;
        });

        //build the query string in a way that works with IE7 and IE8
        plugin_ids = "";
        $.each(plugins, function(count,value){
            plugin_ids += value.id;
            if (count+1 != plugins.length){
                plugin_ids += ";";
            }
        });

        //get plugin metadata
        $.ajax({
          url : "/rundb/api/v1/plugin/set/" + plugin_ids + "/type/?format=json",
          success : function(plugin_types) {
            for(var i=0;i<plugins.length;i++){
              val = plugins[i];
              data = plugin_types[val.id];
              if (data.input != undefined){
                    $("#pluginUL").append('<li data-id="' + val.id + '" class="plugin_input_class" id="' + val.name + '_plugin"><a href="'+data.input+'?report='+djangoPK+'" class="plugin_link colorinput">' + val.name + '</a>' + '<span>' + " &#8212; v" + val.version +'</span></li>');
              }else{
                    $("#pluginUL").append('<li data-id="' + val.id + '" class="plugin_class" id="' + val.name + '_plugin"><a href="#pluginDialog">' + val.name + '</a>' + '<span>' +" &#8212; v" +  val.version + '</span></li>');
              }
            }
                                        },
                                        async : false,
                                        type : 'GET',
                                        dataType : 'json'
                                });

                                $(".colorinput").colorbox({width:"90%", height:"90%", iframe:true})
                                $("#pluginLoad").html('<h5>' + "Click a plugin to run:" + '</h5>');
                                $("#pluginList").show();
                        }
                });

                //now the the for each is done, show the list
                $(".plugin_input_class").die("click");

                $(".plugin_input_class").live("click", function(){
                        $("#pluginDialog").dialog("close");
                });

                $(".plugin_class").die("click");

                $(".plugin_class").live('click', function() {
                        //get the plugin id
                        var id = $(this).data('id');
                        var pluginName = $(this).attr("id");
                        //get the plugin name
                        pluginName = pluginName.substring(0, pluginName.length - 7)
                        //build the JSON to post
                        pluginAPIJSON = { "plugin" : [pluginName] };
                        pluginAPIJSON = JSON.stringify(pluginAPIJSON);
                        $.ajax({
                              type : 'POST',
                              url : djangoURL,
                              async: true,
                              contentType : "application/json; charset=utf-8",
                              data : pluginAPIJSON,
                              dataType : "json",
                              beforeSend : function() {
                                        $("#pluginList").html("");
                                        $("#pluginLoad").html("<span>Launching Plugin " + pluginName + " <img src='/site_media/jquery/colorbox/images/loading.gif'></img></span>");
                              },
                              success : function() {
                                        $("#pluginDialog").dialog("close");
                                        pluginStatusLoad();
                              }
                        });
                });
        });

        $('h2').append('<a href="javascript:;" class="expandCollapseButton" title="Collapse Section"></a>');

        $('.expandCollapseButton').toggle(function() {
                if ( $(this).attr('title') == 'Collapse Section'){
                        $(this).css('background-position','right top');
                        $(this).attr('title','Expand Section');
                }else{
                        $(this).css('background-position','left top');
                        $(this).attr('title','Collapse Section');
                }
        }, function() {
                if ( $(this).attr('title') == 'Expand Section'){
                        $(this).css('background-position','left top');
                        $(this).attr('title','Collapse Section');
                }else{
                        $(this).css('background-position','right top');
                        $(this).attr('title','Expand Section');
                }
        });

        $('.expandCollapseButton').click(function(event){
                $(event.target).parent().parent().toggleClass('small');
                $(event.target).parent().next().slideToggle();
        });

        <?php
                if ($progress){
                        echo 'setInterval("fancy_load()", 5000 );';
                        echo '$(".topbar, .darkline, #progress_box, h1").expose();';
                }

                //when to show and hide the tf
                $hide_tf = true;

                if (isset($_GET['no_header']) == True ){
                        if ($_GET['no_header'] == True){
                                $hide_tf = false ;
                                $is_print = true;
                        }else{
                                //most of the time this should be hidden -- set to true
                                $hide_tf = true;
                        }
                }

                $hide_lib = false;

                if ($tf_only == true){
                        $hide_tf = false;
                        $hide_lib = true;
                }

                //keep the TF collapsed unless this is a print or a tf only run
                if ( $hide_tf ){
                        print "$('#tf .expandCollapseButton').css('background-position','right top');";
                        print "$('#tf .expandCollapseButton').attr('title','Expand Section');";
                        print "$('#tf').parent().toggleClass('small');";
                        print "$('#tf').next().toggle();";
                }
                //hide this, unless it is a print
                if ( $hide_lib && !$is_print){
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
        if ( isset($_GET['no_header'])){
                if ($_GET['no_header'] == True){
                        echo '$(".topbar").toggle();';
                        echo '$(".darkline").toggle();';
                        print "$('.expandCollapseButton').hide();";
                }
        }

        ?>

        var keys    = [];
        var konami  = '38,38,40,40,37,39,37,39,66,65';

        $(document)
                .keydown(
                        function(e) {
                                keys.push( e.keyCode );
                                if ( keys.toString().indexOf( konami ) >= 0 ){
                                        $(".zebra").css("background-color","#FF0080");
                                        keys = [];
                                }
                        }
                );

});

</script>

<?php
        //now if there was a footer found, write it.
        print $footer;
?>
