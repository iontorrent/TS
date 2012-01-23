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
        $meta = parseMeta("expMeta.dat");
	}
    else{
        $meta = parseMeta("../expMeta.dat");
    }
    $resultsName = $meta[2][1];
    $libName = $meta[7];
    $libName = $libName[1];
    $expName = $meta[0][1];
    $base_name = $expName . '_' . $resultsName;

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
	$header = str_replace("<title>Torrent Browser</title>","<title>". $resultsName . "</title>", $header );
	
	print $header;
	
	function format_percent($input_percent, $count){
		$input_percent = $input_percent * 100;
		//this is for the ISP table	
		if ($count >= 0){
			if ( is_numeric($input_percent) ){
				if ($input_percent < 1){
					return "<1%";
				}else{
					$number = number_format($input_percent ,0);
					return $number . "%"; 
				}
			}
		}else{
			return "0%";
		}
	}
	
	function try_number_format($value, $decimals = 0){
		if (is_numeric($value)){
			return number_format($value, $decimals);
		}else{
			return $value;
		}
	}
	
    $json = file_get_contents('ion_params_00.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
    $params = json_decode($json,true);
    $pathToData = $params["pathToData"];
    $fileIn = $pathToData."/explog.txt";
    if(file_exists($fileIn))
    {
      $file = fopen($fileIn, 'r') or die();
      //Output a line of the file until the end is reached
      $dataX = array();
      $dataY = array();
      while(($line = fgets($file)) !== false)
      {
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

     
<h1>Report for <?php print $resultsName?></h1>

<?php 
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
?>


<?php 


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
			$as = parse_to_keys($layout["Quality Summary"]["file_path"]);
			if($as){
				
				print '<table id="q_alignment" class="heading">';
				print "<col width='325px' />";
				print "<col width='520px' />";
				print "<thead>";
				
				print "</thead><tbody>";
				
				print "<tr>";
					print "<th>Total Number of Bases [Mbp]</th>";
					//Convert bases to megabases
					print "<td>" . try_number_format( $as['Number of Bases at Q0']/1000000 , 2 ) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th class='subhead'>&#8227; Number of Q17 Bases [Mbp]</th>";	
					print "<td>" . try_number_format( $as['Number of Bases at Q17']/1000000 , 2 ) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th class='subhead'>&#8227; Number of Q20 Bases [Mbp]</th>";	
					print "<td>" . try_number_format( $as['Number of Bases at Q20']/1000000 , 2 ) ."</td>";
				print "</tr>";

				print "<tr>";	
					print "<th>Total Number of Reads</th>";	
					print "<td>" . try_number_format( $as['Number of Reads at Q0'] ) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Mean Length [bp]</th>";	
					print "<td>" . try_number_format( $as['Mean Read Length at Q0']) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Longest Read [bp]</th>";	
					print "<td>" . try_number_format( $as['Max Read Length at Q0']) ."</td>";
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
		
			//If there is alignment info, print it
			$json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
			$layout = json_decode($json,true);
			$as = parse_to_keys($layout["Alignment Summary"]["file_path"]);
			//once we have as we can check to see if it contains the full data, or the sampled data
			if (isset($as['Total number of Sampled Reads'])) {
				$align_full = false;
			}else{
				$align_full = true;
			}
			
			if($align_full == true && $as){
				print '<h3>Based on Full Library Alignment to Provided Reference</h3>';
				print '<table id="alignment" class="heading">';
				print "<col width='325px' />";
				print "<col width='173px' />";
				print "<col width='173px' />";
				print "<col width='173px' />";
				
				print "<thead><tr><th> </th>";
				
				//header info here
				print "<th>AQ17</th>";
				print "<th>AQ20</th>";
				print "<th>Perfect</th>";
				
				print "</tr></thead><tbody>";
				
				print "<tr>";
					print "<th>Total Number of Bases [Mbp]</th>";
					//Convert bases to megabases
					print "<td>" . try_number_format( $as['Filtered Mapped Bases in Q17 Alignments']/1000000 , 2 ) ."</td>";
					print "<td>" . try_number_format( $as['Filtered Mapped Bases in Q20 Alignments']/1000000 , 2 ) ."</td>";
					print "<td>" . try_number_format( $as['Filtered Mapped Bases in Q47 Alignments']/1000000 , 2 ) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Mean Length [bp]</th>";	
					print "<td>" . try_number_format( $as['Filtered Q17 Mean Alignment Length']) ."</td>";
					print "<td>" . try_number_format( $as['Filtered Q20 Mean Alignment Length']) ."</td>";
					print "<td>" . try_number_format( $as['Filtered Q47 Mean Alignment Length']) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Longest Alignment [bp]</th>";	
					print "<td>" . try_number_format( $as['Filtered Q17 Longest Alignment']) ."</td>";
					print "<td>" . try_number_format( $as['Filtered Q20 Longest Alignment']) ."</td>";
					print "<td>" . try_number_format( $as['Filtered Q47 Longest Alignment']) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Mean Coverage Depth</th>";	
					print "<td>" . try_number_format( $as['Filtered Q17 Mean Coverage Depth'], 2) ."&times;</td>";
					print "<td>" . try_number_format( $as['Filtered Q20 Mean Coverage Depth'], 2) ."&times;</td>";
					print "<td>" . try_number_format( $as['Filtered Q47 Mean Coverage Depth'], 2) ."&times;</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Percentage of Library Covered</th>";	
					print "<td>" . try_number_format( $as['Filtered Q17 Coverage Percentage']) ."%</td>";
					print "<td>" . try_number_format( $as['Filtered Q20 Coverage Percentage']) ."%</td>";
					print "<td>" . try_number_format( $as['Filtered Q47 Coverage Percentage']) ."%</td>";
				print "</tr>";
				
				print "</tbody>";  
				print "</table>";
				
			}elseif ($align_full == false && $as){
				print '<h3>Based on Sampled Library Alignment to Provided Reference</h3>';
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
				print "<th>AQ17</th>";
				print "<th>AQ20</th>";
				print "<th>Perfect</th>";
				print "<th>AQ17</th>";
				print "<th>AQ20</th>";
				print "<th>Perfect</th>";
				
				print "</tr></thead><tbody>";
				
				print "<tr>";
					print "<th>Total Number of Bases [Mbp]</th>";
					//Convert bases to megabases
					print "<td>" . try_number_format( $as['Sampled Filtered Mapped Bases in Q17 Alignments']/1000000 , 2 ) ."</td>";
					print "<td>" . try_number_format( $as['Sampled Filtered Mapped Bases in Q20 Alignments']/1000000 , 2 ) ."</td>";
					print "<td>" . try_number_format( $as['Sampled Filtered Mapped Bases in Q47 Alignments']/1000000 , 2 ) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Mapped Bases in Q17 Alignments']/1000000 , 2 ) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Mapped Bases in Q20 Alignments']/1000000 , 2 ) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Mapped Bases in Q47 Alignments']/1000000 , 2 ) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Mean Length [bp]</th>";	
					print "<td>" . try_number_format( $as['Sampled Filtered Q17 Mean Alignment Length']) ."</td>";
					print "<td>" . try_number_format( $as['Sampled Filtered Q20 Mean Alignment Length']) ."</td>";
					print "<td>" . try_number_format( $as['Sampled Filtered Q47 Mean Alignment Length']) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q17 Mean Alignment Length']) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Mean Alignment Length']) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Mean Alignment Length']) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Longest Alignment [bp]</th>";	
					print "<td>" . try_number_format( $as['Sampled Filtered Q17 Longest Alignment']) ."</td>";
					print "<td>" . try_number_format( $as['Sampled Filtered Q20 Longest Alignment']) ."</td>";
					print "<td>" . try_number_format( $as['Sampled Filtered Q47 Longest Alignment']) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q17 Longest Alignment']) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Longest Alignment']) ."</td>";
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Longest Alignment']) ."</td>";
				print "</tr>";
				
				print "<tr>";	
					print "<th>Mean Coverage Depth</th>";	
					
					print "<td>" . try_number_format( $as['Sampled Filtered Q17 Mean Coverage Depth'], 2 ) ;
					print (is_numeric($as['Sampled Filtered Q17 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Sampled Filtered Q20 Mean Coverage Depth'], 2);
					print (is_numeric($as['Sampled Filtered Q20 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Sampled Filtered Q47 Mean Coverage Depth'], 2);
					print (is_numeric($as['Sampled Filtered Q47 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q17 Mean Coverage Depth'],2);
					print (is_numeric($as['Extrapolated Filtered Q17 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Mean Coverage Depth'],2 );
					print (is_numeric($as['Extrapolated Filtered Q20 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Mean Coverage Depth'],2 );
					print (is_numeric($as['Extrapolated Filtered Q47 Mean Coverage Depth']) ) ?  "&times; </td>" : " </td>" ;
				print "</tr>";
				
				print "<tr>";	
					//slightly special case because we only print % in the cell if the call has a number
					print "<th>Percentage of Library Covered</th>";	
					
					print "<td>" . try_number_format( $as['Sampled Filtered Q17 Coverage Percentage']) ;
					print (is_numeric($as['Sampled Filtered Q17 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Sampled Filtered Q20 Coverage Percentage']);
					print (is_numeric($as['Sampled Filtered Q20 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Sampled Filtered Q47 Coverage Percentage']);
					print (is_numeric($as['Sampled Filtered Q47 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q17 Coverage Percentage']);
					print (is_numeric($as['Extrapolated Filtered Q17 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q20 Coverage Percentage']);
					print (is_numeric($as['Extrapolated Filtered Q20 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;
					
					print "<td>" . try_number_format( $as['Extrapolated Filtered Q47 Coverage Percentage']);
					print (is_numeric($as['Extrapolated Filtered Q47 Coverage Percentage']) ) ?  "% </td>" : " </td>" ;
					
				print "</tr>";
				
				print "</tbody>";  
				print "</table>";
				
			}
		
		print "<h3>Read Alignment Distribution</h3>";
		
		//start alignTable.txt
        if (file_exists("alignTable.txt")) {
		$alignTable = tabs_parse_to_keys("alignTable.txt");
		if ($alignTable){ 
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
			print "</table>";
        	}
        	else{
        		print 'no alignments found';
        	}
	}
        else{
            print 'alignTable.txt not found';
	}
        //end alignTable.txt
		

	print '</div>';
	print '<div style="clear: both;"></div>';
		
	print '</div><!-- end wrapper -->';
print '</div>';
//End Library Summary
?>



<?php
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

		$barCodeSet = $meta[14][1];
		
		print '<div id="barCodes" class="report_block">';
		print	'<h2>Barcode Reports</h2><div>';
			graphCreate("Total number of Reads",$djangoPK["ResultsPK"],"Total%20number%20of%20Reads");
			graphCreate("AQ 20 Bases",$djangoPK["ResultsPK"],"Filtered%20Mapped%20Bases%20in%20Q20%20Alignments");
			graphCreate("Mean AQ20 read length",$djangoPK["ResultsPK"],"Filtered%20Q20%20Mean%20Alignment%20Length");
			graphCreate("AQ20 Reads",$djangoPK["ResultsPK"],"Filtered%20Q20%20Alignments");
		print '</div></div>';

	}
?>

<!--Start TF Summary -->
<div id='TFSummary' class='report_block'> 
 
	<h2 id="tf" >Test Fragment Report</h2>
	<div><!-- start wrapper -->
		
		<?php
		$process = parse_to_keys("processParameters.txt");
		if (in_array("Block",array_keys($process)))
		{
			$tf = parseTfMetrics("TFMapper.stats");
			$cafie = parseCafieMetrics("cafieMetrics.txt");
			
			$json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
			$layout = json_decode($json,true);
			if($tf and $cafie)
			{
				$mixedKeys = array_keys($tf);

				#filter out missing cafie keys
				$mixedKeys = array_keys($cafie);

				$tfKeys = NULL;
				foreach($mixedKeys as $k)
				{
					if($k != 'LIB' and $k != 'system' and $k != ' '){
						$tfKeys[]=$k;
					}
				}
				if ($tfKeys) sort($tfKeys);
				
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
				}

				if(count($tfKeys) > 0) {
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
			}
		} else {
			print"<h3>Test Fragment Summary (Heatmap version)</h3>";
		}
		?>
	</div><!-- end wrapper -->
</div>
<!--End TF Summary Div--> 

<!--  start ion sphere -->

<div id='IonSphere' class="report_block">

	<h2>Ion Sphere&trade; Particle (ISP) Identification Summary</h2>

	<?php
	if (file_exists($pathToData."/explog.txt") and count($dataX) > 0) {
	print '<table id="ispMap" border="0" cellpadding="0" cellspacing="0">';

        foreach (array_reverse($dataY) as $dy) {
        print "<tr class='slimRow'>";
                foreach($dataX as $dx) {
                        $needle = 'block_X'.$dx.'_Y'.$dy;
                        $needleImage = $needle . "/Bead_density_raw.png";
                        print '<td class="slim">';
                        print '<a href="'.$needle.'/Default_Report.php">';
                        if (file_exists($needleImage)){
                                print '<img src="'.$needleImage.'" class="tileSize" alt="'.$needle.'" border="0"/>';
                        }else if (file_exists($needle)) {
                                print '<img src="/site_media/images/dna-small-slow-yellow.gif" class="tileSize" alt="'.$needle.'" border="0"/>';
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
	?>

	<div style="overflow: hidden;"><!-- start wrapper -->
	  <div class="half_width">
			<div class='beadTable'>
				<table class='noheading half_width'>
				<?php
					print '<col width="265px">';
					print '<col width="105px">';
					print '<col width="105px">';

					$json = file_get_contents('report_layout.json', FILE_IGNORE_NEW_LINES and FILE_SKIP_EMPTY_LINES);
					$layout = json_decode($json,true);
                    //peak signal
					$bf = parse_to_keys($layout['Beadfind']['file_path']);
					
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
									print "<span class='tip_r' title='Library ISPs / Total Addressable Wells'><span class='tippy'> ";
										print format_percent($library_isp / ($live_wells) ,$library_isp)  ;
								print "</span></span></td>";
								
							print "</tr>";

					}
					
					print "</table>";

					//now the second table for library ISP info
					$ISP_beadSummary = parse_beadSummary("beadSummary.filtered.txt");
					
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
				
				
				?>
				</table>					
					
			</div>
			

			
		</div><!-- end half width -->
					<?php 	
						if(file_exists('Bead_density_contour.png'))
						{
							print '<div class="on_right">';		
							print ' <a class="box" href="Bead_density_contour.png">';
							print	'<img src="Bead_density_contour.png" width="400" height="300" border="0"/>';
							print '</a>';
							print '</div>';
						}	
					?>
		
	</div><!-- end wrapper -->
		
</div>

<!-- stop ion sphere -->

<!--  Start Report Information -->
<div id='ReportInfo' class="report_block">
	<h2>Report Information</h2>
	
	<div><!-- start wrapper -->
		<h3>Analysis Info</h3>
		
		<div>
			<table class='noheading'>
				<col width="325px" />
				<col width="520px" />
				<?php
					$process = parse_to_keys("processParameters.txt");
					$is_td = true;
					if($meta){
						foreach($meta as $row){
							print"<tr>";
							foreach($row as $x){
								print ($is_td) ? "<th>" : "<td>";
								print "$x";
								print ($is_td) ? "</th>" : "</td>";
								$is_td = !$is_td;
							}
							print "</tr>";
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
				?>
			</table>
		</div>
		
		<h3>Software Version</h3>
			<div>
				<table class='noheading'>
				<col width="325px" />
				<col width="520px" />
				<?php
					$version = parseVersion("version.txt");
					$is_td = true;
					$metaDone = 0;
					if($version){
						foreach($version as $row){
							print"<tr>";
							foreach($row as $x){
								print ($is_td) ? "<th>" : "<td>";
								
								if ($metaDone == 0 || $metaDone == 1){
									$x = str_replace("_"," ",$x);
									print "<strong>" . $x . "</strong>";
									$metaDone += 1;
								}else{
									print $x;
								}
								
								print ($is_td) ? "</th>" : "</td>";
								$is_td = !$is_td;
							}
							print "</tr>";
						}
					}
				?>
				</table>
			</div>
	</div><!--  end wrapper -->

</div>
<!-- stop report info -->


<!-- start of file -->
			<?php 
			print '<div id="FileLinks" class="report_block"><h2>File Links</h2><div><table class="noheading">';

            if(file_exists("expMeta.dat")){
                $postfixsff = $base_name . '.sff' . '.zip';
                $tf_postfixsff = $base_name . '.tf.sff' . '.zip';
                $bamfile = $base_name . '.bam';
                $baifile = $base_name . '.bam.bai';
                $postfixsff_sampled = $base_name . '.sampled.sff';
                $postfixfastq = $base_name . '.fastq';
                $postfixfastq_sampled = $base_name . '.sampled.fastq';
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
				if (!$align_full){  
					print "<tr><td><a href='$postfixsff'>Library Sequence (SFF) </a> </td> </tr>";
					print "<tr><td><a href='$postfixsff_sampled'>Sampled Library Sequence (SFF) </a> </td> </tr>";
				}else{
					print "<tr><td><a href='$postfixsff'>Library Sequence (SFF) </a> </td> ";
					print "</tr>";
				}
				
				if (!$align_full){
					print "<tr><td><a href='$postfixfastq'>Library Sequence (FASTQ) </a> </td></tr>";
					print "<tr><td><a href='$postfixfastq_sampled'>Sampled Library Sequence (FASTQ) </a> </td> </tr>";
				}else{
					print "<tr><td><a href='$postfixfastq'>Library Sequence (FASTQ) </a> </td>";
					print "</tr>";
				}
				
				if (!$align_full){
					print "<tr><td><a href='$bamfile'>Sampled Library Alignments (BAM)</a ></td> </tr>";
					print "<tr><td><a href='$baifile'>Sampled Library Alignments (BAI)</a ></td> </tr>";
				}else{
					print "<tr><td><a href='$bamfile'>Full Library Alignments (BAM)</a ></td> </tr>";
					print "<tr><td><a href='$baifile'>Full Library Alignments (BAI)</a ></td> </tr>";
				}
                
                if (file_exists("alignment_barcode_summary.csv")) {
				//if there are barcodes zip the bam and bai files
				
				print "<tr><td><a href='alignment_barcode_summary.csv'>Barcode Alignment Summary</a ></td> </tr>";
				print "<tr><td><a href='$base_name.barcode.bam.zip'>Barcode-specific Library Alignments (BAM)</a></td> </tr>";
				print "<tr><td><a href='$base_name.barcode.bai.zip'>Barcode-specific Library Alignments Index Files (BAI)</a></td> </tr>";
				print "<tr><td><a href='$base_name.barcode.fastq.zip'>Barcode-specific Library Sequence (FASTQ)</a></td> </tr>";
				print "<tr><td><a href='$base_name.barcode.sff.zip'>Barcode-specific Library Sequence (SFF)</a></td> </tr>";
			}
				
				print "<tr><td><a href='$tf_postfixsff'>Test Fragments (SFF)</a> </td> </tr> ";
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
                    print "<tr><td><a href='drmaa_stdout.txt'>Developer link: Super std_out_err</a> </td> </tr>";
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
				$.each(data.pluginState, function(key,value){
					var row = "";
					//write the plugin name
					row += "<tr><th>";
					row += key;
					row += "</th><th>";
					//write the job status
					row += value;
					row += '<span style="float: right;"><a class="pluginLog ui-icon ui-icon-script" data-title="Log for ' + key +'" title="Log File for ' + key + '" href="plugin_out/' + key + '_out/drmaa_stdout.txt">log</a></span>' ;
					row += "</th></tr>";
					//for each file make a new row under the plugin name
					//if it ends with _block.html or _block.php then render that to a div inline
					row += '<tr colspan=2 id="' + key + '">';
					if (data.pluginFiles[key]){
						$.each(data.pluginFiles[key], function(i, file){
							if (file.endsWith("_block.html") || file.endsWith("_block.php") ){
								row += '<tr ><td  colspan=2 style="padding: 0px;"><div class="pluginLevel1">';
								row += '<iframe id="'+key+'" class="pluginBlock" src="plugin_out/' + key + '_out/' + file+'" width="95%" frameborder="0" height="0px" ></iframe>';
								row += "</div></td></td>";
							}else{
								row += '<tr ><td  colspan=2 style="padding: 0px;">';
								row += '<div class="pluginLevel1">&#8227; <a href=plugin_out/' + key + '_out/' + file + '>';
								row += file;
								row += "</a></div></td></td>";
							}
						});
					}
					row += '</tr>';		
					$("#pluginStatusTable").append(row);			
					$('.pluginLog').tipTip({ position : 'left' });
					
					
					
				});
			},
			error: function(msg) {
				$("#pluginStatusTable").text("Failed to get Plugin Status");
			}
		});
		
	    
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
		print 'djangoURL = "/rundb/api/v1/results/' . $djangoPK["ResultsPK"] .'/plugin/";';
		    
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
				
				$("#pluginList").html('<ul id="pluginUL"></ul>');
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
                $("#pluginUL").append('<li data-id="' + val.id + '" class="plugin_input_class" id="' + val.name + '_plugin"><a href="'+data.input+'?report='+djangoPK+'" class="plugin_link colorinput">' + val.name + '</a></li>');
              }else{
                $("#pluginUL").append('<li data-id="' + val.id + '" class="plugin_class" id="' + val.name + '_plugin"><span class="plugin_link">' + val.name + '</span></li>');
              }
            }
					},
					async : false,
					type : 'GET',
					dataType : 'json'
				});
				
				$(".colorinput").colorbox({width:"90%", height:"90%", iframe:true})
				$("#pluginLoad").text("Click a plugin to run:");
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
