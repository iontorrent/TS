<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<?php include ("parsefiles.php"); ?>
<head>
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type" />
    <?php
        echo"<title>";
            $meta = parseMeta("expMeta.dat");
        	$resultsName = $meta[2][1];
        echo$resultsName;
        echo"</title>";
    ?>
    <link rel="stylesheet" type="text/css" href="/site_media/stylesheet.css"/>
    <script type="text/javascript" src="/site_media/scripts.js"></script>
</head>

<script language="javascript">
<!--
  function toggleDiv(divid){
    if(document.getElementById(divid).style.display == 'none'){
      document.getElementById(divid).style.display = 'block';
    }else{
      document.getElementById(divid).style.display = 'none';
    }
  }
//-->
</script>

<body>
<div class="controlholder centered">
    <div class="roundedcornr_box_526903">
    <div class="roundedcornr_top_526903"><div></div></div>
    <div class="roundedcornr_content_526903">
    <div id="outer" style="height:100%; width:100%; font-family:tahoma;min-width:1024px;">
    <div class="rtopbar">
        <div class="logoholder">
            <img src="/site_media/images/logo_medium.png" width="150"
            alt="IonTorrent Systems, Inc."/>
        </div>
        <div class="tabholder">
            <div class="tab">
                <a href="/rundb/">Runs</a>
            </div>
     
            <div class="tab">
                <a href="/rundb/reports">Reports</a>
            </div>
            <div class="clear"></div>
        </div><!--end tabholder-->
    </div><!--end rtopbar-->

    <div id='inner' style="width:960px; margin-left:auto; margin-right:auto; height:100%;">
        <br/>
        <h2><center>Analysis Summary</center></h2>
        <div id='AnalysisInfo' style="border:thin black solid; background-color:#FFFFFF; padding:10px;">
            <h3>File Links</h3>
            <div id='fileLinks' style='float:left'>
                <a href="log.html">RUN LOG</a>&nbsp;&nbsp;
            </div><!--endfile links-->
            <div style='clear:both;'></div>

            <h3>Analysis Info</h3>
            <div id='analysisInfo' style='float:left'>
                <table border=0 cellpadding=5>
                <?php
                    $meta = parseMeta("expMeta.dat");
                    if($meta){
                        foreach($meta as $row){
                            echo"<tr>";
                            foreach($row as $x){
                                echo "<td>$x</td>";
                            }
                            echo "</tr>";
                        }
                    }
                    $process = parse_to_keys("processParameters.txt");
					if ($process){
						print "<tr>";
						print "<td>Flow Order</td>";
						print "<td>";
						print $process["flowOrder"];
						print "</td>";
						print "</tr>";
					}
                ?>
                </table>
            </div>
            <div style='clear:both;'></div>

            <h3>Software Version</h3>
            <div id='softwareVersion' style='float:left'>
                <table border=0 cellpadding=0>
                <?php
                    $version = parseVersion("version.txt");
                    if($version){
                        foreach($version as $row){
                            echo"<tr>";
                            foreach($row as $x){
                                echo "<td>".ucwords($x)."</td>";
                            }
                            echo "</tr>";
                        }
                    }
                ?>
                </table>
            </div>
            <div style='clear:both;'></div>

            <h3>Beadfind Summary</h3>
            <div id='beadTable' style='float:left'>
                <table border=0>
                <?php
                    $bf = parseBeadfind("bfmask.stats");  
                    if($bf){                
                        foreach($bf as $row){
                            echo "<tr>";
                            echo "<td>$row[0]</td>";
                            echo "<td><div style='text-align:right'>$row[1]</div></td>";
                            echo "</tr>";
                        }
                    }                   
                ?>      
                </table>
            </div>
            <div id='beadHeatMap' style='float:right'>
                <a href='Bead_density_contour.png'><img src='Bead_density_contour.png' width=400 height=300 border=0/></a>
            </div>
            <div style='clear:both;'></div>
            
        </div>
        <br/>
        <!--Start Library Summary-->
        <h2><center>Library Summary</center></h2>
        <div id='LibrarySummary' style="border:thin black solid; background-color:#FFFFFF; padding:10px;">
            <h3>File Links</h3>
            <div id='fileLinks' style='float:left'>
            <?php 
                    	$as = parseAlignmentSummary("alignment.summary");
                        //check to see if a full or sampled alignment was done
						$align_full = true;
	                    if (isset($as['Total number of Sampled Reads'])) {
							$align_full = false;
						}else{
							$align_full = true;
						}
        	        $meta = parseMeta("expMeta.dat");
        	        $expName = $meta[0][1];
        	        $resultsName = $meta[2][1];
        	        $postfixsff = $expName . '_' . $resultsName . '.sff';
                    $postfixfastq = $expName . '_' . $resultsName . '.fastq';
                    $tf_postfixsff = $expName . '_' . $resultsName . '.tf.sff';
                    $base_name = $expName . '_' . $resultsName;
                    
                    //output file links
        	        echo "<a href='$postfixsff.zip'>SFF</a>&nbsp;&nbsp;";
        	        if (!$align_full){
	        	        echo "<a href='$postfixsff_sampled.zip'>SFF (sampled)</a>&nbsp;&nbsp;";
        	        }
                    echo "<a href='$postfixfastq.zip'>FASTQ</a>&nbsp;&nbsp;";
        	        if (!$align_full){
	                    echo "<a href='$postfixfastq_sampled.zip'>FASTQ</a>&nbsp;&nbsp;";
        	        }

        	   ?>
            </div><!--endfile links-->
            <br/>

            <h3>Alignment Summary</h3>
            <div id='alignmentSummary' style='float:left'>
                <table border=0>
                <?php
                    if($as){
                        foreach($as as $row){
                            echo"<tr>";
                            foreach($row as $x){
                                echo "<td>$x</td>";
                            }
                            echo "</tr>";
                        }
                }elseif( file_exists("alignment.error")){
					//If there was an alignment error print that info here.
                	print "<tr><td>";
					print '<p>There was an alignment error. For details see the <a href="log.html">Report Log</a></p>';
					$alignerror = "alignment.error";
					$fh = fopen($alignerror, 'r');
					$alignErrorData= fread($fh, filesize($alignerror));
					fclose($fh);
					print "<p><pre>";
					print $alignErrorData;
					print "</pre></p>";
					print "</td></tr>";
					
                 }else{
                        echo"<tr><td>No Alignments Found</td></tf>";
                 }
                 
                ?>
                </table>
            </div>
            <div style='clear:both;'></div> 
            
            <?php 
                if(file_exists("Region_CF_heat_map_LIB.png") or file_exists("Region_IE_heat_map_LIB.png")){
                    echo "<h3>Heat Maps</h3>";
                        echo "<div id='HeatMaps' style='float:left;'>";
                        echo "<table border=0 style='float:left;'>";
                            echo "<thead>";
                                echo "<tr><th><div style='width:100%; text-align:center;'>CF Heat Map</div></th>";
                                echo "<th><div style='width:100%; text-align:center;'>IE Heat Map</div></th>";
                                echo "<th><div style='width:100%; text-align:center;'>DR Heat Map</div></th>";
                                echo "</tr>";
                                echo "</thead>";
                                echo "<tbody>";
                                    echo "<tr><td><a href='Region_CF_heat_map_LIB.png'><img src='Region_CF_heat_map_LIB.png' width=300 height=225 border=0/></a></td>"; 
                                    echo "<td><a href='Region_IE_heat_map_LIB.png'><img src='Region_IE_heat_map_LIB.png' width=300 height=225 border=0/></a></td>";
                                    echo "<td><a href='Region_DR_heat_map_LIB.png'><img src='Region_DR_heat_map_LIB.png' width=300 height=225 border=0/></a></td>";
                                    echo "</tr>";
                                echo "</tbody>";
                    echo "</table>";
                    echo "</div>";
                    echo "<div style='clear:both;'></div>";
               }
            ?>

            <h3>Library Analysis Graphs</h3>
            <div id='AlignHistograms' style='float:left'>
                <table border=0>
                    <tbody>
                        <?php 
                            if(file_exists('iontrace_Library.png'))
                            {
                                echo "<tr>";
                                echo "<td><a href='iontrace_Library.png'><img src='iontrace_Library.png' width=450 height=225 border=0/></a></td>";
                                echo "</tr>";
                            }
                            if(file_exists('Filtered_Alignments_Q10.png') and file_exists('Filtered_Alignments_Q17.png'))
                            {
                                echo "<tr>";
                                echo "<td><a href='Filtered_Alignments_Q10.png'><img src='Filtered_Alignments_Q10.png' width=450 height=225 border=0/></a></td>";
                                echo "<td><a href='Filtered_Alignments_Q17.png'><img src='Filtered_Alignments_Q17.png' width=450 height=225 border=0/></a></td>";
                                echo "</tr>";
                            }
                        ?>
                    </tbody>
                </table>
            </div>
            <div style='clear:both;'></div> 

            <h3>Filter Metrics</h3>
            <div id='FilterTable' style='float:left'>
                <?php
                    $f = parseFilter("filterMetrics.txt");  
                    if($f){ 
                        $fKey = array_keys($f);         
                        echo "<table border=0 cellpadding=2.5>";
                        foreach($fKey as $k){
                            $pos = strpos($k, "TF");
                            if($pos === false)
                            {
                                echo "<tr>";
                                echo "<td>$k</td>";
                                echo "<td>";
                                echo $f[$k];
                                echo "</td>";
                                echo "</tr>";
                            }
                        }
                        echo "</table>";
                    }                   
                ?>  
            </div>
            <div style='clear:both;'></div>

        </div><!--End Library Summary-->
            
        <br/>
        <!--                 -->
        <!--Start TF Summary -->
        <!--                 -->
        <h2><center>Test Fragment Summary</center></h2>
        <div id='TFSummary' style="border:thin black solid; background-color:#FFFFFF; padding:10px;">
            <h3>File Links</h3>
            <div id='fileLinks' style='float:left'>
            <?php 
                  echo "<a href='$tf_postfixsff.zip'>SFF</a>&nbsp;&nbsp;";
            ?>
            </div><!--endfile links-->
            <div id='rawTrace' style='float:right'>
                <a href="iontrace_Test_Fragment.png"><img src="iontrace_Test_Fragment.png" width=450 height=225 border=0/></a>
            </div>  
            <div style='clear:both;'></div>
            
            <h3>Filter Metrics</h3>
            <div id='FilterTable' style='float:left'>
                <?php
                    $f = parseFilter("filterMetrics.txt");  
                    if($f){ 
                        $fKey = array_keys($f);         
                        echo "<table border=0 cellpadding=2.5>";
                        foreach($fKey as $k){
                            $pos = strpos($k, "Library");
                            if($pos === false)
                            {
                                echo "<tr>";
                                echo "<td>$k</td>";
                                echo "<td>";
                                echo $f[$k];
                                echo "</td>";
                                echo "</tr>";
                            }
                        }
                        echo "</table>";
                    }                   
                ?>  
            </div> 
            <div style='clear:both;'></div>
            
            <!--begin per TF metrics -->
            <?php
                $tf = parseTfMetrics("TFMapper.stats");
                $cafie = parseCafieMetrics("cafieMetrics.txt"); 
                if($tf and $cafie)
                {
                    $arrKeys = array_keys($tf);
                    $cafieKeys = array_keys($cafie);
                    $tfKeys;
                    foreach($arrKeys as $k)
                    {
                        if($k != 'system' and $k != ' '){
                            $tfKeys[]=$k;
                        }
                    }
                    
                    foreach($tfKeys as $TF)
                    {
                        echo"<h3>Test Fragment - $TF</h3>";
                        echo"<h5>Quality Metrics</h5>";
                        echo"<div id=$TF style='float:left';>";
                            $localKey = array_keys($tf[$TF]);
                            $rawHP = Null;
                            $corrHP = Null;
                            $count = count($rawHP);
                            $range = range(0,$count);
                        
                            echo "<table border=0 cellpadding='5'>";
                            foreach($localKey as $metric)
                            {
                                if($metric == "Corrected HP SNR")
                                {
                                    $corrHP = $tf[$TF][$metric];
                                }
                                elseif($metric == "Raw HP SNR")
                                {
                                    $rawHP = $tf[$TF][$metric];
                                }
                                else
                                {   
                                    echo "<tr>";
                                    echo "<td style='white-space: nowrap;'>$metric</td>";
                                    echo "<td>";
                                    echo $tf[$TF][$metric];
                                    echo "</td>";
                                    echo "</tr>";
                                }
                            }
                        echo "</table>";
                        echo"</div>";
                        echo"<div style='clear:both;'></div>";

                        ###################
                        ### CAFIE Table ###
                        ###################
                        echo"<h5>CAFIE Metrics</h5>";
                        echo"<div id=$TF style='float:left'>";
                            echo "<table border=1 cellpadding='5'>";
        
                            if(array_key_exists($TF, $cafie))
                            {
                                $cafieKey = array_keys($cafie[$TF]);
                                foreach($cafieKey as $cM)
                                {
                                    if($cM != 'Avg Ionogram' and $cM != 'Estimated TF' and $cM != 'Corrected Avg Ionogram')
                                    {
                                        echo "<tr>";
                                        echo "<td>$cM</td>";
                                        echo "<td>";
                                        echo $cafie[$TF][$cM];
                                        echo "</td>";
                                        echo "</tr>";
                                    }
                                }
                            }
                            else
                            {
                                echo "<tr>";
                                echo "<td>"."No Cafie Metrics Found"."</td>";
                                echo "</tr>";
                            }
                            echo "</table>";
                        echo"</div>";
                        echo"<div style='clear:both;'></div>";

                        echo"<h5>Raw HP SNR</h5>";
                        echo"<div id=$TF style='float:left'>";
                            echo "<table border=1 cellpadding='5'>";
                            foreach($localKey as $metric)
                            {
                                if ($metric == "Raw HP SNR")
                                {   
                                    echo "<tr>";
                                    $span = $count-1;
                                    echo "<td colspan=$count>$metric</td>";
                                    echo "<tr>";
                                    $c = 0;
                                    foreach($rawHP as $snr)
                                    {   
                                        $string = $c."-mer";
                                        echo "<td>$string</td>";                          
                                        $c = $c + 1;
                                    }
                                    echo "<tr>";
                                    foreach($rawHP as $snr)
                                    {
                                        echo "<td>$snr</td>";
                                    }                  
                                    echo "</tr>";     
                                }
                            }
                            echo "</table>";
                        echo"</div>";
                        echo"<div style='clear:both;'></div>";

                        echo"<h5>Corrected HP SNR</h5>";
                        echo"<div id=$TF style='float:left'>";
                            echo "<table border=1 cellpadding='5'>";
                            foreach($localKey as $metric)
                            {
                                if ($metric == "Corrected HP SNR")
                                {   
                                    echo "<tr>";
                                    $span = $count-1;
                                    echo "<td colspan=$count>$metric</td>";
                                    echo "<tr>";
                                    $c = 0;
                                    foreach($corrHP as $snr)
                                    {   
                                        $string = $c."-mer";
                                        echo "<td>$string</td>";                          
                                        $c = $c + 1;
                                    }
                                    echo "<tr>";
                                    foreach($corrHP as $snr)
                                    {
                                        echo "<td>$snr</td>";
                                    }                  
                                    echo "</tr>";     
                                }
                            }
                            echo "</table>";
                        echo"</div>";
                        echo"<div style='clear:both;'></div>";
                        echo"<br\>";
                    
                        echo"<h5>Q17 Ionograms</h5>";
                        echo"<div id=$TF style='float:left; width:100%';>";
                            echo"<a href='tf_ionograms_$TF.html'>Top TF Ionograms</a>";
                            #REMOVED FOR FASTER PAGE LOADING
                            #echo"<a href='javascript:;' onclick=\"toggleDiv('TopBeads_$TF');\">Top TF Ionograms</a>";
                            #echo"<div id='TopBeads_$TF' style='border:thin black solid; background-color:#FFFFFF; padding:10px; display:none;'>";
                                #echo"<iframe src = 'tf_ionograms_$TF.html' width=100% height=500px frameborder='0'></iframe>";
                            #echo"</div>";    
                        echo"</div>";
                        echo"<div style='clear:both;'></div>";
                        if(file_exists("Region_CF_heat_map_$TF.png") or file_exists("Region_IE_heat_map_$TF.png")){
                            echo "<h3>Heat Maps</h3>";
                                echo "<div id='HeatMaps' style='float:left;'>";
                                echo "<table border=0 style='float:left;'>";
                                    echo "<thead>";
                                        echo "<tr><th><div style='width:100%; text-align:center;'>CF Heat Map</div></th>";
                                        echo "<th><div style='width:100%; text-align:center;'>IE Heat Map</div></th>";
                                        echo "<th><div style='width:100%; text-align:center;'>DR Heat Map</div></th>";
                                        echo "</tr>";
                                    echo "</thead>";
                                    echo "<tbody>";
                                        echo "<tr><td><a href='Region_CF_heat_map_$TF.png'><img src='Region_CF_heat_map_$TF.png' width=300 height=225 border=0/></a></td>"; 
                                        echo "<td><a href='Region_IE_heat_map_$TF.png'><img src='Region_IE_heat_map_$TF.png' width=300 height=225 border=0/></a></td>";
                                        echo "<td><a href='Region_DR_heat_map_$TF.png'><img src='Region_DR_heat_map_$TF.png' width=300 height=225 border=0/></a></td>";
                                        echo "</tr>";
                                    echo "</tbody>";
                                echo "</table>";
                                echo "</div>";
                                echo "<div style='clear:both;'></div>";
                        }
                        echo"<h5>Graphs</h5>";
                        echo"<div id='alignment' style='float:right'>";                                                       
			echo"<a href='Raw signal overlap_$TF.png'><img src='Raw signal overlap_$TF.png' width=450 height=225 border=0/></a>";
			echo"<a href='Corrected signal overlap_$TF.png'><img src='Corrected signal overlap_$TF.png' width=450 height=225 border=0/></a>";
			echo"<a href='perHPAccuracy_$TF.png'><img src='perHPAccuracy_$TF.png' width=450 height=225 border=0/></a>";
			echo"<a href='Match-Mismatch_$TF.png'><img src='Match-Mismatch_$TF.png' width=450 height=225 border=0/></a>";
			echo"<a href='Q10_$TF.png'><img src='Q10_$TF.png' width=450 height=225 border=0/></a>";
			echo"<a href='Q17_$TF.png'><img src='Q17_$TF.png' width=450 height=225 border=0/></a>";
			echo"<a href='Average Raw Ionogram_$TF.png'><img src='Average Raw Ionogram_$TF.png' width=450 height=225 border=0/></a>";
			echo"<a href='Average Corrected Ionogram_$TF.png'><img src='Average Corrected Ionogram_$TF.png' width=450 height=225 border=0/></a>";
                        echo"</div>";
                        echo"<div style='clear:both;'></div>";
                    }
                }
            ?>
            
        </div> <!--End TF Summary Div-->
        <br/>
		
        <!--Start PGM Diagnostics Div-->
        <h2><center>PGM Diagnostics</center></h2>
        <div id='pgmDiagnostics' style="border:thin black solid; background-color:#FFFFFF; padding:10px;">
        	<?php
            /*	Enable error reporting
            error_reporting(E_ALL);
			ini_set('display_errors', '1');
			*/
            
            $process = parse_to_keys("processParameters.txt");
			if ($process)
            {
            	$rawpath = $process["dataDirectory"];
            	
                /*---	Copy files from raw data folder to local results folder ---*/
                $files=array('RawInit.txt','InitLog.txt','RawInit.jpg');
                for ($i=0;$i < sizeof($files); $i++){
                	$src = $rawpath . '/' . $files[$i];
                    if (file_exists($src)){
                		copy ($src, $files[$i]);
                    }
                }
                
                /*---	Create display html 	---*/
            	echo"<table border=1>";
                echo"<tr><th>pH Image</th><th>Data Files</th></tr>";
                echo"<tr>";
                echo"<td>";
                
                // start image display (source image is 800x600 px)
                
				$image = "RawInit.jpg";
                
            	if (file_exists($image))
                {
        			echo"<a href='$image'><img src='$image' width=400 height=300 border=0/></a>";
                }
                else
                {
                	//Legacy PGM code generated a bitmap...
                	$bitmap = $rawpath . "/RawInit.bmp";
                    if (file_exists($bitmap))
                    {
                		system("which /usr/bin/convert >/dev/null && /usr/bin/convert $bitmap $image");
                    	echo"<a href='$image'><img src='$image' width=400 height=300 border=0/></a>";
                    }
                }
                // end of image display
                echo"</td>";
                
                // start list of file links
                echo"<td align='left' valign='top'>";
                	echo"<a href='RawInit.txt'>RawInit.txt</a><br>";
                	echo"<a href='InitLog.txt'>InitLog.txt</a>";
                echo"</td>";
                // end list of file links
                echo"</tr>";
                echo"</table>";
            }
            else
            {
            	echo "No PGM data found";
            }
        	?>
        </div><!--End PGM Diagnostics Div-->
        
        <h2><center>Plugin Summary</center></h2>
        <div id='pluginSummary' style="border:thin black solid; background-color:#FFFFFF; padding:10px;">
            <?php
                $folders = dir_list('plugin_out');
                if (count($folders) > 0)
                {
                    foreach($folders as $path)
                    {
                        $links = parsePlugins($path);
                        if ($links)
                        {
                            echo "<table>";
                            foreach($links as $link)
                            {
                                echo"<tr>";
                                echo"<td><a href='./plugin_out/$path/$link'>$link</a></td>";
                                echo"</tr>";
                            }
                            echo"</table>";
                        } 
                    }
                }   
                else
                {
                    echo "<table>";
                    echo"No links found";
                    echo"</table>";
                }
            ?>
        </div><!--end plugin summary -->
    </div><!--end inner-->
</div><!--end outer-->

<br/>
</div>
<div class="roundedcornr_bottom_526903"><div></div></div>
</div>
</div>
<div id='end' class="footer">
    <a href=http://www.iontorrent.com/support>Request Support</a>&nbsp;|&nbsp;<a href=/ion-docs/Home.html>Help/Documentation</a>&nbsp;|&nbsp;<a href=/licenses/terms-of-use.txt>Terms of Use</a><br/>
    Copyright&nbsp;&copy; 2010 <a href="http://www.iontorrent.com/">Life Technologies, Inc.</a>
</div>
</body>
</html>
