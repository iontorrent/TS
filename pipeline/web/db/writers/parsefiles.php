<?php
function parseBeadfind($fileIn)
{
  $data = NULL;
  if(file_exists($fileIn))
    {
      $file = fopen($fileIn, 'r') or die();
      //Output a line of the file until the end is reached
      $data = array();
      while(($line = fgets($file)) !== false)
	{
	  list($key, $value) = preg_split('/=/', $line); 
	  $data[] = array($key,$value);
	}
      fclose($file);	
    }
  return ($data); 
}

function alignQC_noref()
{
	$fileIn = "ReportLog.html";	
    if(file_exists($fileIn))
    {
        $file = fopen($fileIn, 'r') or die();	
        while(($line=fgets($file)) !== false)
        {	
            if (strncmp($line, "ERROR: /usr/local/bin/alignmentQC.pl: unable to find reference genome tmap-f3/", strlen("ERROR: /usr/local/bin/alignmentQC.pl: unable to find reference genome tmap-f3/")) == 0){
            	$break_line = explode("tmap-f3/", $line);
            	return $break_line[1];
            }
        }

        fclose($file);
    }
    return(false);
    
}

function parseAlignmentSummary($fileIn)
{
	$data = NULL;
    if(file_exists($fileIn))
    {
        $file = fopen($fileIn, 'r') or die();	
        $data = array();
        $assem = array();
        while(($line=fgets($file)) !== false)
        {	
            list($key,$value) = preg_split('/=/', $line);
            $data[] = array($key,$value);
        }

        fclose($file);
    }
    return($data);
}

//make a dict out of the progress file
function parseProgress($fileIn)
{
	$progress_task_name = array( "wellfinding" => "Well Characterization",
		"signalprocessing" => "Signal Processing",
		"basecalling" => "Basecalling",
		"alignment" => "Aligning Reads"
		);
		
	$progress_status_meaning = array( "green" => "green",
		"yellow" => "In Progress",
		"grey" => "Not Yet Started"
		);
		
	$data = NULL;
    if(file_exists($fileIn))
    {
        $file = fopen($fileIn, 'r') or die();	
        $data = array();
        $assem = array();
        while(($line=fgets($file)) !== false)
        {	
            list($key,$value) = explode("=", $line);
			$full_name = $progress_task_name[trim($key)];
			if (trim($value) != "green"){
            	$data[] = array($full_name, $progress_status_meaning[trim($value)]  );
			}
        }

        fclose($file);
    }
    return($data);
}

function parseKeypass($fileIn)
{
  $ret = NULL;
  if(file_exists($fileIn))
    {
      $file = fopen($fileIn, 'r') or die();	
      $data = array();
      
      while(($line=fgets($file)) !== false)
	{
	  $data[] = $line;
	}
      fclose($file);
      $ret = array_slice($data, -4,4);
    }
  
  if($ret == Null){
    $ret=False;
  }
  return($ret);
}

function parseTfMetrics($fileIn)
{
	$ret = NULL;
  if(file_exists($fileIn)){
    $file = fopen($fileIn, 'r') or die();	
    $data = array();
    $currentKey = Null;
    while(($line = fgets($file)) !== false){
      
      $containsComment = strpos($line, "#");
      $containsTop = strpos($line,"Top");
      if($containsComment === false and $containsTop === false and strlen($line)!=0){
	list($key,$value) = preg_split('/=/', $line); 
	$key = trim($key);
	$value = trim($value);
	if($key!="Match-Mismatch" and $key!="Corrected signal overlap" and $key!="Raw signal overlap" and $key!="Match-Mismatch" and $key!="Q10" 
	   and $key!="Q17" and $key != "Per HP accuracy" and $key !="50Q10A" and $key !="50Q17A" and $key !='TransferPlot'){
	  if($key=="TF Name"){
	    $data[$value] = array();
	    $currentKey = $value;
	  }
	  if($key=="Corrected HP SNR" or $key=="Raw HP SNR"){
	    $ar=explode(",", $value);
	    $snr = array();
	    foreach($ar as $x){
	      $p = explode(":", $x);
	      $snr[]=$p[1];
	    }
	    $data[$currentKey][$key]=$snr;
	  }   
	  else{
	    $data[$currentKey][$key]=$value;
	  }
	}
      }
    }
#print_r($data);
    fclose($file);
    $ret = $data;
  }
  
  if($ret == Null)
    {
      $ret=False;
    }
  return($ret);
}    

function parseFilter($fileIn)
{
	$ret = NULL;
  if(file_exists($fileIn))
    {
      $file = fopen($fileIn, 'r') or die();
      //Output a line of the file until the end is reached
      $data = array();
      while(($line = fgets($file)) !== false)
	{
	  list($key, $value) = preg_split('/=/', $line);
	  $key = trim($key);
	  $value = trim($value); 
	  $data[$key] = $value;
	}
      fclose($file);	
#print_r($data);
      $ret = $data;
    }
  if($ret == Null){
    $ret=False;
  }
  return ($ret); 
}

function parseCafieMetrics($fileIn)
{
	$ret = NULL;
  if(file_exists($fileIn))
    {
      $file = fopen($fileIn, 'r') or die();	
      $data = array();
      $currentKey = Null;
      while(($line = fgets($file)) !== false)
	{
	  $containsComment = strpos($line, "#");
	  if($containsComment === false)
	    {
	      list($key,$value) = preg_split('/=/', $line); 
	      $key = trim($key);
	      $value = trim($value);
	      if($key=="TF")
		{
		  $data[$value] = array();
		  $currentKey = $value;
		}
	      elseif(strpos($key,"System") == true)
		{   
		  if($currentKey != "system")
		    {
		      $currentKey = "system";
		      $data[$currentKey] = array();
		    }
		  
		  $data[$currentKey][$key]=$value;
		}
	      else
		{
		  $data[$currentKey][$key]=$value;
		}
	      
	    }
	}
#print_r($data);
      fclose($file);
      $ret = $data;
    }
  if($ret == Null){
    $ret=False;
  }
  return($ret);
}
//parse meta data from run
function parseMeta($fileIn)
{
	$ret = NULL;
  if(file_exists($fileIn))
    {
      $file = fopen($fileIn, 'r') or die();
      //Output a line of the file until the end is reached
      $data = array();
      while(($line = fgets($file)) !== false)
	{
	  list($key, $value) = preg_split('/=/', $line); 
	  $data[] = array(trim($key),trim($value));
	}
#print_r($data);
      fclose($file);
      $ret = $data;	
    }
  if($ret == Null){
    $ret=False;
  }
  return ($ret); 
}    

//parse meta data from run
function parseVersion($fileIn)
{
    if(file_exists($fileIn))
    {
        $file = fopen($fileIn, 'r') or die();
        //Output a line of the file until the end is reached
        $data = array();
        while(($line = fgets($file)) !== false)
        {
            list($key, $value) = preg_split('/=/', $line);
            $data[trim($key)] = trim($value);
        }
        fclose($file);
    }
    // This is a little redundant since Null and False both evaluate falsely?
    return $data ?: False;
}

function parsePlugins($folderIn)
{
	$ret = NULL;
  if ($handle = opendir('./plugin_out/'.$folderIn))
    {
      $link_list = array();
      while(false !== ($file = readdir($handle)))
	{
	  if (strpos($file, 'html'))
	    {
	      $link_list[] = $file;
	    }
	}
      closedir($handle);
      $ret=$link_list;
    }
  if ($ret == Null){
    $ret=False;
  }
  return ($ret);
}

function dir_list($d){
	$l = NULL;
  foreach(array_diff(scandir($d),array('.','..')) as $f)if(is_dir($d.'/'.$f))$l[]=$f;
  return $l;
}

function joinPaths() {
  $args = func_get_args();
  $paths = array();
  foreach ($args as $arg) {
    $paths = array_merge($paths, (array)$arg);
  }
  foreach ($paths as &$path) {
    $path = trim($path, '/');
  }
  return join('/', $paths);
}

function parse_to_keys($fileIn){
	$ret = NULL;
    if(file_exists($fileIn))
    {
        $file = fopen($fileIn, 'r') or die();
        //Output a line of the file until the end is reached
        $data = array();
        while (($line = fgets($file)) !== false)
        {
            #ignore lines starting with a square bracket '[', e.g.[global]
            if (preg_match('/^\[/', $line)) { 
               #print_r($line);
               continue;
            }
            # ignore empty lines
            if (preg_match('/^\n/', $line)) {
               #print_r($line);
               continue;
            }
            list($key, $value) = preg_split('/=/', $line);
            $key = trim($key);
            $value = trim($value); 
            $data[$key] = $value;
        }
        fclose($file);	
        #print_r($data);
        $ret = $data;
    }

    if($ret == Null){
	    $ret=False;
    }

    return ($ret); 
}

function tabs_parse_to_keys($fileIn){
	//this function is exclusively for alignTable.txt
	$arr = NULL;
	$row = 1;

	if (($handle = fopen($fileIn, "r")) !== FALSE) {
		$arr = array();
	    while (($data = fgetcsv($handle, 1000, "\t")) !== FALSE) {
	        $num = count($data);
	        $row++;
			if ($row != 2){
				$rowsum = 0;
				//find the sum of all the errs. If no errors, don't include the row in the table
				$err1 = $data[7];
				$err2 = $data[8];
				$err3 = $data[9];
				$allErrSum = $err1 + $err2 + $err3;
				$SUMerr = $err2 + $err3;
				
				if ($allErrSum != 0){
					//don't include the last 4 cols - which are 3+ err and blank
		        	for ($c=0; $c < $num - 2 ; $c++) {
						$arr[$row-2][] = $data[$c];
			        }

					$arr[$row-2][] = $SUMerr;
				}
			}
	    }
	    fclose($handle);
	}
	return $arr;
}


function parse_beadSummary($fileIn){
	$arr = NULL;
	$row = 1;

	if (($handle = fopen($fileIn, "r")) !== FALSE) {
		$arr = array();
	    while (($data = fgetcsv($handle, 1000, "\t")) !== FALSE) {
	        $num = count($data);
	        $row++;
			if ($row != 2){
		        for ($c=0; $c < $num  ; $c++) {
						$arr[$row-2][] = $data[$c];
				}
			}
	    }
	    fclose($handle);
	}
	return $arr;
}

?>

<?php 
//if progress is set in the query string, return a json object which can be used to update the loading block
if (isset($_GET["progress"])){
	
	$progress_task_name = array( "wellfinding" => "Well_Characterization",
		"signalprocessing" => "Signal_Processing",
		"basecalling" => "Basecalling",
		"alignment" => "Aligning_Reads"
		);
		
	$progress_status_meaning = array( "green" => false,
		"yellow" => true,
		"grey" => true
		);
		
	$data = NULL;
    if(file_exists('progress.txt'))
    {
        $file = fopen('progress.txt', 'r') or die();	
        $data = array();
        while(($line=fgets($file)) !== false)
        {	
            list($key,$value) = explode("=", $line);
			$full_name = $progress_task_name[trim($key)];
            $data[$full_name] = $progress_status_meaning[trim($value)] ;
        }

        fclose($file);
    }
	
	header('Cache-Control: no-cache, must-revalidate');
	header('Expires: Mon, 26 Jul 1997 05:00:00 GMT');
	header('Content-type: application/json');
	echo json_encode($data);
}
?>

<?php 
	if(isset($_GET['log'])){
		echo "<html><body><pre>";
		$sigproc_log = file_get_contents("sigproc_results/sigproc.log");
		echo htmlspecialchars($sigproc_log);
		$report_log = file_get_contents("ReportLog.html");
		echo htmlspecialchars($report_log);
		echo "</pre></body></html>";
	}



 function get2DArrayFromCsv($file,$delimiter) {
        if (($handle = fopen($file, "r")) !== FALSE) {

         $i = 0;

       $header = fgetcsv($handle, 4000, $delimiter);
      while (($lineArray = fgetcsv($handle, 4000, $delimiter)) !== FALSE) {
        for ($j=0; $j<count($lineArray); $j++) {

                //if($i==0)
                //$data2DArray[$header[$j]][$i]="";
                //else
                //{ 
		$data2DArray[$header[$j]][$i]=$lineArray[$j];
			if($data2DArray["Sequence"][$i]=="")
				{ $data2DArray["Sequence"][$i]="No Matching Sequence found";}
                
               }
                $i=$i+1;
            }
            fclose($handle);
        }


        return $data2DArray;
}
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



?>
