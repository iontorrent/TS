<?php
    //PHP script contains methods useful in the Alignment_block.php and Alignment.php


    //method to convert a number into a decimal ??
    function local_try_number_format($value, $decimals = 0){
                if (is_numeric($value)){
                        return number_format($value, $decimals);
                }else{
                        return $value;
                }
        }


     // Function to convert tab seperated txt file into Array
    function local_tabs_parse_to_keys($handle){
    $arr = NULL;
    $row = 1;

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


     return $arr;
    }


    //Parse the alignment.summary file into key,value pairs
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



?>
