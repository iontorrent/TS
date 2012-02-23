<?php
$ext = isset($_REQUEST['ext']) ? $_REQUEST['ext'] : false ;
$dir = isset($_REQUEST['dir']) ? $_REQUEST['dir'] : "." ;
$files = array();

function endswith($string, $test) {
    $strlen = strlen($string);
    $testlen = strlen($test);
    if ($testlen > $strlen) return false;
    return substr_compare($string, $test, -$testlen) === 0;
}

if ($handle = opendir($dir)) {
    while (false !== ($file = readdir($handle))) {
        if ($file != "." && $file != "..") {
            if ($ext){
                if (endswith($file,$ext)){
                    array_push($files,$file);
                }
            }else{
                    array_push($files,$file);
            }
        }
    }
    closedir($handle);
    print(json_encode($files));
}
?>

