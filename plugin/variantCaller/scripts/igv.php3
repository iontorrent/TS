<?php

//get the vars to we need for the rewritter
$host = $_SERVER["HTTP_HOST"];
$page_url = parse_url($_SERVER['REQUEST_URI'],PHP_URL_PATH) . "";

//just get the path of the script
$break = explode('/', $page_url);
$file = array_pop($break);

# Force Basic Auth by using /auth/ variant of urls. TS-5802
if ($break[1] != 'auth') {
    # explode creates array with leading empty element
    $leading = array_shift($break);
    array_unshift($break, 'auth');
    array_unshift($break, $leading);
}

$plugin_out = implode('/', $break);
// move to the results directory
while (0 < count($break)) {
        $dir = array_pop($break);
        if ($dir == "plugin_out") {
                break;
        }
}
$results = implode('/', $break);

$url_root = "http://" . $host . $results;
$plugin_url = "http://" . $host . $plugin_out;

//read the igv file
$igv_session = file_get_contents('./igv_session.xml', true);

//replace the vars
$igv_session = str_replace("{url_root}", $url_root , $igv_session);
$igv_session = str_replace("{plugin_url}", $plugin_url , $igv_session);

//print out the rewritten igv
header("Content-type: text/xml");
header("Content-disposition: attachment; filename=igv_session.xml");
print $igv_session;

?>
