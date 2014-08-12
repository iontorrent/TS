<?php

//get the vars to we need for the rewritter
$host = $_SERVER["HTTP_HOST"];
$page_url = parse_url($_SERVER['REQUEST_URI'],PHP_URL_PATH) . "";

//just get the path of the script
$break = explode('/',$page_url);
$file = array_pop($break);
$plugin_out = implode('/',$break);
// move to the results directory
while( count($break) ) {
  $dir = array_pop($break);
  if( $dir == "plugin_out" ) break;
}
$results = implode('/',$break);

$http_host  = "http://".$host;
$url_root   = $http_host.$results;
$plugin_url = $http_host.$plugin_out;

//read the igv file
$igv_session = file_get_contents('./igv_session.xml',true);

//replace the vars
$igv_session = str_replace("{http_host}", $http_host, $igv_session);
$igv_session = str_replace("{url_root}", $url_root, $igv_session);
$igv_session = str_replace("{plugin_url}", $plugin_url, $igv_session);

//print out the rewritten igv
header("Content-type: text/xml");
print $igv_session;

?>
