<?php
$dataFile = $_GET['dataFile'];
$rows = $_GET['rows'];

$dtfh = fopen($dataFile,"r");
if( $dtfh == 0 || feof($dtfh) ) {
  print 'Could open data file '.$dataFile;
  exit(0);
}

$recNum = 0;
$line = "";
$i = 0;

$vcf_array = array();
array_push($vcf_array , "##fileformat=VCFv4.1");
array_push($vcf_array, "##reference=NA-NA-NA-NA", "##INFO=<ID=NA-NA-NA-NA>", "##FORMAT=<ID=NA-NA-NA-NA>", "#CHROM  POS NA-NA-NA-NA" );


while( ($j = strpos( $rows, ",", $i )) !== false ) {
  $r = substr( $rows, $i, $j-$i ) + 2;
  $i = $j+1;
  while( !feof($dtfh) )
  {
    $line = fgets($dtfh);
    ++$recNum;
    if( $recNum == $r ) break;
  }
  if( $recNum != $r ) break;
  $fields = explode("\t",$line);

//"#CHROM  POS ID  REF ALT QUAL    FILTER  INFO    FORMAT  Sample\n";

//chr - column 0
//pos - column 14
//ref - column 15
//alt - column 16

  $new_row =  $fields[0] . "\t" . $fields[14] . "\t . \t" . $fields[15] . "\t" . $fields[16] . "\t.\t.\t.\t.\t." ;
  array_push($vcf_array , $new_row);
}

foreach($vcf_array as $val) {
//    print $val . "\n";
}

?>

<html>
<head><title>Test Form for TVC to CE Primer Search</title></head>
<body>
<h2>Re-directing to Life Technologies PCR/Sanger Sequencing primer design page ...</h2>

    <form id='search_form' action="https://www.thermofisher.com/order/genome-database/MultipleTargets" method="POST" enctype="multipart/form-data">
        <div style='display:none'>
            <input type="hidden" name="productTypeSelect" value="ceprimer" />
            <input type="hidden" name="species" value="Homo sapiens" />
            <input type="hidden" id="batchTextArea" name="batchText" value="<?php foreach($vcf_array as $val) { print $val . "\n";}?>" />
            <br /><br />
            <input type="hidden" name="CID" value="ION2CEPRIMER" />
            <input type="hidden" name="btnSearch" value="y" />
            <input type="Submit"></input>
        </div>
    </form>
<script language="javascript" type="text/javascript">
    document.getElementById('search_form').submit();
</script>
</body>
</html>


