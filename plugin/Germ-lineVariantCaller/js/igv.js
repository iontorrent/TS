//igvLink needs to be written dynamically

$(function() {
    newURL = window.location.protocol + "//" + window.location.host ;
    pathArray = window.location.pathname.split( '/' );
    newPathname = "";
        for ( i = 0; i < pathArray.length-1; i++ ) {
         if (i > 0){
          newPathname += "/";
        }
          newPathname += pathArray[i];
        }

    igvURL = (newURL + newPathname + "/igv.php3");

    $('#igvLink').attr('href','http://www.broadinstitute.org/igv/projects/current/igv.php?sessionURL=' + igvURL);

    $('.igvTable').each(function(){
        $(this).attr('href','http://www.broadinstitute.org/igv/projects/current/igv.php?locus=' + $(this).data("locus") + '&sessionURL=' + igvURL)
    })

});

